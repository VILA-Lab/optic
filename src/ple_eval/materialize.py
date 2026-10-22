"""Decode a compressed PLE artifact to a BF16 embedding table."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .compact_adapter import artifact_sha256, compact_resident_bytes, load_compact_ple
from .config import write_json_atomic


RECON_KEY = "embed_tokens_per_layer.weight"


def materialize(
    artifact: Path,
    output: Path,
    *,
    device: str,
    chunk_rows: int,
    source_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    import torch
    from safetensors.torch import save_file

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing reconstruction: {output}")
    module = load_compact_ple(artifact, device=device, source_overrides=source_overrides)
    started = time.perf_counter()
    reconstruction = torch.empty(module.ple_shape, dtype=torch.bfloat16, device="cpu")

    def fill_exact(current, destination) -> None:
        if current.format_id == "dout_weighted_pca_rotated_pq_scaled_exceptions_npz":
            # The builder formed one complete layer GEMM at a time.
            ids = torch.arange(current.vocab_size, device=device, dtype=torch.long)
            for layer in range(current.layers):
                column = slice(layer * current.d_ple, (layer + 1) * current.d_ple)
                destination[:, column] = current.decode_layer(ids, layer).to("cpu")
            return
        if current.format_id in {
            "affine_calibrated_pca_rotated_pq_scaled_exceptions_npz",
            "scale_calibrated_dout_weighted_pca_rotated_pq_scaled_exceptions_npz",
        }:
            source_reconstruction = torch.empty(current.source.ple_shape, dtype=torch.bfloat16, device="cpu")
            fill_exact(current.source, source_reconstruction)
            for layer in range(current.layers):
                column = slice(layer * current.d_ple, (layer + 1) * current.d_ple)
                scale = current.layer_coordinate_scale[layer].cpu()
                bias = (
                    current.layer_coordinate_bias[layer].cpu()
                    if current.has_bias
                    else torch.zeros_like(scale)
                )
                for start in range(0, current.vocab_size, chunk_rows):
                    stop = min(current.vocab_size, start + chunk_rows)
                    source = source_reconstruction[start:stop, column].to(torch.float32)
                    destination[start:stop, column] = (source * scale + bias).to(torch.bfloat16)
            return
        for start in range(0, current.vocab_size, chunk_rows):
            stop = min(current.vocab_size, start + chunk_rows)
            ids = torch.arange(start, stop, device=device, dtype=torch.long)
            destination[start:stop] = current(ids).to(device="cpu", dtype=torch.bfloat16)

    with torch.inference_mode():
        if module.format_id == "optic_npz":
            # Match the builder's FP32, layer-at-a-time reconstruction.
            decoder = module.decoder_weight.reshape(module.layers, module.d_ple, module.latent_dim).float()
            bias = module.decoder_bias.reshape(module.layers, module.d_ple).float()
            for layer in range(module.layers):
                for start in range(0, module.vocab_size, chunk_rows):
                    stop = min(module.vocab_size, start + chunk_rows)
                    ids = torch.arange(start, stop, device=device, dtype=torch.long)
                    code_ids = module.token_codes.index_select(0, ids).to(torch.long) + module.group_offsets
                    latent = torch.nn.functional.embedding(
                        code_ids, module.codebooks.reshape(-1, module.group_dim)
                    ).reshape(stop - start, module.latent_dim)
                    decoded = latent.float() @ decoder[layer].T
                    decoded.add_(bias[layer])
                    column = slice(layer * module.d_ple, (layer + 1) * module.d_ple)
                    reconstruction[start:stop, column] = decoded.to(device="cpu", dtype=torch.bfloat16)
        else:
            fill_exact(module, reconstruction)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({RECON_KEY: reconstruction.contiguous()}, str(output))
    receipt = {
        "schema_version": "ple-materialization-v1",
        "status": "passed",
        "format_id": module.format_id,
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": artifact_sha256(artifact),
        "reconstruction_path": str(output.resolve()),
        "reconstruction_sha256": artifact_sha256(output),
        "ple_shape": list(module.ple_shape),
        "dtype": "torch.bfloat16",
        "persistent_resident_bytes": compact_resident_bytes(module),
        "chunk_rows": chunk_rows,
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--source-override",
        action="append",
        default=[],
        metavar="FORMAT_ID=PATH",
        help="Resolve a nested source artifact when its recorded absolute path is unavailable.",
    )
    args = parser.parse_args()
    overrides: dict[str, str] = {}
    for value in args.source_override:
        key, separator, path = value.partition("=")
        if not separator or not key or not path:
            raise SystemExit(f"invalid --source-override: {value!r}")
        overrides[key] = path
    receipt = materialize(
        args.artifact,
        args.output,
        device=args.device,
        chunk_rows=args.chunk_rows,
        source_overrides=overrides or None,
    )
    write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
