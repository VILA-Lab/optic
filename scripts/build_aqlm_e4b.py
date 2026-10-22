#!/usr/bin/env python3
"""Build a 100x AQLM Euclidean beam-encoder adaptation for Gemma 4 PLE.

The build starts from a fixed additive-VQ initialization and then runs the
L2 beam-search implementation of the AQLM repository to update every additive
code tuple.  The adaptation changes AQLM's small linear-weight groups to one
256-dimensional PLE vector and omits per-vector scales, so the complete online
representation stays within the 100x budget.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_BUILDERS = ROOT / "builders" / "baselines"
sys.path.insert(0, str(BASELINE_BUILDERS))

from baseline_common import PLE_KEY, RECON_KEY  # noqa: E402
from baseline_residual_vq import pack_nbit, unpack_nbit  # noqa: E402


FORMAT_ID = "joint_refined_additive_vq_npz"
METHOD_ID = "aqlm_l2_beam_objective_ple_adaptation"
EXPECTED_AQLM_COMMIT = "e79a896ed6656fe4ed06193d42d004e7d0bbdbb2"
EXPECTED_BEAM_SEARCH_SHA256 = "8dd39717dc9e44f7c8c8dba1426e616333878418dbe950ebeb7596c15848bdbf"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_author_beam_search(aqlm_root: Path):
    source = aqlm_root / "src" / "beam_search_l2.py"
    require(source.is_file(), f"missing author AQLM beam search: {source}")
    require(
        sha256_file(source) == EXPECTED_BEAM_SEARCH_SHA256,
        "author AQLM beam_search_l2.py differs from the audited commit",
    )
    head = aqlm_root / ".git" / "HEAD"
    if head.is_file():
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=aqlm_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        require(commit == EXPECTED_AQLM_COMMIT, f"AQLM commit differs: {commit}")
    sys.path.insert(0, str(aqlm_root))
    spec = importlib.util.spec_from_file_location("audited_aqlm_beam_search_l2", source)
    require(spec is not None and spec.loader is not None, "cannot load AQLM beam-search module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.beam_search_optimal_codes, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--it-weights", required=True)
    parser.add_argument("--initialization-artifact", required=True)
    parser.add_argument("--aqlm-root", required=True)
    parser.add_argument("--num-layers", type=int, default=42)
    parser.add_argument("--d-ple", type=int, default=256)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--row-chunk", type=int, default=16384)
    parser.add_argument("--author-chunk-size-bytes", type=int, default=536870912)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-recon", required=True)
    parser.add_argument("--save-compressed-ple-npz", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    require(args.beam_size == 2, "the AQLM adaptation requires beam size 2")
    require(args.seed == 0, "the AQLM adaptation requires seed 0")
    torch.manual_seed(args.seed)
    started = time.time()
    device = torch.device(args.device)
    beam_search, author_source = load_author_beam_search(Path(args.aqlm_root))

    with safe_open(args.it_weights, framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(PLE_KEY).to(torch.float32)
    vocab = int(packed.shape[0])
    expected_shape = (vocab, args.num_layers * args.d_ple)
    require(tuple(packed.shape) == expected_shape, f"checkpoint PLE shape {tuple(packed.shape)} != {expected_shape}")
    reference = (
        packed.reshape(vocab, args.num_layers, args.d_ple)
        .permute(1, 0, 2)
        .reshape(-1, args.d_ple)
        .contiguous()
    )
    del packed
    rows = int(reference.shape[0])

    initialization_path = Path(args.initialization_artifact)
    with np.load(initialization_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        require(
            metadata.get("compressed_ple_format_id") == FORMAT_ID,
            "initialization is not the audited local additive-VQ artifact",
        )
        codebooks_np = archive["codebooks"].copy()
        m_count, clusters, width = codebooks_np.shape
        bits = int(metadata["bits"])
        require((m_count, clusters, width, bits) == (5, 256, args.d_ple, 8), "initialization geometry differs")
        codes = np.stack(
            [unpack_nbit(archive[f"codes_m{stage:02d}"], bits, rows).astype(np.uint16) for stage in range(m_count)],
            axis=1,
        )

    codebooks_resident = codebooks_np.astype(np.float16, copy=False)
    codebooks_device = torch.from_numpy(codebooks_resident).to(device=device, dtype=torch.float32).reshape(
        m_count, clusters, 1, args.d_ple
    )
    output_codes = np.empty_like(codes)
    recon_packed = torch.empty(expected_shape, dtype=torch.bfloat16)
    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    total_before = 0.0
    total_after = 0.0
    total_after_bf16 = 0.0
    total_norm = 0.0
    changed = 0
    objective_increased_chunks = 0
    maximum_objective_sse_increase = 0.0
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for start in range(0, rows, args.row_chunk):
            stop = min(rows, start + args.row_chunk)
            target = reference[start:stop].to(device)
            old = torch.from_numpy(codes[start:stop].astype(np.int64, copy=False)).to(device)
            old_grouped = old[:, None, :]
            new_grouped = beam_search(
                reference_weight=target,
                codebooks=codebooks_device,
                prev_codes=old_grouped,
                scales=None,
                beam_size=args.beam_size,
                stochastic_rounding_tau=0.0,
                chunk_size_bytes=args.author_chunk_size_bytes,
                dim_rng=None,
                force_update=False,
            )
            new = new_grouped[:, 0, :]
            offsets = torch.arange(m_count, device=device) * clusters
            flat_books = codebooks_device.flatten(0, 1).flatten(-2, -1)
            old_recon = torch.nn.functional.embedding_bag(old + offsets, flat_books, mode="sum")
            new_recon = torch.nn.functional.embedding_bag(new + offsets, flat_books, mode="sum")
            before = (target - old_recon).double().square().sum().item()
            after = (target - new_recon).double().square().sum().item()
            after_bf16 = (target - new_recon.to(torch.bfloat16).to(torch.float32)).double().square().sum().item()
            norm = target.double().square().sum().item()
            objective_delta = after - before
            objective_increased = objective_delta > max(1e-6, 1e-10 * before)
            if objective_increased:
                objective_increased_chunks += 1
                maximum_objective_sse_increase = max(maximum_objective_sse_increase, objective_delta)
            changed_chunk = int((new != old).any(dim=1).sum().item())
            output_codes[start:stop] = new.to(device="cpu", dtype=torch.uint16).numpy()
            for layer in range(args.num_layers):
                layer_start = layer * vocab
                left = max(start, layer_start)
                right = min(stop, layer_start + vocab)
                if left >= right:
                    continue
                source_slice = slice(left - start, right - start)
                token_slice = slice(left - layer_start, right - layer_start)
                col_slice = slice(layer * args.d_ple, (layer + 1) * args.d_ple)
                recon_packed[token_slice, col_slice] = new_recon[source_slice].to(torch.bfloat16).cpu()
            record = {
                "row_start": start,
                "row_stop": stop,
                "objective_sse_before": before,
                "objective_sse_after": after,
                "objective_sse_delta": objective_delta,
                "objective_increased": objective_increased,
                "bf16_reconstruction_sse_after": after_bf16,
                "reference_squared_norm": norm,
                "changed_vectors": changed_chunk,
            }
            raw_handle.write(json.dumps(record, sort_keys=True) + "\n")
            raw_handle.flush()
            total_before += before
            total_after += after
            total_after_bf16 += after_bf16
            total_norm += norm
            changed += changed_chunk
            print(
                f"[aqlm-ple] rows={start}:{stop} before_nmse={before / max(norm, 1e-30):.6f} "
                f"after_nmse={after / max(norm, 1e-30):.6f} changed={changed_chunk}",
                flush=True,
            )

    recon_path = Path(args.save_recon)
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({RECON_KEY: recon_packed.contiguous()}, recon_path)

    code_bytes = math.ceil(rows * m_count * bits / 8)
    codebook_bytes = int(codebooks_resident.nbytes)
    resident_bytes = code_bytes + codebook_bytes
    dense_bytes = rows * args.d_ple * 2
    artifact_metadata = {
        "compressed_ple_format_id": FORMAT_ID,
        "method": METHOD_ID,
        "kind": METHOD_ID,
        "schema_version": "1.0",
        "full_ple_coverage": True,
        "M": m_count,
        "K": clusters,
        "bits": bits,
        "uses_coordinate_weights": False,
        "uses_per_vector_scales": False,
        "ple_key": PLE_KEY,
        "ple_shape": list(expected_shape),
        "storage_bytes": resident_bytes,
        "seed": args.seed,
        "initialization": "local_two_sweep_block_coordinate_additive_vq",
        "objective_update": "author_aqlm_l2_beam_search",
        "beam_size": args.beam_size,
        "aqlm_commit": EXPECTED_AQLM_COMMIT,
        "aqlm_beam_search_sha256": EXPECTED_BEAM_SEARCH_SHA256,
        "calibration_corpus_used": False,
    }
    payload = {"metadata": np.array(json.dumps(artifact_metadata)), "codebooks": codebooks_resident}
    for stage in range(m_count):
        payload[f"codes_m{stage:02d}"] = pack_nbit(output_codes[:, stage], bits)
    artifact_path = Path(args.save_compressed_ple_npz)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(artifact_path, **payload)

    summary = {
        **artifact_metadata,
        "classification": "adapted_baseline",
        "source_ple": args.it_weights,
        "source_ple_sha256": sha256_file(args.it_weights),
        "initialization_artifact": str(initialization_path),
        "initialization_artifact_sha256": sha256_file(initialization_path),
        "author_source_path": str(author_source),
        "raw_output_path": str(raw_path),
        "raw_output_sha256": sha256_file(raw_path),
        "objective_nmse_before": total_before / max(total_norm, 1e-30),
        "objective_nmse_after_fp32_decode": total_after / max(total_norm, 1e-30),
        "decoded_source_ple_nmse": total_after_bf16 / max(total_norm, 1e-30),
        "changed_vector_count": changed,
        "changed_vector_fraction": changed / rows,
        "objective_increased_chunk_count": objective_increased_chunks,
        "maximum_objective_sse_increase": maximum_objective_sse_increase,
        "code_bytes": code_bytes,
        "codebook_bytes": codebook_bytes,
        "resident_bytes": resident_bytes,
        "dense_reference_bytes": dense_bytes,
        "compression_ratio": dense_bytes / resident_bytes,
        "dense_reconstruction_path": str(recon_path),
        "dense_reconstruction_sha256": sha256_file(recon_path),
        "compressed_artifact_path": str(artifact_path),
        "compressed_artifact_sha256": sha256_file(artifact_path),
        "compressed_artifact_file_bytes": artifact_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[aqlm-ple] objective NMSE {summary['objective_nmse_before']:.6f} -> "
        f"{summary['objective_nmse_after_fp32_decode']:.6f}; resident={resident_bytes}; "
        f"ratio={summary['compression_ratio']:.6f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
