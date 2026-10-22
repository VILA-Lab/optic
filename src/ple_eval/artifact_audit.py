from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from .compact_adapter import compact_resident_bytes, load_compact_ple
from .config import file_sha256, load_method, load_model, model_config_binding, write_json_atomic


RECON_KEY = "embed_tokens_per_layer.weight"
CHECKPOINT_PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"


HASH_INPUTS = (
    ("artifact_sha256", "compressed_artifact_path", "compressed_artifact_sha256"),
    ("reconstruction_sha256", "reconstruction_path", "reconstruction_sha256"),
    ("builder_sha256", "builder_path", "builder_sha256"),
    ("online_adapter_sha256", "online_adapter_path", "online_adapter_sha256"),
    ("roundtrip_verifier_sha256", "roundtrip_verifier_path", "roundtrip_verifier_sha256"),
    ("training_metrics_sha256", "training_metrics_path", "training_metrics_sha256"),
    ("source_artifact_sha256", "source_artifact_path", "source_artifact_sha256"),
    ("source_builder_sha256", "source_builder_path", "source_builder_sha256"),
    ("source_training_metrics_sha256", "source_training_metrics_path", "source_training_metrics_sha256"),
    ("source_model_weights_sha256", "source_model_weights_path", "source_model_weights_sha256"),
    ("transformation_receipt_sha256", "transformation_receipt_path", "transformation_receipt_sha256"),
)


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("ple_eval_optic_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash_inputs(method: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    actual: dict[str, str] = {}
    expected: dict[str, str] = {}
    for receipt_key, path_key, hash_key in HASH_INPUTS:
        path_value = method.get(path_key)
        hash_value = method.get(hash_key)
        if path_value is None and hash_value is None:
            continue
        if not isinstance(path_value, str) or not isinstance(hash_value, str):
            raise RuntimeError(f"method config must provide both {path_key} and {hash_key}")
        actual[receipt_key] = file_sha256(Path(path_value))
        expected[receipt_key] = hash_value
    return actual, expected




def _update_nmse(torch: Any, decoded: Any, reference: Any) -> tuple[float, float]:
    diff = decoded.to(torch.float32) - reference.to(torch.float32)
    return diff.double().square().sum().item(), reference.to(torch.float32).double().square().sum().item()


def _compare_compact_reconstruction(
    *,
    torch: Any,
    safe_open: Any,
    module: Any,
    reconstruction: Path,
    reference: Path,
    device: Any,
    chunk: int,
) -> tuple[float, float]:
    """Stream a complete compact decode and compare it to capability/reference tensors."""
    max_abs_diff = 0.0
    squared_error = 0.0
    squared_norm = 0.0
    with safe_open(str(reconstruction), framework="pt", device="cpu") as recon_file:
        with safe_open(str(reference), framework="pt", device="cpu") as reference_file:
            if RECON_KEY not in set(recon_file.keys()):
                raise RuntimeError(f"missing reconstruction key {RECON_KEY!r}")
            if CHECKPOINT_PLE_KEY not in set(reference_file.keys()):
                raise RuntimeError(f"missing checkpoint PLE key {CHECKPOINT_PLE_KEY!r}")
            recon_slice = recon_file.get_slice(RECON_KEY)
            reference_slice = reference_file.get_slice(CHECKPOINT_PLE_KEY)
            expected_shape = list(module.ple_shape)
            if list(recon_slice.get_shape()) != expected_shape:
                raise RuntimeError(f"reconstruction shape {list(recon_slice.get_shape())} != {expected_shape}")
            if list(reference_slice.get_shape()) != expected_shape:
                raise RuntimeError(f"checkpoint PLE shape {list(reference_slice.get_shape())} != {expected_shape}")

            # The output-weighted PCA builder used one full-layer GEMM,
            # followed (for wrappers) by a BF16 source materialization and a
            # coordinate-wise calibration. Reproduce that exact operation order.
            if hasattr(getattr(module, "source", module), "decode_layer"):
                ids = torch.arange(module.vocab_size, device=device, dtype=torch.long)
                source = getattr(module, "source", module)
                for layer in range(module.layers):
                    decoded_layer = source.decode_layer(ids, layer)
                    for start in range(0, module.vocab_size, chunk):
                        stop = min(module.vocab_size, start + chunk)
                        decoded = decoded_layer[start:stop].to(torch.float32)
                        if source is not module:
                            decoded = decoded * module.layer_coordinate_scale[layer]
                            if module.has_bias:
                                decoded = decoded + module.layer_coordinate_bias[layer]
                        decoded = decoded.to(torch.bfloat16).cpu()
                        columns = slice(layer * module.d_ple, (layer + 1) * module.d_ple)
                        saved = recon_slice[start:stop, columns]
                        reference_rows = reference_slice[start:stop, columns]
                        diff = float((decoded.float() - saved.float()).abs().max().item())
                        max_abs_diff = max(max_abs_diff, diff)
                        err, norm = _update_nmse(torch, saved, reference_rows)
                        squared_error += err
                        squared_norm += norm
                return max_abs_diff, squared_error / max(squared_norm, 1e-30)

            for start in range(0, module.vocab_size, chunk):
                stop = min(module.vocab_size, start + chunk)
                ids = torch.arange(start, stop, device=device, dtype=torch.long)
                decoded = module(ids).to(device="cpu", dtype=torch.bfloat16)
                saved = recon_slice[start:stop]
                reference_rows = reference_slice[start:stop]
                diff = float((decoded.float() - saved.float()).abs().max().item())
                max_abs_diff = max(max_abs_diff, diff)
                err, norm = _update_nmse(torch, saved, reference_rows)
                squared_error += err
                squared_norm += norm
    return max_abs_diff, squared_error / max(squared_norm, 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch
    from safetensors import safe_open

    method = load_method(args.method)
    selected_model_config = args.model_config or method.get("source_model_config")
    model = load_model(selected_model_config)
    selected_model_binding = model_config_binding(selected_model_config)
    reference = Path(model["snapshot"]) / model["weight_file"]
    if method.get("source_model_revision") is not None and method["source_model_revision"] != model["revision"]:
        raise RuntimeError("artifact audit source checkpoint revision differs from the fitting source")
    if method.get("source_model_weights_path") is not None and not Path(method["source_model_weights_path"]).samefile(reference):
        raise RuntimeError("artifact audit checkpoint path differs from the exact fitting weights")
    if method["capability_mode"] != "dense_reconstruction_injection":
        raise RuntimeError("roundtrip audit applies only to reconstruction arms")
    artifact = Path(method["compressed_artifact_path"])
    reconstruction = Path(method["reconstruction_path"])
    actual_hashes, expected_hashes = _hash_inputs(method)
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"artifact input hash mismatch: {actual_hashes}")

    with np.load(artifact, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    with safe_open(str(reference), framework="pt", device="cpu") as checkpoint:
        checkpoint_shape = list(checkpoint.get_slice(CHECKPOINT_PLE_KEY).get_shape())
    format_id = method.get("format_id", metadata["compressed_ple_format_id"])
    required = {"compressed_ple_format_id": format_id, "ple_shape": checkpoint_shape}
    if "full_ple_coverage" in metadata:
        required["full_ple_coverage"] = True
    if "hidden_raw_copy_allowed" in metadata:
        required["hidden_raw_copy_allowed"] = False
    if "seed" in metadata:
        required["seed"] = method["seed"]
    if "storage_bytes" in metadata:
        required["storage_bytes"] = method["ple_resident_bytes"]
    mismatches = {key: metadata.get(key) for key, value in required.items() if metadata.get(key) != value}
    if mismatches:
        raise RuntimeError(f"compressed artifact metadata mismatch: {mismatches}")

    started = time.perf_counter()
    device = torch.device(args.device)
    module = load_compact_ple(
        artifact,
        device=device,
        source_overrides=(
            {method["source_format_id"]: method["source_artifact_path"]}
            if method.get("source_format_id") and method.get("source_artifact_path")
            else None
        ),
    )
    resident_bytes = compact_resident_bytes(module)
    if resident_bytes != method["ple_resident_bytes"]:
        raise RuntimeError(f"compact resident bytes {resident_bytes} != {method['ple_resident_bytes']}")

    online_max_abs_diff, reconstruction_nmse = _compare_compact_reconstruction(
        torch=torch,
        safe_open=safe_open,
        module=module,
        reconstruction=reconstruction,
        reference=reference,
        device=device,
        chunk=int(method.get("audit_chunk_rows", args.chunk)),
    )

    # Use the OPTIC builder's own verifier. It checks the builder's
    # layer-at-a-time reconstruction independently.
    builder_max_abs_diff = None
    if format_id == "optic_npz":
        verifier_path = Path(method.get("roundtrip_verifier_path", method.get("builder_path")))
        verifier = load_builder(verifier_path)
        if method.get("roundtrip_verifier_method_from_artifact", False):
            verifier.METHOD_ID = str(metadata["method"])
        builder_max_abs_diff = verifier.verify_resident_roundtrip(
            torch,
            np,
            safe_open,
            artifact,
            reconstruction,
            int(metadata["vocab_size"]),
            int(metadata["num_layers"]),
            int(metadata["d_ple"]),
            str(metadata["latent_mode"]),
            int(metadata["latent_dim"]),
            int(metadata["n_groups"]),
            int(metadata["codebook_size"]),
            int(metadata["storage_bytes"]),
            device,
            int(method.get("audit_chunk_rows", args.chunk)),
            float(method.get("roundtrip_atol", 0.0)),
        )

    tolerance = float(method.get("roundtrip_atol", 0.0))
    online_tolerance = float(method.get("online_roundtrip_atol", tolerance))
    max_abs_diff = builder_max_abs_diff if builder_max_abs_diff is not None else online_max_abs_diff
    if max_abs_diff > tolerance:
        raise RuntimeError(f"capability roundtrip max_abs_diff {max_abs_diff} exceeds {tolerance}")
    if online_max_abs_diff > online_tolerance:
        raise RuntimeError(
            f"online compact roundtrip max_abs_diff {online_max_abs_diff} exceeds {online_tolerance}"
        )
    receipt = {
        "schema_version": "ple-artifact-roundtrip-v1",
        "status": "passed",
        "method": args.method,
        "method_id": method["method_id"],
        "actual_hashes": actual_hashes,
        "model_config": selected_model_binding,
        "source_model_revision": model["revision"],
        "metadata": metadata,
        "roundtrip": {
            "source_artifact": str(artifact),
            "dense_reconstruction": str(reconstruction),
            "device": args.device,
            "chunk_rows": int(method.get("audit_chunk_rows", args.chunk)),
            "max_abs_diff_after_bfloat16_materialization": max_abs_diff,
            "builder_max_abs_diff_after_bfloat16_materialization": builder_max_abs_diff,
            "online_compact_max_abs_diff_against_capability_materialization": online_max_abs_diff,
            "online_compact_tolerance": online_tolerance,
            "reconstruction_nmse_against_checkpoint_ple": reconstruction_nmse,
            "tolerance": tolerance,
            "persistent_resident_bytes": resident_bytes,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(args.output, receipt)
    print(json.dumps({"status": "passed", "method": args.method, "max_abs_diff": max_abs_diff}))


if __name__ == "__main__":
    main()
