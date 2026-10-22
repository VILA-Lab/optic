#!/usr/bin/env python3
"""Check requested-row decoding for a newly fitted compact artifact."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from ple_eval.compact_adapter import (
    artifact_sha256,
    compact_resident_bytes,
    load_compact_ple,
    read_compressed_npz,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "src/ple_eval/compact_adapter.py"
OUTPUT_ROOT = ROOT / "receipts/compact-requested-row-receipt-refresh"
REQUESTED_TOKEN_IDS = (0, 1, 2, 1024, 131072, 258884)



def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _tensor_record(name: str, value: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "numel": int(value.numel()),
        "element_size": int(value.element_size()),
        "nbytes": int(value.numel() * value.element_size()),
    }


def resident_census(module: torch.nn.Module) -> dict[str, Any]:
    persistent: list[dict[str, Any]] = []
    non_persistent: list[dict[str, Any]] = []
    parameters = [_tensor_record(name, value) for name, value in module.named_parameters()]

    def visit(current: torch.nn.Module, prefix: str) -> None:
        excluded = getattr(current, "_non_persistent_buffers_set", set())
        for name, value in current.named_buffers(recurse=False):
            if value is None:
                continue
            qualified = f"{prefix}.{name}" if prefix else name
            target = non_persistent if name in excluded else persistent
            target.append(_tensor_record(qualified, value))
        for name, child in current.named_children():
            visit(child, f"{prefix}.{name}" if prefix else name)

    visit(module, "")
    persistent_bytes = sum(int(row["nbytes"]) for row in persistent)
    parameter_bytes = sum(int(row["nbytes"]) for row in parameters)
    return {
        "accounting_semantics": "actual loaded persistent tensors; derived non-persistent buffers excluded",
        "persistent_buffers": persistent,
        "persistent_buffer_bytes": persistent_bytes,
        "parameters": parameters,
        "parameter_bytes": parameter_bytes,
        "total_persistent_tensor_bytes": persistent_bytes + parameter_bytes,
        "excluded_non_persistent_buffers": non_persistent,
        "excluded_non_persistent_buffer_bytes": sum(int(row["nbytes"]) for row in non_persistent),
    }


def decode_codes(packed: np.ndarray, indices: np.ndarray, bits: int) -> np.ndarray:
    """Independent little-endian packed-code decoder for selected values."""
    flat = np.asarray(packed, dtype=np.uint8).reshape(-1)
    values_array = np.asarray(indices, dtype=np.int64)
    value_shape = values_array.shape
    values = values_array.reshape(-1)
    offsets = values * int(bits)
    byte_offsets = offsets // 8
    shifts = offsets % 8
    padded = np.pad(flat, (0, 2), mode="constant")
    combined = (
        padded[byte_offsets].astype(np.uint32)
        | (padded[byte_offsets + 1].astype(np.uint32) << 8)
        | (padded[byte_offsets + 2].astype(np.uint32) << 16)
    )
    return ((combined >> shifts) & ((1 << bits) - 1)).astype(np.int64).reshape(value_shape)


def independent_selected_rows(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray], token_ids: tuple[int, ...]
) -> torch.Tensor:
    """Decode only selected rows without invoking the production adapter."""
    format_id = str(metadata["compressed_ple_format_id"])
    vocab, width = (int(value) for value in metadata["ple_shape"])
    ids = np.asarray(token_ids, dtype=np.int64)

    if format_id == "packed_pq_npz":
        codebooks = arrays["codebooks"]
        d_ple = int(codebooks.shape[0] * codebooks.shape[2])
        layers = width // d_ple
        row_ids = ids[:, None] + np.arange(layers, dtype=np.int64)[None, :] * vocab
        parts = []
        for subspace in range(int(metadata["num_subspaces"])):
            codes = decode_codes(arrays[f"codes_m{subspace:02d}"], row_ids, int(metadata["bits"]))
            parts.append(codebooks[subspace][codes])
        decoded = np.concatenate(parts, axis=-1).reshape(len(ids), width)
        return torch.from_numpy(decoded.copy()).to(torch.bfloat16)

    if format_id in {"residual_vq_npz", "joint_refined_additive_vq_npz"}:
        codebooks = arrays["codebooks"]
        d_ple = int(codebooks.shape[2])
        layers = width // d_ple
        row_ids = ids[:, None] + np.arange(layers, dtype=np.int64)[None, :] * vocab
        decoded = torch.zeros((len(ids), layers, d_ple), dtype=torch.float32)
        for codebook in range(int(metadata["M"])):
            codes = decode_codes(
                arrays[f"codes_m{codebook:02d}"], row_ids, int(metadata["bits"])
            )
            decoded.add_(torch.from_numpy(codebooks[codebook][codes]).to(torch.float32))
        if bool(metadata.get("uses_coordinate_weights")):
            scales = torch.from_numpy(arrays["coord_weights_fp16"]).to(torch.float32).sqrt()
            decoded.div_(scales)
        return decoded.to(torch.bfloat16).reshape(len(ids), width)

    if format_id in {"parametric_opq_npz", "nonparametric_opq_npz"}:
        codebooks = arrays["codebooks_fp16"]
        d_ple = int(arrays["mean_fp16"].shape[0])
        layers = width // d_ple
        row_ids = ids[:, None] + np.arange(layers, dtype=np.int64)[None, :] * vocab
        parts = []
        for subspace in range(int(metadata.get("num_subspaces") or metadata["subspaces"])):
            bits = int(metadata.get("bits_per_code") or metadata["bits"])
            codes = decode_codes(arrays[f"codes_m{subspace:02d}"], row_ids, bits)
            parts.append(torch.from_numpy(codebooks[subspace][codes]).to(torch.float32))
        latent = torch.cat(parts, dim=-1).reshape(-1, d_ple)
        rotation_t = torch.from_numpy(arrays["rotation_fp16"]).to(torch.float32).T.contiguous()
        mean = torch.from_numpy(arrays["mean_fp16"]).to(torch.float32)
        decoded = (latent @ rotation_t + mean).reshape(len(ids), layers, d_ple)
        decoded.div_(torch.from_numpy(arrays["coord_scales_fp16"]).to(torch.float32))
        exception_ids = np.asarray(arrays["exception_token_ids_uint32"], dtype=np.int64)
        if exception_ids.size:
            residuals = torch.from_numpy(arrays["exception_scaled_residuals_fp16"]).to(torch.float32)
            positions = {int(token): index for index, token in enumerate(exception_ids.tolist())}
            for output_row, token in enumerate(ids.tolist()):
                if token in positions:
                    decoded[output_row].add_(residuals[positions[token]])
        return decoded.to(torch.bfloat16).reshape(len(ids), width)

    raise ValueError(f"independent selected-row decoder does not support {format_id!r}")


def dense_selected_rows(path: Path, token_ids: tuple[int, ...], expected_width: int) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if keys != ["embed_tokens_per_layer.weight"]:
            raise ValueError(f"unexpected capability reconstruction keys: {keys}")
        view = handle.get_slice(keys[0])
        shape = tuple(int(value) for value in view.get_shape())
        if shape[1] != expected_width:
            raise ValueError(f"capability width {shape[1]} != {expected_width}")
        return torch.cat([view[token : token + 1] for token in token_ids], dim=0).to(torch.bfloat16)


def existing_hash(path: str | None) -> str | None:
    candidate = Path(path) if path else None
    return sha256(candidate) if candidate is not None and candidate.is_file() else None


def refresh_one(
    config_path: Path,
    *,
    write: bool,
    overwrite: bool,
    provenance_before: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    artifact = Path(config["compressed_artifact_path"])
    if not artifact.is_file():
        raise FileNotFoundError(f"compact artifact is missing: {artifact}")
    observed_artifact_sha = artifact_sha256(artifact)
    if observed_artifact_sha != config["compressed_artifact_sha256"]:
        raise ValueError(f"compact artifact digest differs for {config['method_id']}")

    old_roundtrip = config.get("roundtrip_receipt_path")
    old_runtime = config.get("runtime_receipt_path")
    before = {
        "roundtrip_receipt_path": old_roundtrip,
        "roundtrip_receipt_sha256": existing_hash(old_roundtrip),
        "runtime_receipt_path": old_runtime,
        "runtime_receipt_sha256": existing_hash(old_runtime),
        "ple_resident_bytes": config.get("ple_resident_bytes"),
    }
    binding_before = provenance_before or before

    load_start = time.perf_counter()
    module = load_compact_ple(artifact, device="cpu")
    load_seconds = time.perf_counter() - load_start
    census = resident_census(module)
    measured_resident = int(census["total_persistent_tensor_bytes"])
    helper_resident = int(compact_resident_bytes(module))
    if measured_resident != helper_resident:
        raise ValueError("production helper differs from explicit loaded-tensor census")
    if census["parameter_bytes"] != 0:
        raise ValueError("compact adapter unexpectedly contains trainable/resident parameters")
    if measured_resident != int(config["ple_resident_bytes"]):
        raise ValueError(
            f"loaded persistent bytes {measured_resident} != method config {config['ple_resident_bytes']}"
        )

    metadata, arrays = read_compressed_npz(artifact)
    serialized_array_bytes = sum(int(value.nbytes) for value in arrays.values())
    if serialized_array_bytes != measured_resident:
        raise ValueError(
            f"serialized array bytes {serialized_array_bytes} != loaded persistent bytes {measured_resident}"
        )
    vocab, width = (int(value) for value in metadata["ple_shape"])
    if max(REQUESTED_TOKEN_IDS) >= vocab:
        raise ValueError("fixed requested-row ids exceed compact vocabulary")
    dense_numel = vocab * width
    if any(int(row["numel"]) >= dense_numel for row in census["persistent_buffers"]):
        raise ValueError("a dense-shaped persistent buffer is resident")

    ids = torch.tensor(REQUESTED_TOKEN_IDS, dtype=torch.long)
    decode_start = time.perf_counter()
    with torch.inference_mode():
        observed = module(ids)
    decode_seconds = time.perf_counter() - decode_start
    if tuple(observed.shape) != (len(REQUESTED_TOKEN_IDS), width):
        raise ValueError(f"unexpected requested-row output shape: {tuple(observed.shape)}")
    if not bool(torch.isfinite(observed.float()).all().item()):
        raise ValueError("requested-row output contains non-finite values")

    independent = independent_selected_rows(metadata, arrays, REQUESTED_TOKEN_IDS)
    independent_diff = float((observed.float() - independent.float()).abs().max().item())
    atol = float(config.get("roundtrip_atol", 0.0))
    if independent_diff > atol:
        raise ValueError(f"independent selected-row difference {independent_diff} > atol {atol}")

    reconstruction = Path(config["reconstruction_path"])
    if reconstruction.is_file():
        reference = dense_selected_rows(reconstruction, REQUESTED_TOKEN_IDS, width)
        reference_kind = "existing_bfloat16_capability_reconstruction_row_slices"
        dense_diff = float((observed.float() - reference.float()).abs().max().item())
        if dense_diff > atol:
            raise ValueError(f"capability reconstruction selected-row difference {dense_diff} > atol {atol}")
    else:
        reference = independent
        reference_kind = "independent_deterministic_selected_row_decoder_from_compact_arrays"
        dense_diff = None

    prior_roundtrip_binding = None
    prior_roundtrip_path = binding_before.get("roundtrip_receipt_path")
    if prior_roundtrip_path and Path(prior_roundtrip_path).is_file():
        prior_roundtrip = json.loads(Path(prior_roundtrip_path).read_text())
        prior_artifact_sha = prior_roundtrip.get("artifact_sha256")
        if prior_artifact_sha is not None and prior_artifact_sha != observed_artifact_sha:
            raise ValueError("prior round-trip receipt is bound to a different compact artifact")
        if prior_roundtrip.get("status") != "passed":
            raise ValueError("prior round-trip receipt did not pass")
        prior_roundtrip_binding = {
            "path": str(prior_roundtrip_path),
            "sha256": binding_before.get("roundtrip_receipt_sha256") or sha256(Path(prior_roundtrip_path)),
            "status": "passed",
            "artifact_sha256": prior_artifact_sha,
            "requested_token_ids": prior_roundtrip.get("requested_token_ids"),
            "reference_source": prior_roundtrip.get("reference_source")
            or prior_roundtrip.get("reconstruction_path"),
        }

    completed = utc_now()
    common = {
        "artifact_path": str(artifact),
        "artifact_sha256": observed_artifact_sha,
        "adapter_path": str(ADAPTER_PATH),
        "adapter_sha256": sha256(ADAPTER_PATH),
        "benchmark_inputs_used": False,
        "completed_utc": completed,
        "dense_ple_tensor_resident": False,
        "device": "cpu",
        "finite": True,
        "full_table_materialized": False,
        "method": config["method_id"],
        "output_shape": list(observed.shape),
        "persistent_resident_bytes": measured_resident,
        "persistent_tensor_census": census,
        "requested_rows": len(REQUESTED_TOKEN_IDS),
        "requested_token_ids": list(REQUESTED_TOKEN_IDS),
        "serialized_resident_array_bytes": serialized_array_bytes,
        "status": "passed",
    }
    roundtrip = {
        **common,
        "schema_version": "ple-compact-requested-row-roundtrip-v2",
        "atol": atol,
        "max_abs_diff_against_independent_decoder_after_bfloat16_materialization": independent_diff,
        "max_abs_diff_against_existing_capability_reconstruction_after_bfloat16_materialization": dense_diff,
        "reference_kind": reference_kind,
        "reconstruction_path": str(reconstruction),
        "reconstruction_present": reconstruction.is_file(),
        "reconstruction_sha256_bound_by_method_config": config.get("reconstruction_sha256"),
        "selected_rows_reconstructed_only": not reconstruction.is_file(),
        "prior_builder_roundtrip_binding": prior_roundtrip_binding,
    }
    runtime = {
        **common,
        "schema_version": "ple-compact-requested-row-runtime-v2",
        "module_load_seconds": load_seconds,
        "requested_row_decode_seconds": decode_seconds,
        "reference_kind": reference_kind,
    }

    roundtrip_path = artifact.parent / "requested_row_roundtrip_receipt_v2.json"
    runtime_path = artifact.parent / "requested_row_runtime_receipt_v2.json"
    if write:
        atomic_json(roundtrip_path, roundtrip, overwrite=overwrite)
        atomic_json(runtime_path, runtime, overwrite=overwrite)
        config["ple_resident_bytes"] = measured_resident
        config["roundtrip_receipt_path"] = str(roundtrip_path)
        config["roundtrip_receipt_sha256"] = sha256(roundtrip_path)
        config["runtime_receipt_path"] = str(runtime_path)
        config["runtime_receipt_sha256"] = sha256(runtime_path)
        atomic_json(config_path, config, overwrite=True)

    result = {
        "method_id": config["method_id"],
        "method_config_path": str(config_path),
        "status": "passed",
        "before": before,
        "after": {
            "roundtrip_receipt_path": str(roundtrip_path),
            "roundtrip_receipt_sha256": sha256(roundtrip_path) if write else None,
            "runtime_receipt_path": str(runtime_path),
            "runtime_receipt_sha256": sha256(runtime_path) if write else None,
            "ple_resident_bytes": measured_resident,
            "method_config_sha256": sha256(config_path) if write else None,
        },
        "reference_kind": reference_kind,
        "independent_decoder_max_abs_diff_bfloat16": independent_diff,
        "capability_reconstruction_max_abs_diff_bfloat16": dense_diff,
        "reconstruction_present": reconstruction.is_file(),
    }
    del reference, independent, observed, module, arrays
    gc.collect()
    return result


