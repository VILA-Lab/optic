#!/usr/bin/env python3
"""Layer-coordinate affine calibration for a PCA-rotated PQ source PLE.

This builder fits one fp16 scale and one fp16 bias for each
layer/coordinate pair against the original Gemma PLE tensor. The dense BF16 PLE
reconstruction is an evaluation file. The resident compressed PLE
representation is the source compressed PLE representation plus the two counted
fp16 affine tensors in ``compressed_ple.npz``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

BASELINE_BUILDERS = Path(__file__).resolve().parents[1] / "baselines"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASELINE_BUILDERS))

from baseline_common import PLE_KEY, RECON_KEY, VOCAB_SIZE  # noqa: E402
from baseline_pca_rotated_pq import DEFAULT_IT_WEIGHTS  # noqa: E402


METHOD_ID = "affine_calibrated_pca_rotated_pq_scaled_exceptions"
FORMAT_ID = "affine_calibrated_pca_rotated_pq_scaled_exceptions_npz"
SOURCE_METHOD_ID = "pca_rotated_pq_scaled_exceptions"
SOURCE_FORMAT_ID = "pca_rotated_pq_scaled_exceptions_npz"
LOG_PREFIX = "[affine-pca-pq-scaled-exc]"
SCALE_ARRAY = "layer_coordinate_scale_fp16"
BIAS_ARRAY = "layer_coordinate_bias_fp16"
STORAGE_BYTES_FORMULA = "source_resident_bytes + L*d*2*2"
SINGLE_TENSOR_STORAGE_BYTES_FORMULA = "source_resident_bytes + L*d*2"
CALIBRATION_MODES = {"affine", "scale_only", "bias_only"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def benchmark_id_for_metrics_output(output_path: Path) -> str:
    for parent in (output_path.parent, *output_path.parents):
        record_path = parent / "experiment.json"
        if not record_path.is_file():
            continue
        try:
            experiment = load_json(record_path)
        except SystemExit:
            continue
        benchmark_id = experiment.get("benchmark_id")
        if isinstance(benchmark_id, str) and benchmark_id:
            return benchmark_id
    return "build_full_ple_reconstruction"


def load_npz_metadata(npz, source: Path) -> dict[str, Any]:
    if "metadata" not in npz.files:
        raise SystemExit(f"{source} missing scalar JSON metadata")
    raw = npz["metadata"]
    if raw.shape != ():
        raise SystemExit(f"{source} metadata shape {raw.shape} must be scalar")
    try:
        metadata = json.loads(str(raw.item()))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{source} metadata is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit(f"{source} metadata JSON must be an object")
    return metadata


def require_npz_array(np: Any, npz, source: Path, name: str, dtype, shape: tuple[int, ...]):
    if name not in npz.files:
        raise SystemExit(f"{source} missing required resident array {name!r}")
    value = npz[name]
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise SystemExit(f"{source} resident array {name!r} dtype {value.dtype} != {expected_dtype}")
    if tuple(value.shape) != shape:
        raise SystemExit(f"{source} resident array {name!r} shape {tuple(value.shape)} != {shape}")
    return value


def find_ple_key(path: Path, *, allow_recon_key: bool) -> tuple[str, tuple[int, ...]]:
    from safetensors import safe_open

    allowed = [PLE_KEY]
    if allow_recon_key:
        allowed.append(RECON_KEY)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        key = next((candidate for candidate in allowed if candidate in keys), None)
        if key is None:
            raise SystemExit(f"{path} missing PLE tensor key; expected one of {allowed}")
        shape = tuple(handle.get_slice(key).get_shape())
    return key, shape


def resolved_record_path(record_cwd: str, item: str) -> Path:
    path = Path(item)
    if path.is_absolute():
        return path.resolve()
    cwd = Path(record_cwd or ".")
    if not cwd.is_absolute():
        cwd = ROOT / cwd
    return (cwd / path).resolve()


def verify_source_record(
    args: argparse.Namespace,
    source_dense_path: Path,
    source_compressed_path: Path,
) -> dict[str, Any]:
    if not args.source_record_dir:
        return {}

    record_dir = Path(args.source_record_dir)
    normalized = str(record_dir).replace("\\", "/")
    experiment_path = record_dir / "experiment.json"
    if not experiment_path.is_file():
        raise SystemExit(f"--source-record-dir is missing experiment.json: {record_dir}")

    experiment = load_json(experiment_path)
    source_method_id = experiment.get("compression_method_id")
    if source_method_id != args.expected_source_method:
        raise SystemExit(
            f"source record compression_method_id {source_method_id!r} != --expected-source-method {args.expected_source_method!r}"
        )
    config_id = experiment.get("compression_configuration_id") or experiment.get("config_id")
    if config_id != args.source_config_id:
        raise SystemExit(f"source config id mismatch: {config_id!r} != {args.source_config_id!r}")

    reproducibility = experiment.get("reproducibility", {})
    if not isinstance(reproducibility, dict):
        raise SystemExit("source record reproducibility must be an object")
    output_paths = reproducibility.get("output_paths", [])
    if not isinstance(output_paths, list):
        raise SystemExit("source record reproducibility.output_paths must be a list")
    record_cwd = str(reproducibility.get("cwd") or ".")
    resolved_outputs = {
        resolved_record_path(record_cwd, item)
        for item in output_paths
        if isinstance(item, str)
    }
    if source_dense_path.resolve() not in resolved_outputs:
        raise SystemExit("--source-dense-ple-reconstruction is not bound by the source record output_paths")
    if source_compressed_path.resolve() not in resolved_outputs:
        raise SystemExit("--source-compressed-ple-npz is not bound by the source record output_paths")

    return {
        "source_record_dir": str(record_dir),
        "source_record_sha256": sha256_file(experiment_path),
    }


def source_compressed_ple_info(
    np: Any,
    path: Path,
    *,
    source_method: str,
    expected_source_format_id: str,
    source_resident_bytes: int,
    expected_shape: list[int],
) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=False) as npz:
        metadata = load_npz_metadata(npz, path)
        if metadata.get("method") != source_method:
            raise SystemExit(f"source compressed PLE method {metadata.get('method')!r} != {source_method!r}")
        source_format_id = str(metadata.get("compressed_ple_format_id", ""))
        if expected_source_format_id and source_format_id != expected_source_format_id:
            raise SystemExit(
                f"source compressed PLE format {source_format_id!r} != --expected-source-format-id {expected_source_format_id!r}"
            )
        if metadata.get("ple_shape") != expected_shape:
            raise SystemExit(f"source compressed PLE ple_shape {metadata.get('ple_shape')} != {expected_shape}")
        resident_tensor_bytes = sum(int(npz[name].nbytes) for name in npz.files if name != "metadata")
        if resident_tensor_bytes != int(source_resident_bytes):
            raise SystemExit(
                "source compressed PLE resident tensor bytes "
                f"{resident_tensor_bytes} != --source-resident-bytes {source_resident_bytes}"
            )
        return {
            "source_compressed_ple_format_id": source_format_id,
            "source_compressed_ple_resident_tensor_bytes": resident_tensor_bytes,
            "source_compressed_ple_npz_total_bytes": path.stat().st_size,
            "source_compressed_ple_npz_sha256": sha256_file(path),
        }


def affine_storage_bytes(layers: int, d: int, mode: str) -> int:
    if mode == "affine":
        return int(layers) * int(d) * 2 * 2
    if mode in {"scale_only", "bias_only"}:
        return int(layers) * int(d) * 2
    raise SystemExit(f"unsupported calibration mode: {mode}")


def storage_bytes_formula(mode: str) -> str:
    if mode == "affine":
        return STORAGE_BYTES_FORMULA
    if mode in {"scale_only", "bias_only"}:
        return SINGLE_TENSOR_STORAGE_BYTES_FORMULA
    raise SystemExit(f"unsupported calibration mode: {mode}")


def resident_tensors_for_mode(mode: str) -> list[str]:
    if mode == "affine":
        return [SCALE_ARRAY, BIAS_ARRAY]
    if mode == "scale_only":
        return [SCALE_ARRAY]
    if mode == "bias_only":
        return [BIAS_ARRAY]
    raise SystemExit(f"unsupported calibration mode: {mode}")


def validate_args(args: argparse.Namespace) -> None:
    if args.num_layers <= 0 or args.d_ple != 256:
        raise SystemExit("this active builder requires a positive layer count and d_ple=256")
    if args.source_method != args.expected_source_method:
        raise SystemExit(f"--source-method must match --expected-source-method: {args.expected_source_method!r}")
    if not args.output_method_id or any(ch.isspace() for ch in args.output_method_id):
        raise SystemExit("--output-method-id must be a nonempty identifier without whitespace")
    if not args.output_format_id or any(ch.isspace() for ch in args.output_format_id):
        raise SystemExit("--output-format-id must be a nonempty identifier without whitespace")
    if not args.source_config_id:
        raise SystemExit("--source-config-id is required for source compressed-PLE binding")
    if args.source_resident_bytes <= 0:
        raise SystemExit("--source-resident-bytes must be positive")
    if args.calibration_mode not in CALIBRATION_MODES:
        raise SystemExit(f"--calibration-mode must be one of {sorted(CALIBRATION_MODES)}")
    if not (0.0 <= args.alpha <= 1.0) or not math.isfinite(args.alpha):
        raise SystemExit("--alpha must be finite and in [0, 1]")
    if args.eps <= 0.0 or not math.isfinite(args.eps):
        raise SystemExit("--eps must be finite and positive")
    if args.chunk_rows <= 0:
        raise SystemExit("--chunk-rows must be positive")
    if args.resident_roundtrip_atol < 0.0 or not math.isfinite(args.resident_roundtrip_atol):
        raise SystemExit("--resident-roundtrip-atol must be finite and nonnegative")
    if Path(args.save_recon).name != "dense_ple_reconstruction.safetensors":
        raise SystemExit("--save-recon must write dense_ple_reconstruction.safetensors")
    if Path(args.save_compressed_ple_npz).name != "compressed_ple.npz":
        raise SystemExit("--save-compressed-ple-npz must write compressed_ple.npz")
    if Path(args.output).name != "metrics.json":
        raise SystemExit("--output must write metrics.json")


def fit_affine_parameters(
    torch: Any,
    *,
    source_path: Path,
    source_key: str,
    reference_path: Path,
    reference_key: str,
    vocab: int,
    layers: int,
    d: int,
    chunk_rows: int,
    eps: float,
    alpha: float,
    mode: str,
) -> tuple[Any, Any, float]:
    from safetensors import safe_open

    source_sum = torch.zeros((layers, d), dtype=torch.float64)
    source_sumsq = torch.zeros((layers, d), dtype=torch.float64)
    reference_sum = torch.zeros((layers, d), dtype=torch.float64)
    reference_sumsq = torch.zeros((layers, d), dtype=torch.float64)
    source_err = 0.0
    reference_norm = 0.0

    with safe_open(str(source_path), framework="pt", device="cpu") as source_file:
        with safe_open(str(reference_path), framework="pt", device="cpu") as reference_file:
            source_slice = source_file.get_slice(source_key)
            reference_slice = reference_file.get_slice(reference_key)
            for layer in range(layers):
                col_slice = slice(layer * d, (layer + 1) * d)
                for start in range(0, vocab, chunk_rows):
                    stop = min(vocab, start + chunk_rows)
                    source = source_slice[start:stop, col_slice].to(torch.float64)
                    reference = reference_slice[start:stop, col_slice].to(torch.float64)
                    source_sum[layer] += source.sum(dim=0)
                    source_sumsq[layer] += source.square().sum(dim=0)
                    reference_sum[layer] += reference.sum(dim=0)
                    reference_sumsq[layer] += reference.square().sum(dim=0)
                    diff = source - reference
                    source_err += diff.square().sum().item()
                    reference_norm += reference.square().sum().item()
                    del source, reference, diff
                print(f"{LOG_PREFIX} fitted affine statistics layer {layer + 1}/{layers}", flush=True)

    source_mean = source_sum / float(vocab)
    reference_mean = reference_sum / float(vocab)
    source_var = (source_sumsq / float(vocab) - source_mean.square()).clamp_min(0.0)
    reference_var = (reference_sumsq / float(vocab) - reference_mean.square()).clamp_min(0.0)
    full_scale = reference_var.sqrt() / source_var.sqrt().clamp_min(float(eps))
    full_bias = reference_mean - full_scale * source_mean
    if mode == "affine":
        scale = (1.0 - float(alpha)) + float(alpha) * full_scale
        bias = float(alpha) * full_bias
    elif mode == "scale_only":
        scale = (1.0 - float(alpha)) + float(alpha) * full_scale
        bias = torch.zeros_like(reference_mean)
    elif mode == "bias_only":
        scale = torch.ones_like(full_scale)
        bias = float(alpha) * (reference_mean - source_mean)
    else:
        raise SystemExit(f"unsupported calibration mode: {mode}")
    source_nmse = source_err / max(reference_norm, 1e-30)
    return scale.to(torch.float16), bias.to(torch.float16), source_nmse


def write_compressed_ple_npz(
    np: Any,
    path: Path,
    *,
    metadata: dict[str, Any],
    scale_fp16,
    bias_fp16,
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {"metadata": np.array(json.dumps(metadata))}
    if mode in {"affine", "scale_only"}:
        entries[SCALE_ARRAY] = scale_fp16
    if mode in {"affine", "bias_only"}:
        entries[BIAS_ARRAY] = bias_fp16
    np.savez(str(path), **entries)


def load_affine_payload(
    np: Any,
    path: Path,
    *,
    layers: int,
    d: int,
    storage_bytes: int,
    method_id: str,
    format_id: str,
    storage_bytes_formula: str,
    mode: str,
) -> dict[str, Any]:
    expected_entries = {"metadata", *resident_tensors_for_mode(mode)}
    with np.load(str(path), allow_pickle=False) as npz:
        actual_entries = set(npz.files)
        if actual_entries != expected_entries:
            raise SystemExit(
                "compressed PLE representation file entries "
                f"{sorted(actual_entries)} != {sorted(expected_entries)}"
            )
        metadata = load_npz_metadata(npz, path)
        if metadata.get("compressed_ple_format_id") != format_id:
            raise SystemExit(f"resident metadata compressed_ple_format_id {metadata.get('compressed_ple_format_id')!r} != {format_id!r}")
        if metadata.get("method") != method_id:
            raise SystemExit(f"resident metadata method {metadata.get('method')!r} != {method_id!r}")
        if metadata.get("storage_bytes_formula") != storage_bytes_formula:
            raise SystemExit("resident metadata storage_bytes_formula does not match the method record formula")
        if int(metadata.get("storage_bytes", -1)) != int(storage_bytes):
            raise SystemExit(f"resident metadata storage_bytes {metadata.get('storage_bytes')} != {storage_bytes}")
        if metadata.get("resident_tensors") != resident_tensors_for_mode(mode):
            raise SystemExit("resident metadata resident_tensors does not match the calibration mode")
        if mode in {"affine", "scale_only"}:
            scale = require_npz_array(np, npz, path, SCALE_ARRAY, np.float16, (layers, d)).copy()
        else:
            scale = np.ones((layers, d), dtype=np.float16)
        if mode in {"affine", "bias_only"}:
            bias = require_npz_array(np, npz, path, BIAS_ARRAY, np.float16, (layers, d)).copy()
        else:
            bias = np.zeros((layers, d), dtype=np.float16)

    affine_bytes = sum(int(npz_array.nbytes) for npz_array in [scale if mode in {"affine", "scale_only"} else None, bias if mode in {"affine", "bias_only"} else None] if npz_array is not None)
    expected_affine_bytes = affine_storage_bytes(layers, d, mode)
    if affine_bytes != expected_affine_bytes:
        raise SystemExit(f"affine resident tensor bytes {affine_bytes} != {expected_affine_bytes}")
    if int(metadata.get("affine_layer_coordinate_bytes", -1)) != expected_affine_bytes:
        raise SystemExit("resident metadata affine_layer_coordinate_bytes does not match resident arrays")
    return {"metadata": metadata, "scale_fp16": scale, "bias_fp16": bias, "affine_bytes": affine_bytes}


def materialize_dense_reconstruction(
    torch: Any,
    *,
    source_path: Path,
    source_key: str,
    reference_path: Path,
    reference_key: str,
    scale_fp16,
    bias_fp16,
    recon_path: Path,
    vocab: int,
    layers: int,
    d: int,
    chunk_rows: int,
) -> float:
    from safetensors import safe_open
    from safetensors.torch import save_file

    scale = scale_fp16.to(torch.float32)
    bias = bias_fp16.to(torch.float32)
    recon_packed = torch.empty((vocab, layers * d), dtype=torch.bfloat16)
    err = 0.0
    norm = 0.0

    with safe_open(str(source_path), framework="pt", device="cpu") as source_file:
        with safe_open(str(reference_path), framework="pt", device="cpu") as reference_file:
            source_slice = source_file.get_slice(source_key)
            reference_slice = reference_file.get_slice(reference_key)
            for layer in range(layers):
                col_slice = slice(layer * d, (layer + 1) * d)
                layer_scale = scale[layer]
                layer_bias = bias[layer]
                for start in range(0, vocab, chunk_rows):
                    stop = min(vocab, start + chunk_rows)
                    source = source_slice[start:stop, col_slice].to(torch.float32)
                    reference = reference_slice[start:stop, col_slice].to(torch.float32)
                    decoded = (source * layer_scale) + layer_bias
                    decoded_bf16 = decoded.to(torch.bfloat16)
                    recon_packed[start:stop, col_slice] = decoded_bf16
                    diff = decoded_bf16.to(torch.float32) - reference
                    err += diff.double().square().sum().item()
                    norm += reference.double().square().sum().item()
                    del source, reference, decoded, decoded_bf16, diff
                print(f"{LOG_PREFIX} materialized dense PLE reconstruction layer {layer + 1}/{layers}", flush=True)

    recon_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({RECON_KEY: recon_packed.contiguous()}, str(recon_path))
    return err / max(norm, 1e-30)


def verify_resident_roundtrip(
    torch: Any,
    np: Any,
    *,
    compressed_path: Path,
    source_path: Path,
    source_key: str,
    recon_path: Path,
    vocab: int,
    layers: int,
    d: int,
    chunk_rows: int,
    storage_bytes: int,
    method_id: str,
    format_id: str,
    storage_bytes_formula: str,
    mode: str,
    atol: float,
) -> float:
    from safetensors import safe_open

    payload = load_affine_payload(
        np,
        compressed_path,
        layers=layers,
        d=d,
        storage_bytes=storage_bytes,
        method_id=method_id,
        format_id=format_id,
        storage_bytes_formula=storage_bytes_formula,
        mode=mode,
    )
    scale = torch.from_numpy(payload["scale_fp16"].astype(np.float32, copy=False))
    bias = torch.from_numpy(payload["bias_fp16"].astype(np.float32, copy=False))
    max_abs_diff = 0.0

    with safe_open(str(source_path), framework="pt", device="cpu") as source_file:
        with safe_open(str(recon_path), framework="pt", device="cpu") as recon_file:
            if RECON_KEY not in set(recon_file.keys()):
                raise SystemExit(f"{recon_path} missing dense PLE reconstruction tensor key {RECON_KEY!r}")
            source_slice = source_file.get_slice(source_key)
            recon_slice = recon_file.get_slice(RECON_KEY)
            expected_shape = [vocab, layers * d]
            if list(recon_slice.get_shape()) != expected_shape:
                raise SystemExit(f"dense PLE reconstruction shape {list(recon_slice.get_shape())} != {expected_shape}")
            for layer in range(layers):
                col_slice = slice(layer * d, (layer + 1) * d)
                layer_scale = scale[layer]
                layer_bias = bias[layer]
                for start in range(0, vocab, chunk_rows):
                    stop = min(vocab, start + chunk_rows)
                    source = source_slice[start:stop, col_slice].to(torch.float32)
                    decoded = ((source * layer_scale) + layer_bias).to(torch.bfloat16)
                    saved = recon_slice[start:stop, col_slice]
                    decoded_f32 = decoded.to(torch.float32)
                    saved_f32 = saved.to(torch.float32)
                    if not bool(torch.isfinite(decoded_f32).all().item()):
                        raise SystemExit("resident affine decode produced non-finite values")
                    if not bool(torch.isfinite(saved_f32).all().item()):
                        raise SystemExit("saved BF16 dense PLE reconstruction contains non-finite values")
                    diff = float((decoded_f32 - saved_f32).abs().max().item())
                    max_abs_diff = max(max_abs_diff, diff)
                    if max_abs_diff > atol:
                        raise SystemExit(
                            f"resident reload/decode max_abs_diff {max_abs_diff} exceeds tolerance {atol}"
                        )
                    del source, decoded, saved, decoded_f32, saved_f32
    return max_abs_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--it-weights", default=DEFAULT_IT_WEIGHTS)
    parser.add_argument("--source-dense-ple-reconstruction", "--source-recon", dest="source_dense_ple_reconstruction", required=True)
    parser.add_argument("--source-compressed-ple-npz", required=True)
    parser.add_argument("--source-method", default=SOURCE_METHOD_ID)
    parser.add_argument("--expected-source-method", default=SOURCE_METHOD_ID)
    parser.add_argument("--expected-source-format-id", default=SOURCE_FORMAT_ID)
    parser.add_argument("--source-config-id", required=True)
    parser.add_argument("--source-record-dir", default="")
    parser.add_argument("--source-resident-bytes", type=int, required=True)
    parser.add_argument("--output-method-id", default=METHOD_ID)
    parser.add_argument("--output-format-id", default=FORMAT_ID)
    parser.add_argument("--calibration-mode", choices=sorted(CALIBRATION_MODES), default="affine")
    parser.add_argument("--num-layers", type=int, default=35)
    parser.add_argument("--d-ple", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--resident-roundtrip-atol", type=float, default=0.0)
    parser.add_argument("--save-recon", required=True)
    parser.add_argument("--save-compressed-ple-npz", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    import numpy as np
    import torch

    t0 = time.time()
    source_dense_path = Path(args.source_dense_ple_reconstruction)
    source_compressed_path = Path(args.source_compressed_ple_npz)
    reference_path = Path(args.it_weights)
    recon_path = Path(args.save_recon)
    compressed_path = Path(args.save_compressed_ple_npz)
    output_path = Path(args.output)
    for label, path in [
        ("--source-dense-ple-reconstruction", source_dense_path),
        ("--source-compressed-ple-npz", source_compressed_path),
        ("--it-weights", reference_path),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")

    source_key, source_shape = find_ple_key(source_dense_path, allow_recon_key=True)
    reference_key, reference_shape = find_ple_key(reference_path, allow_recon_key=False)
    expected_shape = (VOCAB_SIZE, args.num_layers * args.d_ple)
    if source_shape != expected_shape:
        raise SystemExit(f"source dense PLE reconstruction shape {source_shape} != {expected_shape}")
    if reference_shape != expected_shape:
        raise SystemExit(f"original checkpoint PLE shape {reference_shape} != {expected_shape}")

    vocab = expected_shape[0]
    layers = int(args.num_layers)
    d = int(args.d_ple)
    source_record_info = verify_source_record(args, source_dense_path, source_compressed_path)
    source_compressed_info = source_compressed_ple_info(
        np,
        source_compressed_path,
        source_method=args.source_method,
        expected_source_format_id=args.expected_source_format_id,
        source_resident_bytes=args.source_resident_bytes,
        expected_shape=list(expected_shape),
    )
    calibration_mode = str(args.calibration_mode)
    affine_bytes = affine_storage_bytes(layers, d, calibration_mode)
    resident_storage_formula = storage_bytes_formula(calibration_mode)
    storage_bytes = int(args.source_resident_bytes) + affine_bytes
    raw_bytes = vocab * layers * d * 2

    print(
        f"{LOG_PREFIX} V={vocab} L={layers} d={d} alpha={args.alpha} "
        f"calibration_mode={calibration_mode} "
        f"source_resident_bytes={args.source_resident_bytes} calibration_bytes={affine_bytes}",
        flush=True,
    )
    scale_fp16_t, bias_fp16_t, source_nmse = fit_affine_parameters(
        torch,
        source_path=source_dense_path,
        source_key=source_key,
        reference_path=reference_path,
        reference_key=reference_key,
        vocab=vocab,
        layers=layers,
        d=d,
        chunk_rows=args.chunk_rows,
        eps=args.eps,
        alpha=args.alpha,
        mode=calibration_mode,
    )
    if not bool(torch.isfinite(scale_fp16_t).all().item()):
        raise SystemExit("fp16 layer-coordinate affine scale contains non-finite values")
    if not bool(torch.isfinite(bias_fp16_t).all().item()):
        raise SystemExit("fp16 layer-coordinate affine bias contains non-finite values")

    scale_fp16 = scale_fp16_t.numpy()
    bias_fp16 = bias_fp16_t.numpy()
    metadata = {
        "compressed_ple_format_id": args.output_format_id,
        "method": args.output_method_id,
        "schema_version": "1.0",
        "source_method": args.source_method,
        "source_config_id": args.source_config_id,
        "source_dense_ple_reconstruction": str(source_dense_path),
        "source_compressed_ple_npz": str(source_compressed_path),
        "source_resident_bytes": int(args.source_resident_bytes),
        "source_dense_ple_reconstruction_key": source_key,
        "source_compressed_ple_resident_tensor_bytes": source_compressed_info[
            "source_compressed_ple_resident_tensor_bytes"
        ],
        "num_layers": layers,
        "d_ple": d,
        "vocab_size": vocab,
        "ple_shape": list(expected_shape),
        "recon_key": RECON_KEY,
        "full_ple_coverage": True,
        "alpha": float(args.alpha),
        "calibration_mode": calibration_mode,
        "eps": float(args.eps),
        "affine_dtype": "fp16",
        "affine_granularity": "layer_coordinate",
        "resident_tensors": resident_tensors_for_mode(calibration_mode),
        "source_resident_byte_count_included": True,
        "hidden_raw_copy_allowed": False,
        "affine_layer_coordinate_bytes": affine_bytes,
        "storage_bytes_formula": resident_storage_formula,
        "storage_bytes": storage_bytes,
        "nonresident_file_content": ["metadata JSON string"],
    }
    metadata.update(source_record_info)
    write_compressed_ple_npz(
        np,
        compressed_path,
        metadata=metadata,
        scale_fp16=scale_fp16,
        bias_fp16=bias_fp16,
        mode=calibration_mode,
    )
    payload = load_affine_payload(
        np,
        compressed_path,
        layers=layers,
        d=d,
        storage_bytes=storage_bytes,
        method_id=args.output_method_id,
        format_id=args.output_format_id,
        storage_bytes_formula=resident_storage_formula,
        mode=calibration_mode,
    )
    if payload["affine_bytes"] != affine_bytes:
        raise SystemExit("affine resident byte accounting mismatch after compressed-PLE write")

    calibrated_nmse = materialize_dense_reconstruction(
        torch,
        source_path=source_dense_path,
        source_key=source_key,
        reference_path=reference_path,
        reference_key=reference_key,
        scale_fp16=scale_fp16_t,
        bias_fp16=bias_fp16_t,
        recon_path=recon_path,
        vocab=vocab,
        layers=layers,
        d=d,
        chunk_rows=args.chunk_rows,
    )
    roundtrip_max_abs_diff = verify_resident_roundtrip(
        torch,
        np,
        compressed_path=compressed_path,
        source_path=source_dense_path,
        source_key=source_key,
        recon_path=recon_path,
        vocab=vocab,
        layers=layers,
        d=d,
        chunk_rows=args.chunk_rows,
        storage_bytes=storage_bytes,
        method_id=args.output_method_id,
        format_id=args.output_format_id,
        storage_bytes_formula=resident_storage_formula,
        mode=calibration_mode,
        atol=args.resident_roundtrip_atol,
    )

    out = dict(metadata)
    out.update(
        {
            "benchmark": benchmark_id_for_metrics_output(output_path),
            "validated_full_ple_shape": True,
            "source_dense_ple_reconstruction_nmse_against_original_checkpoint_ple": source_nmse,
            "affine_calibrated_dense_ple_reconstruction_nmse_against_original_checkpoint_ple": calibrated_nmse,
            "target_nmse": calibrated_nmse,
            "raw_nmse": calibrated_nmse,
            "raw_bytes": raw_bytes,
            "affine_layer_coordinate_scale_bytes": int(scale_fp16.nbytes) if calibration_mode in {"affine", "scale_only"} else 0,
            "affine_layer_coordinate_bias_bytes": int(bias_fp16.nbytes) if calibration_mode in {"affine", "bias_only"} else 0,
            "resident_tensor_byte_check_passed": True,
            "resident_file_roundtrip_max_abs_diff": roundtrip_max_abs_diff,
            "resident_roundtrip_atol": float(args.resident_roundtrip_atol),
            "resident_file_roundtrip_passed": roundtrip_max_abs_diff <= args.resident_roundtrip_atol,
            "storage_gib": storage_bytes / 1024**3,
            "compression_ratio": raw_bytes / max(storage_bytes, 1),
            "source_dense_ple_reconstruction_sha256": sha256_file(source_dense_path),
            **source_compressed_info,
            "reference_it_weights_sha256": sha256_file(reference_path),
            "dense_recon_sha256": sha256_file(recon_path),
            "compressed_ple_sha256": sha256_file(compressed_path),
            "compressed_ple_npz_total_bytes": compressed_path.stat().st_size,
            "elapsed_seconds": time.time() - t0,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(
        f"{LOG_PREFIX} source NMSE={source_nmse:.6f} calibrated NMSE={calibrated_nmse:.6f}",
        flush=True,
    )
    print(
        f"{LOG_PREFIX} storage={storage_bytes} bytes ratio={out['compression_ratio']:.4f}x "
        f"roundtrip_max_abs_diff={roundtrip_max_abs_diff:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
