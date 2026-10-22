#!/usr/bin/env python3
"""Export frequency-tiered OPTIC with INT8 decoders for BF16 E4B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_eval.tiered_optic_e4b_adapter import (  # noqa: E402
    load_compact_ple,
)
from ple_eval.tiered_optic_e4b_codec import (  # noqa: E402
    BASE_GROUPS,
    CODEBOOK_SIZE,
    DENSE_BYTES,
    D_PLE,
    EXPECTED_RESIDENT_BYTES,
    EXTRA_GROUPS,
    FORMAT_ID,
    GROUP_DIM,
    HIGH_RATE_COUNT,
    LATENT_DIM,
    LAYERS,
    MAX_GROUPS,
    METHOD_NAME,
    SCHEMA_VERSION,
    SPECIAL_TOKEN_IDS,
    STRICT_100X_MAX_BYTES,
    VOCAB_SIZE,
    artifact_schema,
    resident_census,
    validate_artifact,
    verify_requested_row_roundtrip,
)


NAME = (
    "frequency-adaptive-variable-rate-cross-layer-latent-vector-quantization-"
    "with-int8-decoding-and-exact-tokenizer-special-rows"
)
LAUNCH_SCHEMA = f"{NAME}-finalization-launch-v1"
RECEIPT_SCHEMA = f"{NAME}-finalization-receipt-v1"
FIT_RECEIPT_SCHEMA = f"{NAME}-fit-receipt-v1"
RAW_FP16_RESIDENT_BYTES = 70_063_104
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
ROUNDTRIP_CONTROL_IDS = tuple(range(512)) + tuple(range(VOCAB_SIZE - 4, VOCAB_SIZE))


class FinalizationContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bound_file(record: Any, label: str) -> Path:
    require(isinstance(record, dict), f"{label} binding must be an object")
    path = resolve_path(str(record.get("path", "")))
    require(path.is_file(), f"missing {label}: {path}")
    require(sha256_file(path) == record.get("sha256"), f"{label} SHA-256 differs")
    return path


def validate_launch(path: Path) -> dict[str, Any]:
    launch = load_json(path.resolve())
    for key in ("fit_recipe", "fit_runner", "fit_artifact", "fit_receipt",
                "fit_frequency", "checkpoint_weights", "tokenizer_json", "codec", "adapter"):
        bound_file(launch[key], key)
    receipt = load_json(resolve_path(launch["fit_receipt"]["path"]))
    require(receipt["fit_recipe_sha256"] == launch["fit_recipe"]["sha256"], "fit recipe differs")
    require(receipt["artifact_sha256"] == launch["fit_artifact"]["sha256"], "fit artifact differs")
    require(receipt["low_rate_extra_codes_all_zero"] is True, "low-rate codes differ")
    return launch


def _load_fit_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frequency_adaptive_fit_for_finalization", path)
    require(spec is not None and spec.loader is not None, "cannot load the fit runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_fit_artifact(path: Path, fit_recipe: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        require("metadata" in archive.files, "fit metadata is missing")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files if name != "metadata"}
    expected = {
        "token_codes_packed_uint8": (np.dtype("uint8"), (VOCAB_SIZE * MAX_GROUPS,)),
        "codebooks_fp16": (np.dtype("float16"), (MAX_GROUPS, CODEBOOK_SIZE, GROUP_DIM)),
        "decoder_weight_fp16": (np.dtype("float16"), (LAYERS, D_PLE, LATENT_DIM)),
        "decoder_bias_fp16": (np.dtype("float16"), (LAYERS, D_PLE)),
    }
    require(set(arrays) == set(expected), "fit array set differs")
    for name, (dtype, shape) in expected.items():
        require(arrays[name].dtype == dtype and tuple(arrays[name].shape) == shape, f"fit {name} differs")
    require(metadata.get("compressed_ple_format_id") == "optic_npz", "fit format differs")
    require(int(metadata.get("latent_dim", -1)) == LATENT_DIM, "fit latent dimension differs")
    require(int(metadata.get("n_groups", -1)) == MAX_GROUPS, "fit group count differs")
    require(Path(metadata.get("fit_config", "")).resolve() == fit_recipe, "fit metadata recipe differs")
    require(metadata.get("fit_config_sha256") == sha256_file(fit_recipe), "fit metadata recipe SHA differs")
    return metadata, arrays


def quantize_decoder(decoder_fp16: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    decoder = np.asarray(decoder_fp16, dtype=np.float32)
    maximum = np.max(np.abs(decoder), axis=2)
    scales = np.ascontiguousarray(np.where(maximum > 0, maximum / 127.0, 1.0).astype(np.float16))
    require(bool(np.isfinite(scales).all()) and bool((scales > 0).all()), "decoder scales are invalid")
    quantized = np.ascontiguousarray(
        np.clip(np.rint(decoder / scales.astype(np.float32)[..., None]), -127, 127).astype(np.int8)
    )
    dequantized = quantized.astype(np.float32) * scales.astype(np.float32)[..., None]
    error = dequantized - decoder
    denominator = float(np.square(decoder.astype(np.float64)).sum())
    return quantized, scales, {
        "nmse": float(np.square(error.astype(np.float64)).sum() / max(denominator, 1e-30)),
        "cosine_similarity": float(
            np.sum(dequantized.astype(np.float64) * decoder.astype(np.float64))
            / max(np.linalg.norm(dequantized.astype(np.float64)) * np.linalg.norm(decoder.astype(np.float64)), 1e-30)
        ),
    }


def _tensor_header(path: Path, key: str) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        require(len(raw) == 8, "invalid safetensors header")
        header_length = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(header_length))
    require(key in header, f"checkpoint lacks {key}")
    return 8 + header_length, header[key]


def exact_checkpoint_special_rows(path: Path) -> np.ndarray:
    data_start, metadata = _tensor_header(path, PLE_KEY)
    require(metadata["dtype"] == "BF16", "checkpoint PLE is not BF16")
    require(metadata["shape"] == [VOCAB_SIZE, LAYERS * D_PLE], "checkpoint PLE shape differs")
    tensor_start, tensor_stop = (int(value) for value in metadata["data_offsets"])
    require(tensor_stop - tensor_start == DENSE_BYTES, "checkpoint PLE byte span differs")
    rows = np.empty((len(SPECIAL_TOKEN_IDS), LAYERS * D_PLE), dtype="<u2")
    with path.open("rb") as handle:
        for index, token_id in enumerate(SPECIAL_TOKEN_IDS):
            handle.seek(data_start + tensor_start + token_id * LAYERS * D_PLE * 2)
            raw = handle.read(LAYERS * D_PLE * 2)
            require(len(raw) == LAYERS * D_PLE * 2, "short checkpoint special-row read")
            rows[index] = np.frombuffer(raw, dtype="<u2")
    return np.ascontiguousarray(rows, dtype=np.uint16)


def reference_rows(arrays: Mapping[str, np.ndarray], row_ids: np.ndarray):
    import torch

    base = arrays["base_token_codes_uint8"][row_ids.astype(np.int64)].astype(np.int64)
    high_ids = arrays["high_rate_token_ids_uint32"]
    positions = np.searchsorted(high_ids, row_ids)
    bounded = np.minimum(positions, HIGH_RATE_COUNT - 1)
    high = (positions < HIGH_RATE_COUNT) & (high_ids[bounded] == row_ids)
    extra = np.zeros((row_ids.size, EXTRA_GROUPS), dtype=np.int64)
    extra[high] = arrays["high_rate_extra_codes_uint8"][bounded[high]]
    codes = np.concatenate((base, extra), axis=1)
    books = arrays["codebooks_fp16"]
    latent = np.concatenate([books[group, codes[:, group]] for group in range(MAX_GROUPS)], axis=1)
    latent_tensor = torch.from_numpy(np.ascontiguousarray(latent)).to(torch.float32)
    decoded = []
    for layer in range(LAYERS):
        weight = torch.from_numpy(arrays["decoder_weight_int8"][layer]).to(torch.float32)
        scale = torch.from_numpy(arrays["decoder_scale_per_output_fp16"][layer]).to(torch.float32)
        bias = torch.from_numpy(arrays["decoder_bias_fp16"][layer]).to(torch.float32)
        decoded.append(latent_tensor @ (weight * scale[:, None]).T + bias)
    rows = torch.cat(decoded, dim=1).to(torch.bfloat16)
    exact = torch.from_numpy(arrays["special_rows_bf16_bits_uint16"]).view(torch.bfloat16)
    for index, token_id in enumerate(SPECIAL_TOKEN_IDS):
        mask = torch.from_numpy(row_ids.astype(np.int64)) == token_id
        if bool(mask.any().item()):
            rows[mask] = exact[index]
    return rows


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def finalize(path: Path) -> dict[str, Any]:
    launch = validate_launch(path)
    fit_recipe = resolve_path(launch["fit_recipe"]["path"])
    fit_runner_path = resolve_path(launch["fit_runner"]["path"])
    fit_artifact = resolve_path(launch["fit_artifact"]["path"])
    fit_receipt_path = resolve_path(launch["fit_receipt"]["path"])
    fit_metadata, source = read_fit_artifact(fit_artifact, fit_recipe)
    frequency_path = resolve_path(launch["fit_frequency"]["path"])
    frequency = np.fromfile(frequency_path, dtype="<u8")
    selector = _load_fit_runner(fit_runner_path)
    high_ids = selector.select_high_rate_token_ids(frequency, SPECIAL_TOKEN_IDS)
    codes = source["token_codes_packed_uint8"].reshape(VOCAB_SIZE, MAX_GROUPS)
    high_mask = np.zeros(VOCAB_SIZE, dtype=bool)
    high_mask[high_ids.astype(np.int64)] = True
    require(bool((codes[~high_mask, BASE_GROUPS:] == 0).all()), "low-rate extra codes drifted after fit")
    quantized, scales, quantization_audit = quantize_decoder(source["decoder_weight_fp16"])
    checkpoint = resolve_path(launch["checkpoint_weights"]["path"])
    special_rows = exact_checkpoint_special_rows(checkpoint)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(launch["tokenizer_snapshot"]), local_files_only=True, trust_remote_code=True,
    )
    require(tuple(int(value) for value in tokenizer.all_special_ids) == SPECIAL_TOKEN_IDS, "tokenizer special IDs differ")
    arrays = {
        "base_token_codes_uint8": np.ascontiguousarray(codes[:, :BASE_GROUPS], dtype=np.uint8),
        "high_rate_token_ids_uint32": np.ascontiguousarray(high_ids, dtype=np.uint32),
        "high_rate_extra_codes_uint8": np.ascontiguousarray(
            codes[high_ids.astype(np.int64), BASE_GROUPS:], dtype=np.uint8,
        ),
        "codebooks_fp16": np.ascontiguousarray(source["codebooks_fp16"], dtype=np.float16),
        "decoder_weight_int8": quantized,
        "decoder_scale_per_output_fp16": scales,
        "decoder_bias_fp16": np.ascontiguousarray(source["decoder_bias_fp16"], dtype=np.float16),
        "special_rows_bf16_bits_uint16": special_rows,
    }
    launch_sha = sha256_file(path.resolve())
    metadata = {
        "compressed_ple_format_id": FORMAT_ID,
        "schema_version": SCHEMA_VERSION,
        **artifact_schema()["fixed_metadata"],
        "resident_byte_census": resident_census(),
        "group_dim_allocation": fit_metadata.get("group_dim_allocation"),
        "fit_frequency_sha256": sha256_file(frequency_path),
        "high_rate_token_ids_sha256": array_sha256(high_ids.astype("<u4", copy=False)),
        "checkpoint_ple_key": PLE_KEY,
        "checkpoint_special_rows_sha256": array_sha256(special_rows),
        "tokenizer_json_sha256": launch["tokenizer_json"]["sha256"],
        "provenance_sha256": {
            "finalizer": sha256_file(Path(__file__).resolve()),
            "finalization_launch": launch_sha,
            "fit_artifact": sha256_file(fit_artifact),
            "fit_receipt": sha256_file(fit_receipt_path),
            "fit_frequency": sha256_file(frequency_path),
            "checkpoint_weights": sha256_file(checkpoint),
            "tokenizer_snapshot_identity": launch["tokenizer_json"]["sha256"],
        },
    }
    census = validate_artifact(metadata, arrays)
    output = resolve_path(launch["compact_output_root"])
    require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    require(not partial.exists(), f"refusing to reuse {partial}")
    partial.mkdir()
    artifact_path = partial / "compressed_ple.npz"
    with artifact_path.open("xb") as handle:
        np.savez(handle, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    artifact_sha = sha256_file(artifact_path)
    metrics = {
        "schema_version": f"{NAME}-fit-and-finalization-metrics-v1",
        "method_name": METHOD_NAME,
        "fit_artifact_sha256": sha256_file(fit_artifact),
        "final_artifact_sha256": artifact_sha,
        "resident_byte_census": census,
        "compression_ratio": DENSE_BYTES / census["total"],
        "strict_100x_slack_bytes": STRICT_100X_MAX_BYTES - census["total"],
        "decoder_quantization_audit": quantization_audit,
        "high_rate_token_ids_sha256": array_sha256(high_ids.astype("<u4", copy=False)),
        "low_rate_extra_codes_all_zero": True,
        "exact_special_rows_sha256": array_sha256(special_rows),
    }
    write_json_new(partial / "fit_and_finalization_metrics.json", metrics)
    control = np.ascontiguousarray(
        np.unique(np.asarray(ROUNDTRIP_CONTROL_IDS + SPECIAL_TOKEN_IDS + tuple(high_ids[:8]), dtype=np.uint32))
    )
    expected_rows = reference_rows(arrays, control)
    module = load_compact_ple(artifact_path, device="cpu")
    roundtrip = verify_requested_row_roundtrip(module, control, expected_rows, atol=0.0)
    require(roundtrip["persistent_resident_bytes"] == EXPECTED_RESIDENT_BYTES, "online resident bytes differ")
    import torch

    online_special = module(torch.as_tensor(SPECIAL_TOKEN_IDS, dtype=torch.long)).view(torch.uint16).cpu().numpy()
    require(np.array_equal(online_special, special_rows), "online special rows are not exact checkpoint bits")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "finalized",
        "method_name": METHOD_NAME,
        "finalization_launch_path": str(path.resolve()),
        "finalization_launch_sha256": launch_sha,
        "finalizer_sha256": sha256_file(Path(__file__).resolve()),
        "codec_sha256": launch["codec"]["sha256"],
        "adapter_sha256": launch["adapter"]["sha256"],
        "fit_receipt_sha256": sha256_file(fit_receipt_path),
        "fit_artifact_sha256": sha256_file(fit_artifact),
        "artifact": {
            "path": str(output / "compressed_ple.npz"),
            "sha256": artifact_sha,
            "serialized_file_bytes": int(artifact_path.stat().st_size),
        },
        "resident_byte_census": census,
        "resident_bytes": EXPECTED_RESIDENT_BYTES,
        "compression_ratio": DENSE_BYTES / EXPECTED_RESIDENT_BYTES,
        "strict_100x_slack_bytes": STRICT_100X_MAX_BYTES - EXPECTED_RESIDENT_BYTES,
        "roundtrip": roundtrip,
        "low_rate_extra_codes_all_zero": True,
        "sorted_high_rate_token_ids": True,
        "persistent_special_token_id_tensor_present": False,
        "dense_ple_or_dense_cache_present": False,
        "quality_evaluation_run": False,
        "benchmark_inputs_used": False,
        "l1_or_l2_inputs_used": False,
    }
    write_json_new(partial / "finalization_receipt.json", receipt)
    os.replace(partial, output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=ROOT / "configs/fit/tiered_optic_e4b_bf16_recipe.json")
    args = parser.parse_args()
    recipe = load_json(args.recipe)
    def bind(value):
        path = resolve_path(value)
        return {"path": str(path), "sha256": sha256_file(path)}
    snapshot = resolve_path(recipe["model"]["tokenizer_snapshot"])
    launch = {
        "fit_recipe": bind(args.recipe),
        "fit_runner": bind(ROOT / "scripts/fit_tiered_optic_e4b.py"),
        "fit_artifact": bind(args.fit_dir / "full_code_fit_artifact.npz"),
        "fit_receipt": bind(args.fit_dir / "fit_receipt.json"),
        "fit_frequency": bind(recipe["fit_inputs"]["fit_only_frequency"]["path"]),
        "checkpoint_weights": bind(recipe["model"]["weights"]),
        "tokenizer_json": bind(snapshot / "tokenizer.json"),
        "tokenizer_snapshot": str(snapshot),
        "codec": bind(ROOT / "src/ple_eval/tiered_optic_e4b_codec.py"),
        "adapter": bind(ROOT / "src/ple_eval/tiered_optic_e4b_adapter.py"),
        "compact_output_root": str(args.output_dir.resolve()),
    }
    launch_path = args.fit_dir / "finalization_inputs.json"
    write_json_new(launch_path, launch)
    print(json.dumps(finalize(launch_path), indent=2))

if __name__ == "__main__":
    main()
