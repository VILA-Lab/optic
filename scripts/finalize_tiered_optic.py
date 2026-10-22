#!/usr/bin/env python3
"""Export a frequency-tiered OPTIC fit with its configured geometry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_eval.tiered_optic_geometry import (  # noqa: E402
    CODEBOOK_SIZE,
    GROUP_DIM,
    RateGeometry,
    SPECIAL_TOKEN_IDS,
    VOCAB_SIZE,
)
from ple_eval.tiered_optic_codec import (  # noqa: E402
    FORMAT_ID,
    METHOD_NAME,
    SCHEMA_VERSION,
    artifact_schema,
    validate_artifact,
)


FIT_WRAPPER = ROOT / "scripts/fit_tiered_optic.py"
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"


class FinalizationContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationContractError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_fit_wrapper() -> Any:
    spec = importlib.util.spec_from_file_location(
        "frequency_adaptive_multiratio_fit_for_finalization", FIT_WRAPPER
    )
    require(spec is not None and spec.loader is not None, "cannot load fit wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_paths(
    config: Mapping[str, Any], recipe: Mapping[str, Any]
) -> dict[str, Path]:
    fit_output = resolve_path(config["fit_output_directory"])
    inputs = recipe["fit_inputs"]
    return {
        "fit_artifact": fit_output / "full_code_fit_artifact.npz",
        "fit_receipt": fit_output / "fit_receipt.json",
        "fit_frequency": resolve_path(inputs["fit_only_frequency"]["path"]),
        "checkpoint_weights": resolve_path(recipe["model"]["weights"]),
        "tokenizer_snapshot": resolve_path(recipe["model"]["tokenizer_snapshot"]),
    }


def contract(path: Path) -> dict[str, Any]:
    fit_wrapper = _load_fit_wrapper()
    config, geometry = fit_wrapper.load_build_config(path)
    _, recipe = fit_wrapper.load_base_recipe(config)
    dependencies = dependency_paths(config, recipe)
    missing = [
        str(value)
        for name, value in dependencies.items()
        if not (value.is_dir() if name == "tokenizer_snapshot" else value.is_file())
    ]
    output = resolve_path(config["compact_output_root"])
    if missing:
        status = "blocked_waiting_for_fit_or_dependency"
    elif output.exists():
        status = "blocked_compact_output_exists"
    else:
        status = "ready_to_finalize"
    return {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-finalization-contract-v1",
        "status": status,
        "rate_geometry": geometry.as_dict(),
        "resident_byte_census": geometry.resident_census(),
        "compact_output_root": str(output),
        "dependencies": {name: str(value) for name, value in dependencies.items()},
        "missing_dependencies": missing,
        "fit_or_evaluation_started": False,
    }


def read_fit_artifact(
    path: Path, geometry: RateGeometry
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        require("metadata" in archive.files, "fit metadata is missing")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {
            name: np.ascontiguousarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    expected = {
        "token_codes_packed_uint8": (
            np.dtype("uint8"),
            (VOCAB_SIZE * geometry.maximum_groups,),
        ),
        "codebooks_fp16": (
            np.dtype("float16"),
            (geometry.maximum_groups, CODEBOOK_SIZE, GROUP_DIM),
        ),
        "decoder_weight_fp16": (
            np.dtype("float16"),
            (geometry.n_layers, geometry.d_ple, geometry.latent_dim),
        ),
        "decoder_bias_fp16": (
            np.dtype("float16"),
            (geometry.n_layers, geometry.d_ple),
        ),
    }
    require(set(arrays) == set(expected), "fit array set differs")
    for name, (dtype, shape) in expected.items():
        require(
            arrays[name].dtype == dtype and tuple(arrays[name].shape) == shape,
            f"fit {name} differs",
        )
    require(
        sum(int(value.nbytes) for value in arrays.values())
        == geometry.raw_fit_resident_bytes,
        "raw fit resident bytes differ",
    )
    require(
        int(metadata.get("latent_dim", -1)) == geometry.latent_dim
        and int(metadata.get("n_groups", -1)) == geometry.maximum_groups,
        "fit metadata geometry differs",
    )
    return metadata, arrays


def quantize_decoder(decoder_fp16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decoder = np.asarray(decoder_fp16, dtype=np.float32)
    maximum = np.max(np.abs(decoder), axis=2)
    scales = np.ascontiguousarray(
        np.where(maximum > 0, maximum / 127.0, 1.0).astype(np.float16)
    )
    require(
        bool(np.isfinite(scales).all()) and bool((scales > 0).all()),
        "decoder scales are invalid",
    )
    quantized = np.ascontiguousarray(
        np.clip(
            np.rint(decoder / scales.astype(np.float32)[..., None]), -127, 127
        ).astype(np.int8)
    )
    return quantized, scales


def _tensor_header(path: Path, key: str) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        require(len(raw) == 8, "invalid safetensors header")
        header_length = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(header_length))
    require(key in header, f"checkpoint lacks {key}")
    return 8 + header_length, header[key]


def exact_checkpoint_special_rows(path: Path, geometry: RateGeometry) -> np.ndarray:
    data_start, metadata = _tensor_header(path, PLE_KEY)
    require(metadata["dtype"] == "BF16", "checkpoint PLE is not BF16")
    require(
        metadata["shape"]
        == [VOCAB_SIZE, geometry.n_layers * geometry.d_ple],
        "checkpoint PLE shape differs",
    )
    tensor_start, tensor_stop = (int(value) for value in metadata["data_offsets"])
    require(
        tensor_stop - tensor_start == geometry.dense_bytes,
        "checkpoint PLE byte span differs",
    )
    row_width = geometry.n_layers * geometry.d_ple
    rows = np.empty((len(SPECIAL_TOKEN_IDS), row_width), dtype="<u2")
    with path.open("rb") as handle:
        for index, token_id in enumerate(SPECIAL_TOKEN_IDS):
            handle.seek(data_start + tensor_start + token_id * row_width * 2)
            raw = handle.read(row_width * 2)
            require(len(raw) == row_width * 2, "short checkpoint special-row read")
            rows[index] = np.frombuffer(raw, dtype="<u2")
    return np.ascontiguousarray(rows, dtype=np.uint16)


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def finalize(path: Path) -> dict[str, Any]:
    fit_wrapper = _load_fit_wrapper()
    config, geometry = fit_wrapper.load_build_config(path)
    _, recipe = fit_wrapper.load_base_recipe(config)
    dependencies = dependency_paths(config, recipe)
    missing = [
        str(value)
        for name, value in dependencies.items()
        if not (value.is_dir() if name == "tokenizer_snapshot" else value.is_file())
    ]
    require(not missing, f"missing dependencies: {missing}")
    output = resolve_path(config["compact_output_root"])
    require(not output.exists(), f"refusing to overwrite {output}")

    fit_receipt = load_json(dependencies["fit_receipt"])
    require(
        fit_receipt.get("status") == "fit_complete",
        "fit receipt status differs",
    )
    receipt_geometry = fit_receipt.get("rate_geometry", {})
    require(
        int(receipt_geometry.get("base_group_count", -1)) == geometry.base_groups
        and int(receipt_geometry.get("extra_group_count", -1)) == geometry.extra_groups,
        "fit receipt geometry differs",
    )
    fit_metadata, source = read_fit_artifact(dependencies["fit_artifact"], geometry)
    model = recipe["model"]
    if config.get("require_source_lineage_metadata", False):
        require(
            fit_metadata.get("source_model_identifier") == model["identifier"]
            and fit_metadata.get("source_model_revision") == model["revision"]
            and fit_metadata.get("source_checkpoint_ple_key") == PLE_KEY,
            "fit artifact source lineage differs",
        )
    frequency = np.fromfile(dependencies["fit_frequency"], dtype="<u8")
    high_ids = fit_wrapper.select_high_rate_token_ids(frequency, geometry)
    codes = source["token_codes_packed_uint8"].reshape(
        VOCAB_SIZE, geometry.maximum_groups
    )
    high_mask = np.zeros(VOCAB_SIZE, dtype=bool)
    high_mask[high_ids.astype(np.int64)] = True
    require(
        bool((codes[~high_mask, geometry.base_groups :] == 0).all()),
        "low-rate extra codes drifted after fit",
    )
    quantized, scales = quantize_decoder(source["decoder_weight_fp16"])
    special_rows = exact_checkpoint_special_rows(
        dependencies["checkpoint_weights"], geometry
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(dependencies["tokenizer_snapshot"]),
        local_files_only=True,
        trust_remote_code=True,
    )
    require(
        tuple(int(value) for value in tokenizer.all_special_ids) == SPECIAL_TOKEN_IDS,
        "tokenizer special IDs differ",
    )
    arrays = {
        "base_token_codes_uint8": np.ascontiguousarray(
            codes[:, : geometry.base_groups], dtype=np.uint8
        ),
        "high_rate_token_ids_uint32": np.ascontiguousarray(high_ids, dtype=np.uint32),
        "high_rate_extra_codes_uint8": np.ascontiguousarray(
            codes[high_ids.astype(np.int64), geometry.base_groups :], dtype=np.uint8
        ),
        "codebooks_fp16": np.ascontiguousarray(source["codebooks_fp16"], dtype=np.float16),
        "decoder_weight_int8": quantized,
        "decoder_scale_per_output_fp16": scales,
        "decoder_bias_fp16": np.ascontiguousarray(
            source["decoder_bias_fp16"], dtype=np.float16
        ),
        "special_rows_bf16_bits_uint16": special_rows,
    }
    metadata = {
        "compressed_ple_format_id": FORMAT_ID,
        "schema_version": SCHEMA_VERSION,
        **artifact_schema(geometry)["fixed_metadata"],
        "resident_byte_census": geometry.resident_census(),
        "group_dim_allocation": fit_metadata.get("group_dim_allocation"),
        "source_model_identifier": model["identifier"],
        "source_model_revision": model["revision"],
        "source_checkpoint_ple_key": PLE_KEY,
    }
    _, census = validate_artifact(metadata, arrays)
    require(census["total"] == geometry.resident_bytes, "final resident bytes differ")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    require(not partial.exists(), f"refusing to reuse {partial}")
    partial.mkdir()
    artifact_path = partial / "compressed_ple.npz"
    with artifact_path.open("xb") as handle:
        np.savez(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )
        handle.flush()
        os.fsync(handle.fileno())
    require(artifact_path.is_file(), "compact artifact was not written")
    metrics = {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-finalization-metrics-v1",
        "method_name": METHOD_NAME,
        "rate_geometry": geometry.as_dict(),
        "resident_byte_census": census,
        "low_rate_extra_codes_all_zero": True,
        "quality_evaluation_run": False,
    }
    write_json_new(partial / "fit_and_finalization_metrics.json", metrics)
    receipt = {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-finalization-receipt-v1",
        "status": "finalized_in_dev_shm",
        "method_name": METHOD_NAME,
        "build_config_path": str(path.resolve()),
        "rate_geometry": geometry.as_dict(),
        "artifact_path": str(output / "compressed_ple.npz"),
        "serialized_file_bytes": int(artifact_path.stat().st_size),
        "resident_byte_census": census,
        "resident_bytes": geometry.resident_bytes,
        "compression_ratio": geometry.compression_ratio,
        "source_model_identifier": model["identifier"],
        "source_model_revision": model["revision"],
        "source_checkpoint_ple_key": PLE_KEY,
        "low_rate_extra_codes_all_zero": True,
        "dense_ple_or_dense_cache_present": False,
        "quality_evaluation_run": False,
    }
    write_json_new(partial / "finalization_receipt.json", receipt)
    os.replace(partial, output)
    require(
        (output / "compressed_ple.npz").is_file()
        and (output / "finalization_receipt.json").is_file(),
        "final outputs are incomplete",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract_parser = subparsers.add_parser("contract")
    contract_parser.add_argument("--build-config", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--build-config", type=Path, required=True)
    args = parser.parse_args()
    result = (
        contract(args.build_config.resolve())
        if args.command == "contract"
        else finalize(args.build_config.resolve())
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
