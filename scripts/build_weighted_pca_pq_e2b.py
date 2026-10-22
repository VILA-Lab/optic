#!/usr/bin/env python3
"""Build the output-path-weighted PCA-PQ r100 artifact for the original BF16 E2B checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/fit/weighted_pca_pq_e2b_bf16.json"
SCHEMA = "original-bf16-e2b-output-path-weighted-pca-pq-r100-build-v1"
SOURCE_METHOD = "dout_weighted_pca_rotated_pq_scaled_exceptions"
SOURCE_FORMAT = "dout_weighted_pca_rotated_pq_scaled_exceptions_npz"


def require(value: bool, message: str) -> None:
    if not value:
        raise SystemExit(message)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_contract(path: Path, key: str) -> tuple[str, list[int], int]:
    with path.open("rb") as handle:
        header_bytes = handle.read(8)
        require(len(header_bytes) == 8, f"invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", header_bytes)[0]
        header = json.loads(handle.read(header_size))
    require(key in header, f"missing PLE tensor {key}: {path}")
    spec = header[key]
    offsets = spec["data_offsets"]
    return str(spec["dtype"]), list(spec["shape"]), int(offsets[1]) - int(offsets[0])


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def validate(config_path: Path) -> dict[str, Any]:
    config = read_object(config_path)
    require(config.get("schema_version") == SCHEMA, "schema differs")
    require(config.get("status") == "fixed", "config is not fixed")
    model = config["source_model"]
    require(model["identifier"] == "google/gemma-4-E2B-it", "source is not Original E2B")
    require(model["revision"] == "b446025c61ecea876162774ee247706056963aba", "Original E2B revision differs")
    weights = Path(model["snapshot"]) / model["weight_file"]
    require(weights.is_file() and weights.stat().st_size == int(model["weight_file_bytes"]), "source model file differs")
    require(sha256_file(weights) == model["weight_file_sha256"], "source model SHA-256 differs")
    dtype, shape, byte_count = tensor_contract(weights, model["ple_key"])
    require((dtype, shape, byte_count) == (model["ple_dtype"], model["ple_shape"], model["ple_dense_bytes"]), "source PLE contract differs")

    calibration = config["calibration"]
    assets = [
        ("calibration_text", "calibration_text_sha256"),
        ("frequency_manifest", "frequency_manifest_sha256"),
        ("coordinate_weights", "coordinate_weights_sha256"),
        ("coordinate_weights_metadata", "coordinate_weights_metadata_sha256"),
        ("coordinate_weights_metrics", "coordinate_weights_metrics_sha256"),
        ("gate_cache_config", "gate_cache_config_sha256"),
        ("gate_cache_metrics", "gate_cache_metrics_sha256"),
        ("package_manifest", "package_manifest_sha256"),
    ]
    for path_field, hash_field in assets:
        path = Path(calibration[path_field])
        require(path.is_file(), f"missing calibration asset: {path}")
        require(sha256_file(path) == calibration[hash_field], f"calibration asset SHA-256 differs: {path}")
    require(calibration["calibration_source_weight_file_sha256"] == model["weight_file_sha256"], "calibration source checkpoint differs")

    coordinate_metadata = read_object(Path(calibration["coordinate_weights_metadata"]))
    gate_config = read_object(Path(calibration["gate_cache_config"]))
    package = read_object(Path(calibration["package_manifest"]))
    frequency = read_object(Path(calibration["frequency_manifest"]))
    require(coordinate_metadata.get("source_model_id") == model["identifier"], "coordinate-weight model differs")
    require(coordinate_metadata.get("source_checkpoint_sha256") == model["weight_file_sha256"], "coordinate-weight checkpoint differs")
    require(coordinate_metadata.get("shape") == [model["num_layers"], model["d_ple"]], "coordinate-weight shape differs")
    require(coordinate_metadata.get("positions_total") == calibration["positions"], "coordinate-weight position count differs")
    require(coordinate_metadata.get("input_gate_cache_protocol_id") == "s7_e2b_wikitext2_train_calibration_gate_cache_v1", "coordinate-weight gate protocol differs")
    require(gate_config.get("source_model_id") == model["identifier"], "gate-cache model differs")
    require(gate_config.get("source_checkpoint_sha256") == model["weight_file_sha256"], "gate-cache checkpoint differs")
    require(gate_config.get("n_layers") == model["num_layers"] and gate_config.get("d_ple") == model["d_ple"], "gate-cache geometry differs")
    require(gate_config.get("positions_per_corpus") == calibration["positions"], "gate-cache position count differs")
    require(gate_config.get("quality_evaluation_allowed") is False, "gate cache is marked evaluable")
    require(package.get("protocol_complete") is True and package.get("source_checkpoint_sha256") == model["weight_file_sha256"], "calibration package differs")
    records = frequency.get("records")
    require(isinstance(records, list) and len(records) == 1, "frequency manifest records differ")
    require(records[0].get("corpus_tag") == "wikitext2_train_calibration", "frequency corpus differs")
    require(records[0].get("sha256") == calibration["calibration_text_sha256"], "frequency text binding differs")
    require(records[0].get("tokenizer_tokens") == calibration["positions"], "frequency token count differs")

    import numpy as np
    coordinate_weights = np.load(calibration["coordinate_weights"], mmap_mode="r", allow_pickle=False)
    require(list(coordinate_weights.shape) == [model["num_layers"], model["d_ple"]], "coordinate-weight array shape differs")
    require(str(coordinate_weights.dtype) == "float32", "coordinate-weight dtype differs")

    for path_field, hash_field in [("source_builder", "source_builder_sha256"), ("wrapper_builder", "wrapper_builder_sha256")]:
        path = Path(config[path_field])
        require(path.is_file() and sha256_file(path) == config[hash_field], f"builder differs: {path}")
    recipe = config["recipe"]
    require(recipe["seed"] == 0, "seed differs")
    require((recipe["num_subspaces"], recipe["clusters"], recipe["iters"], recipe["exception_count"]) == (4, 1024, 8, 128), "fixed geometry differs")
    require(recipe["pca_scope"] == "global" and recipe["exception_selector"] == "ppl_text_frequency", "method definition differs")
    require(model["ple_dense_bytes"] / recipe["final_resident_bytes"] == recipe["actual_compression_ratio"], "compression-ratio equation differs")
    require(not Path(config["persistent_output_root"]).exists(), "persistent output root already exists")
    require(not Path(config["capability_output_root"]).exists(), "capability output root already exists")
    return config


def build_commands(config: dict[str, Any], device: str) -> list[list[str]]:
    model, calibration, recipe = config["source_model"], config["calibration"], config["recipe"]
    weights = str(Path(model["snapshot"]) / model["weight_file"])
    persistent = Path(config["persistent_output_root"])
    capability = Path(config["capability_output_root"])
    source = [
        config["python"], config["source_builder"], "--it-weights", weights,
        "--model-path", model["snapshot"], "--num-layers", str(model["num_layers"]), "--d-ple", str(model["d_ple"]),
        "--num-subspaces", str(recipe["num_subspaces"]), "--clusters", str(recipe["clusters"]),
        "--iters", str(recipe["iters"]), "--batch-size", str(recipe["batch_size"]),
        "--eval-batch-size", str(recipe["eval_batch_size"]), "--exception-count", str(recipe["exception_count"]),
        "--exception-scale", str(recipe["exception_scale"]), "--exception-selector", recipe["exception_selector"],
        "--frequency-manifest", calibration["frequency_manifest"], "--frequency-corpus-tag", recipe["frequency_corpus_tag"],
        "--coord-weights", calibration["coordinate_weights"], "--coord-weights-provenance", calibration["path_weight_source"],
        "--coord-weights-metadata", calibration["coordinate_weights_metadata"], "--coord-weight-alpha", str(recipe["coordinate_weight_alpha"]),
        "--device", device, "--seed", str(recipe["seed"]), "--resident-roundtrip-atol", "0.002",
        "--save-recon", str(capability / "source/dense_ple_reconstruction.safetensors"),
        "--save-compressed-ple-npz", str(persistent / "source/compressed_ple.npz"),
        "--output", str(persistent / "source/metrics.json"),
    ]
    wrapper = [
        config["python"], config["wrapper_builder"], "--it-weights", weights,
        "--source-dense-ple-reconstruction", str(capability / "source/dense_ple_reconstruction.safetensors"),
        "--source-compressed-ple-npz", str(persistent / "source/compressed_ple.npz"),
        "--source-method", SOURCE_METHOD, "--expected-source-method", SOURCE_METHOD,
        "--expected-source-format-id", SOURCE_FORMAT, "--source-config-id", config["source_config_id"],
        "--source-resident-bytes", str(recipe["source_resident_bytes"]), "--calibration-mode", recipe["calibration_mode"],
        "--num-layers", str(model["num_layers"]), "--d-ple", str(model["d_ple"]),
        "--alpha", str(recipe["calibration_alpha"]), "--eps", str(recipe["calibration_eps"]),
        "--resident-roundtrip-atol", "0", "--save-recon", str(capability / "wrapper/dense_ple_reconstruction.safetensors"),
        "--save-compressed-ple-npz", str(persistent / "wrapper/compressed_ple.npz"),
        "--output", str(persistent / "wrapper/metrics.json"),
    ]
    return [source, wrapper]


def finalize(config_path: Path, config: dict[str, Any], commands: list[list[str]]) -> None:
    persistent = Path(config["persistent_output_root"])
    capability = Path(config["capability_output_root"])
    source_artifact = persistent / "source/compressed_ple.npz"
    source_metrics_path = persistent / "source/metrics.json"
    wrapper_artifact = persistent / "wrapper/compressed_ple.npz"
    wrapper_metrics_path = persistent / "wrapper/metrics.json"
    reconstruction = capability / "wrapper/dense_ple_reconstruction.safetensors"
    source_metrics = read_object(source_metrics_path)
    wrapper_metrics = read_object(wrapper_metrics_path)
    recipe, model = config["recipe"], config["source_model"]
    require(source_metrics.get("storage_bytes") == recipe["source_resident_bytes"], "source resident bytes differ")
    require(source_metrics.get("seed") == recipe["seed"], "source seed differs")
    require(source_metrics.get("resident_file_roundtrip_passed") is True, "source round-trip failed")
    require(wrapper_metrics.get("storage_bytes") == recipe["final_resident_bytes"], "wrapper resident bytes differ")
    require(wrapper_metrics.get("compression_ratio") == recipe["actual_compression_ratio"], "wrapper ratio differs")
    require(wrapper_metrics.get("resident_file_roundtrip_passed") is True, "wrapper round-trip failed")
    require(reconstruction.is_file(), "wrapper capability reconstruction is absent")
    reconstruction_sha = sha256_file(reconstruction)
    require(reconstruction_sha == wrapper_metrics.get("dense_recon_sha256"), "wrapper reconstruction SHA-256 differs")
    method = {
        "schema_version": "ple-method-config-v1",
        "status": "ready_for_evaluation",
        "method_id": "original_bf16_e2b_output_path_weighted_pca_pq_r100_seed0",
        "label": f"Output-path-weighted PCA-PQ (global bank; {recipe['actual_compression_ratio']:.3f}x)",
        "publication_name": config["publication_method"],
        "classification": "our_method",
        "method_family": "output_path_weighted_pca_pq",
        "data_access_regime": "wikitext2_train_calibration",
        "ratio_bucket": "r100",
        "retention_reference_method_key": "dense_gemma4_e2b_it",
        "capability_mode": "dense_reconstruction_injection",
        "reconstruction_path": str(reconstruction),
        "reconstruction_sha256": reconstruction_sha,
        "compressed_artifact_path": str(wrapper_artifact),
        "compressed_artifact_sha256": sha256_file(wrapper_artifact),
        "compressed_artifact_file_bytes": wrapper_artifact.stat().st_size,
        "format_id": wrapper_metrics["compressed_ple_format_id"],
        "builder_path": config["wrapper_builder"],
        "builder_sha256": config["wrapper_builder_sha256"],
        "training_metrics_path": str(wrapper_metrics_path),
        "training_metrics_sha256": sha256_file(wrapper_metrics_path),
        "source_format_id": source_metrics["compressed_ple_format_id"],
        "source_artifact_path": str(source_artifact),
        "source_artifact_sha256": sha256_file(source_artifact),
        "source_artifact_file_bytes": source_artifact.stat().st_size,
        "source_builder_path": config["source_builder"],
        "source_builder_sha256": config["source_builder_sha256"],
        "source_training_metrics_path": str(source_metrics_path),
        "source_training_metrics_sha256": sha256_file(source_metrics_path),
        "reported_reconstruction_nmse": wrapper_metrics["target_nmse"],
        "ple_resident_bytes": recipe["final_resident_bytes"],
        "ple_compression_ratio": recipe["actual_compression_ratio"],
        "compression_ratio_scope": "source compact tensors plus persistent layer-coordinate scale tensor",
        "deployment_mode": "nested_compact_representation_with_audited_decode",
        "roundtrip_receipt_path": config["roundtrip_receipt_output"],
        "runtime_receipt_path": config["runtime_receipt_output"],
        "seed": recipe["seed"],
        "build_config_path": str(config_path),
        "build_receipt_path": config["build_receipt_output"],
    }
    roundtrip_receipt = {
        "schema_version": "output-path-weighted-pca-pq-resident-roundtrip-v1",
        "status": "passed",
        "method_id": method["method_id"],
        "artifact_path": str(source_artifact),
        "artifact_sha256": method["source_artifact_sha256"],
        "persistent_resident_bytes": recipe["final_resident_bytes"],
        "source_resident_file_roundtrip_passed": source_metrics["resident_file_roundtrip_passed"],
        "source_resident_file_roundtrip_max_abs_diff": source_metrics["resident_file_roundtrip_max_abs_diff"],
        "wrapper_resident_file_roundtrip_passed": wrapper_metrics["resident_file_roundtrip_passed"],
        "wrapper_resident_file_roundtrip_max_abs_diff": wrapper_metrics["resident_file_roundtrip_max_abs_diff"],
        "reconstruction_path": str(reconstruction),
        "reconstruction_sha256": reconstruction_sha,
        "benchmark_inputs_used": False,
    }
    runtime_receipt = {
        "schema_version": "output-path-weighted-pca-pq-materialization-runtime-v1",
        "status": "passed",
        "method_id": method["method_id"],
        "artifact_path": str(source_artifact),
        "artifact_sha256": method["source_artifact_sha256"],
        "persistent_resident_bytes": recipe["final_resident_bytes"],
        "capability_materialization_path": str(reconstruction),
        "capability_materialization_sha256": reconstruction_sha,
        "source_builder_elapsed_seconds": source_metrics["elapsed_seconds"],
        "wrapper_builder_elapsed_seconds": wrapper_metrics["elapsed_seconds"],
        "scope": "builder resident-NPZ decode and audited capability materialization",
        "benchmark_inputs_used": False,
    }
    atomic_json(Path(config["roundtrip_receipt_output"]), roundtrip_receipt)
    atomic_json(Path(config["runtime_receipt_output"]), runtime_receipt)
    atomic_json(Path(config["method_config_output"]), method)
    receipt = {
        "schema_version": "original-bf16-e2b-output-path-weighted-pca-pq-r100-build-receipt-v1",
        "status": "completed",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_model_id": model["identifier"],
        "source_model_revision": model["revision"],
        "source_weight_file_sha256": model["weight_file_sha256"],
        "calibration_source_weight_file_sha256": config["calibration"]["calibration_source_weight_file_sha256"],
        "seed": recipe["seed"],
        "commands": commands,
        "outputs": {
            "source_artifact": {"path": str(source_artifact), "sha256": method["source_artifact_sha256"], "file_bytes": source_artifact.stat().st_size},
            "wrapper_artifact": {"path": str(wrapper_artifact), "sha256": method["compressed_artifact_sha256"], "file_bytes": wrapper_artifact.stat().st_size},
            "wrapper_reconstruction": {"path": str(reconstruction), "sha256": reconstruction_sha, "file_bytes": reconstruction.stat().st_size},
            "method_config": {"path": config["method_config_output"], "sha256": sha256_file(Path(config["method_config_output"]))}
        },
        "resident_bytes": recipe["final_resident_bytes"],
        "compression_ratio": recipe["actual_compression_ratio"],
        "source_roundtrip_passed": True,
        "wrapper_roundtrip_passed": True
    }
    atomic_json(Path(config["build_receipt_output"]), receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--print-commands", action="store_true")
    action.add_argument("--build", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = validate(config_path)
    commands = build_commands(config, args.device)
    if args.validate_only:
        print(json.dumps({"status": "valid", "source": "Original BF16 E2B", "seed": 0, "geometry": "M4-K1024-N128-I8"}, indent=2))
        return
    if args.print_commands:
        print(json.dumps(commands, indent=2))
        return
    for command in commands:
        subprocess.run(command, check=True, cwd=ROOT)
    finalize(config_path, config, commands)


if __name__ == "__main__":
    main()
