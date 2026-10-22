#!/usr/bin/env python3
"""Configure and optionally run the two-stage PCA-PQ recipe on a QAT-parent PLE."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/fit/weighted_pca_pq_e4b_w4a16.json"
SUPPORTED_SCHEMAS = {
    "qat-e4b-parent-source-path-weighted-pca-pq-r100-build-v1",
    "qat-e2b-parent-source-path-weighted-pca-pq-r100-build-v1",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_region(path: Path, key: str) -> tuple[int, int, str, list[int]]:
    with path.open("rb") as handle:
        header_bytes = handle.read(8)
        require(len(header_bytes) == 8, f"invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", header_bytes)[0]
        header = json.loads(handle.read(header_size))
    require(key in header, f"{path} missing {key}")
    spec = header[key]
    offsets = spec["data_offsets"]
    return 8 + header_size + int(offsets[0]), int(offsets[1]) - int(offsets[0]), str(spec["dtype"]), list(spec["shape"])


def regions_equal(left: Path, left_start: int, right: Path, right_start: int, count: int) -> bool:
    chunk_bytes = 64 * 1024 * 1024
    with left.open("rb") as lhs, right.open("rb") as rhs:
        lhs.seek(left_start)
        rhs.seek(right_start)
        remaining = count
        while remaining:
            wanted = min(chunk_bytes, remaining)
            if lhs.read(wanted) != rhs.read(wanted):
                return False
            remaining -= wanted
    return True


def load_and_validate(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["schema_version"] in SUPPORTED_SCHEMAS, "schema differs")
    require(config["status"] == "fixed", "build status differs")
    parent = Path(config["source_snapshot"]) / "model.safetensors"
    child = Path(config["matched_child_snapshot"]) / "model.safetensors"
    require(parent.stat().st_size == int(config["source_model_bytes"]), "parent model bytes differ")
    require(child.stat().st_size == int(config["matched_child_model_bytes"]), "child model bytes differ")
    key = str(config["ple_key"])
    parent_start, parent_bytes, parent_dtype, parent_shape = tensor_region(parent, key)
    child_start, child_bytes, child_dtype, child_shape = tensor_region(child, key)
    expected_shape = list(config["ple_shape"])
    expected_bytes = int(config["ple_dense_bytes"])
    require((parent_dtype, parent_shape, parent_bytes) == ("BF16", expected_shape, expected_bytes), "parent PLE contract differs")
    require((child_dtype, child_shape, child_bytes) == ("BF16", expected_shape, expected_bytes), "child PLE contract differs")
    require(regions_equal(parent, parent_start, child, child_start, expected_bytes), "parent and child PLE tensors differ")
    calibration = config["calibration"]
    for path_field, hash_field in [
        ("frequency_manifest", "frequency_manifest_sha256"),
        ("calibration_text", "calibration_text_sha256"),
        ("coordinate_weights", "coordinate_weights_sha256"),
        ("coordinate_weights_metadata", "coordinate_weights_metadata_sha256"),
    ]:
        path = Path(calibration[path_field])
        require(path.is_file(), f"missing calibration asset: {path}")
        require(sha256_file(path) == calibration[hash_field], f"calibration asset hash differs: {path}")
    for path_field, hash_field in [
        ("source_builder", "source_builder_sha256"),
        ("wrapper_builder", "wrapper_builder_sha256"),
    ]:
        path = Path(config[path_field])
        require(path.is_file(), f"missing builder: {path}")
        require(sha256_file(path) == config[hash_field], f"builder hash differs: {path}")
    recipe = config["recipe"]
    raw_bytes = int(config["ple_dense_bytes"])
    require(raw_bytes / int(recipe["final_resident_bytes"]) == float(recipe["actual_compression_ratio"]), "ratio equation differs")
    if config["schema_version"] == "qat-e4b-parent-source-path-weighted-pca-pq-r100-build-v1":
        require(calibration["qat_parent_path_refit"] is False, "the E4B recipe reuses the original E4B path weights")
    else:
        require(calibration["qat_parent_path_refit"] is True, "the E2B recipe must use QAT-parent-derived coordinate weights")
    return config


def build_commands(config: dict[str, Any], device: str) -> list[list[str]]:
    recipe = config["recipe"]
    calibration = config["calibration"]
    output_root = Path(config["output_root"])
    source_dir = output_root / "source"
    wrapper_dir = output_root / "wrapper"
    parent_weights = str(Path(config["source_snapshot"]) / "model.safetensors")
    source = [
        config["python"], config["source_builder"],
        "--it-weights", parent_weights,
        "--model-path", config["source_snapshot"],
        "--num-layers", str(config["num_layers"]), "--d-ple", str(config["d_ple"]),
        "--num-subspaces", str(recipe["num_subspaces"]), "--clusters", str(recipe["clusters"]),
        "--iters", str(recipe["iters"]), "--batch-size", str(recipe["batch_size"]),
        "--eval-batch-size", str(recipe["eval_batch_size"]),
        "--exception-count", str(recipe["exception_count"]), "--exception-scale", str(recipe["exception_scale"]),
        "--exception-selector", recipe["exception_selector"],
        "--frequency-manifest", calibration["frequency_manifest"],
        "--frequency-corpus-tag", recipe["frequency_corpus_tag"],
        "--coord-weights", calibration["coordinate_weights"],
        "--coord-weights-provenance", calibration["path_weight_source"],
        "--coord-weights-metadata", calibration["coordinate_weights_metadata"],
        "--coord-weight-alpha", str(recipe["coordinate_weight_alpha"]),
        "--device", device, "--seed", str(recipe["seed"]),
        "--resident-roundtrip-atol", "0.002",
        "--save-recon", str(source_dir / "dense_ple_reconstruction.safetensors"),
        "--save-compressed-ple-npz", str(source_dir / "compressed_ple.npz"),
        "--output", str(source_dir / "metrics.json"),
    ]
    wrapper = [
        config["python"], config["wrapper_builder"],
        "--it-weights", parent_weights,
        "--source-dense-ple-reconstruction", str(source_dir / "dense_ple_reconstruction.safetensors"),
        "--source-compressed-ple-npz", str(source_dir / "compressed_ple.npz"),
        "--source-method", "dout_weighted_pca_rotated_pq_scaled_exceptions",
        "--expected-source-method", "dout_weighted_pca_rotated_pq_scaled_exceptions",
        "--expected-source-format-id", "dout_weighted_pca_rotated_pq_scaled_exceptions_npz",
        "--source-config-id", str(config.get("source_config_id", "qat_e4b_parent_source_dout_pca_pq_m4_k1024_n128_iters8_seed0")),
        "--source-resident-bytes", str(recipe["source_resident_bytes"]),
        "--calibration-mode", recipe["calibration_mode"],
        "--num-layers", str(config["num_layers"]), "--d-ple", str(config["d_ple"]),
        "--alpha", str(recipe["calibration_alpha"]), "--eps", str(recipe["calibration_eps"]),
        "--resident-roundtrip-atol", "0",
        "--save-recon", str(wrapper_dir / "dense_ple_reconstruction.safetensors"),
        "--save-compressed-ple-npz", str(wrapper_dir / "compressed_ple.npz"),
        "--output", str(wrapper_dir / "metrics.json"),
    ]
    return [source, wrapper]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--print-commands", action="store_true")
    action.add_argument("--build", action="store_true")
    args = parser.parse_args()
    config = load_and_validate(args.config.resolve())
    commands = build_commands(config, args.device)
    if args.validate_only:
        print(json.dumps({"status": "valid", "config": str(args.config.resolve()), "parent_child_ple": "byte_identical"}, indent=2))
        return
    if args.print_commands:
        print(json.dumps(commands, indent=2))
        return
    output_root = Path(config["output_root"])
    require(not output_root.exists(), f"refusing to overwrite output root: {output_root}")
    for command in commands:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
