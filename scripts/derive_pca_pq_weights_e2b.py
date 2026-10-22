#!/usr/bin/env python3
"""Derive E2B QAT-parent output-path weights from the broad gate cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value["schema_version"] == "e2b-qat-parent-broad-output-path-coordinate-weights-v1", "schema differs")
    require(value["status"] == "fixed", "status differs")
    gate = value["gate_cache"]
    shape = tuple(int(item) for item in gate["shape"])
    require(shape == (35, 786432, 256), "gate geometry differs")
    gate_path = Path(gate["file"])
    gate_config_path = Path(gate["config"])
    weights_path = Path(value["model"]["weights"])
    require(gate_path.is_file() and gate_config_path.is_file() and weights_path.is_file(), "input is missing")
    require(gate_path.stat().st_size == int(np.prod(shape)) * 2, "gate bytes differ")
    gate_config = json.loads(gate_config_path.read_text(encoding="utf-8"))
    require(
        gate_config.get("complete") is True
        and gate_config.get("quality_evaluation_allowed") is False
        and gate_config.get("model_identifier") == value["model"]["identifier"]
        and gate_config.get("model_revision") == value["model"]["revision"]
        and tuple(gate_config.get("gate_shape", [])) == shape
        and gate_config.get("gate_file_sha256") == gate["sha256"],
        "gate provenance differs",
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    output = Path(config["output_directory"])
    if args.contract:
        require(not output.exists(), "output already exists")
        print(json.dumps({"status": "ready_to_enqueue", "output_directory": str(output)}, indent=2))
        return
    require(not output.exists(), "refusing to overwrite output directory")
    gate_path = Path(config["gate_cache"]["file"])
    require(sha256_file(gate_path) == config["gate_cache"]["sha256"], "gate SHA-256 differs")
    shape = tuple(int(item) for item in config["gate_cache"]["shape"])
    gates = np.memmap(gate_path, dtype="<f2", mode="r", shape=shape)
    result = np.empty((shape[0], shape[2]), dtype=np.float64)
    key_pattern = config["model"]["projection_key_pattern"]
    with safe_open(config["model"]["weights"], framework="pt", device="cpu") as checkpoint:
        for layer in range(shape[0]):
            gate_square_sum = np.zeros(shape[2], dtype=np.float64)
            for start in range(0, shape[1], 16384):
                block = np.asarray(gates[layer, start : start + 16384], dtype=np.float32)
                gate_square_sum += np.sum(block * block, axis=0, dtype=np.float64)
            projection = checkpoint.get_tensor(key_pattern.format(layer=layer)).float().numpy()
            require(projection.ndim == 2 and projection.shape[1] == shape[2], f"projection geometry differs at layer {layer}")
            result[layer] = gate_square_sum / shape[1] * np.sum(projection.astype(np.float64) ** 2, axis=0)
    result /= max(float(result.mean()), 1e-30)
    floor = float(config["recipe"]["coordinate_weight_floor"])
    result = np.maximum(result, floor)
    result /= max(float(result.mean()), 1e-30)
    array = result.astype(np.float32)
    require(array.shape == (35, 256) and np.isfinite(array).all() and float(array.min()) > 0, "invalid coordinate weights")
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    partial.mkdir(parents=True)
    array_path = partial / "dout_coordinate_weights.npy"
    metadata_path = partial / "dout_coordinate_weights_metadata.json"
    np.save(array_path, array)
    metadata = {
        "schema_version": "e2b-qat-parent-broad-output-path-coordinate-weights-receipt-v1",
        "status": "passed",
        "model_identifier": config["model"]["identifier"],
        "model_revision": config["model"]["revision"],
        "gate_file": str(gate_path),
        "gate_file_sha256": config["gate_cache"]["sha256"],
        "shape": [35, 256],
        "dtype": "float32",
        "formula": config["recipe"]["formula"],
        "coordinate_weight_floor": floor,
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "benchmark_inputs_used": False,
        "quality_results_used": False,
        "output_file": str(output / array_path.name),
        "output_file_sha256": sha256_file(array_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    partial.rename(output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
