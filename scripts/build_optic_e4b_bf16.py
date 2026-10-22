#!/usr/bin/env python3
"""Fit OPTIC on BF16 E4B with the broad calibration inputs and host-streamed gates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/fit/optic_e4b_bf16.json"
MODES = ("natural_model_rows_special",)
ACTIVE_CONFIG: dict[str, Any] = {}
ACTIVE_MODE = ""
ACTIVE_DEVICE = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


class LazyLayerGate:
    """Device-facing gate tensor backed by a layer-major host memmap."""

    def __init__(
        self, torch: Any, gate_path: Path, layer: int, natural_positions: int,
        layers: int, d_ple: int, total_positions: int, device: str
    ) -> None:
        self._torch = torch
        self._natural_positions = natural_positions
        self._total_positions = total_positions
        self._d_ple = d_ple
        self._device = torch.device(device)
        full = np.memmap(
            gate_path, dtype="<f2", mode="r",
            shape=(layers, natural_positions, d_ple),
        )
        self._natural = full[layer]

    @property
    def device(self):
        return self._device

    @property
    def shape(self) -> tuple[int, int]:
        return (self._total_positions, self._d_ple)

    def to(self, *args: Any, device: Any = None, **_kwargs: Any):
        requested = device if device is not None else (args[0] if args else self._device)
        if isinstance(requested, (str, self._torch.device)):
            require(self._torch.device(requested) == self._device, "lazy gate device differs")
        return self

    def _fetch(self, indices: np.ndarray):
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= self._total_positions):
            raise IndexError("lazy gate index is outside the training population")
        output = np.ones((indices.size, self._d_ple), dtype=np.float16)
        natural = indices < self._natural_positions
        if bool(np.any(natural)):
            output[natural] = self._natural[indices[natural]]
        return self._torch.from_numpy(output).to(self._device)

    def index_select(self, dim: int, indices: Any):
        require(dim == 0, "lazy gate only supports position-axis index_select")
        values = indices.detach().to(device="cpu", dtype=self._torch.int64).numpy()
        return self._fetch(np.asarray(values, dtype=np.int64))

    def __getitem__(self, key: slice):
        require(isinstance(key, slice), "lazy gate only supports position slices")
        start, stop, step = key.indices(self._total_positions)
        return self._fetch(np.arange(start, stop, step, dtype=np.int64))


def validate_config(config: dict[str, Any]) -> None:
    inputs = config["fit_inputs"]
    builder = config["core_builder"]
    for key, digest_key in (
        ("gate_file", "gate_file_sha256"),
        ("gate_manifest", "gate_manifest_sha256"),
        ("token_manifest", "token_manifest_sha256"),
    ):
        path = Path(inputs[key])
        require(path.is_file(), f"missing fit input: {path}")
        require(sha256_file(path) == inputs[digest_key], f"hash mismatch for {key}")
    builder_path = Path(builder["path"])
    require(builder_path.is_file(), "OPTIC builder is missing")
    require(sha256_file(builder_path) == builder["sha256"], "OPTIC builder hash mismatch")
    weights_path = Path(config["model"]["weights"])
    require(weights_path.is_file(), "model weights are missing")
    require(
        weights_path.resolve().name == config["model"]["weights_sha256"],
        "model weight identity differs",
    )
    gate_manifest = json.loads(Path(inputs["gate_manifest"]).read_text(encoding="utf-8"))
    token_manifest = json.loads(Path(inputs["token_manifest"]).read_text(encoding="utf-8"))
    require(gate_manifest.get("complete") is True, "gate cache is incomplete")
    require(gate_manifest.get("quality_evaluation_allowed") is False, "fit gate is marked evaluable")
    require(gate_manifest.get("gate_file_sha256") == inputs["gate_file_sha256"], "gate binding differs")
    require(token_manifest.get("gate_cache_allowed") is True, "token coverage gate did not pass")
    require(token_manifest.get("selected_token_file_sha256") == inputs["natural_token_file_sha256"], "token binding differs")
    require(
        sha256_file(Path(token_manifest["selected_token_file"]))
        == inputs["natural_token_file_sha256"],
        "natural token file hash differs",
    )
    require(int(token_manifest.get("selected_natural_unique_token_ids", -1)) == int(inputs["natural_unique_token_ids"]), "natural coverage count differs")
    complete_path = Path(token_manifest["complete_tokenizer_row_file"])
    require(sha256_file(complete_path) == inputs["complete_tokenizer_row_file_sha256"], "complete row file hash differs")
    composite = config["variants"]["natural_model_rows_special"]
    require(len(composite["special_token_ids"]) == len(set(composite["special_token_ids"])), "special IDs repeat")
    require(all(0 <= int(value) < int(config["model"]["vocab_size"]) for value in composite["special_token_ids"]), "special ID out of range")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["tokenizer_snapshot"],
        local_files_only=True,
        trust_remote_code=True,
    )
    require(
        list(tokenizer.all_special_ids) == composite["special_token_ids"],
        "tokenizer special IDs differ",
    )
    special_positions = len(composite["special_token_ids"]) * int(composite["repetitions_per_special_token"])
    require(special_positions == int(composite["special_token_positions"]), "special position count differs")
    expected_total = int(inputs["natural_positions"]) + int(config["model"]["vocab_size"]) + special_positions
    require(expected_total == int(composite["total_training_positions"]), "composite position count differs")


def load_broad_training_inputs(
    torch: Any, _safe_open: Any, gate_cache_path: str,
    gate_cache_config_path: str, vocab: int, layers: int, d: int
):
    config = ACTIVE_CONFIG
    inputs = config["fit_inputs"]
    model = config["model"]
    require(Path(gate_cache_path).resolve() == Path(inputs["gate_file"]).resolve(), "unexpected broad gate file")
    require(Path(gate_cache_config_path).resolve() == Path(inputs["gate_manifest"]).resolve(), "unexpected broad gate manifest")
    require((vocab, layers, d) == (int(model["vocab_size"]), int(model["num_layers"]), int(model["d_ple"])), "model geometry differs")
    gate_manifest = json.loads(Path(inputs["gate_manifest"]).read_text(encoding="utf-8"))
    token_manifest_path = Path(inputs["token_manifest"])
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    natural_path = Path(token_manifest["selected_token_file"])
    natural_positions = int(inputs["natural_positions"])
    natural = np.memmap(natural_path, dtype="<u4", mode="r", shape=(natural_positions,))
    token_parts = [np.asarray(natural, dtype=np.int64)]
    synthetic_positions = 0
    special_positions = 0
    if ACTIVE_MODE == "natural_model_rows_special":
        complete_path = Path(token_manifest["complete_tokenizer_row_file"])
        complete = np.memmap(complete_path, dtype="<u4", mode="r", shape=(vocab,))
        token_parts.append(np.asarray(complete, dtype=np.int64))
        variant = config["variants"][ACTIVE_MODE]
        special_ids = np.asarray(variant["special_token_ids"], dtype=np.int64)
        repeated = np.repeat(special_ids, int(variant["repetitions_per_special_token"]))
        token_parts.append(repeated)
        synthetic_positions = vocab + int(repeated.size)
        special_positions = int(repeated.size)
    token_ids = torch.from_numpy(np.concatenate(token_parts)).long().contiguous()
    total_positions = int(token_ids.numel())
    require(total_positions == int(config["variants"][ACTIVE_MODE]["total_training_positions"]), "training position count differs")
    gate_path = Path(inputs["gate_file"])
    gates = [
        LazyLayerGate(
            torch, gate_path, layer, natural_positions, layers, d,
            total_positions, ACTIVE_DEVICE,
        )
        for layer in range(layers)
    ]
    adapter_path = Path(__file__).resolve()
    metadata = {
        "training_gate_cache_file": str(gate_path.resolve()),
        "training_gate_cache_sha256": sha256_file(gate_path),
        "training_gate_cache_config_file": str(Path(inputs["gate_manifest"]).resolve()),
        "training_gate_cache_config_sha256": sha256_file(Path(inputs["gate_manifest"])),
        "training_gate_cache_protocol_id": gate_manifest["protocol_id"],
        "training_gate_cache_matches_locked_protocol": True,
        "training_gate_cache_quality_evaluation_allowed": False,
        "training_gate_cache_corpus_tags": [source["tag"] for source in token_manifest["sources"]],
        "training_gate_cache_manifest": str(token_manifest_path.resolve()),
        "training_gate_cache_manifest_sha256": sha256_file(token_manifest_path),
        "training_gate_cache_positions": total_positions,
        "training_natural_context_positions": natural_positions,
        "training_natural_context_unique_token_ids": int(inputs["natural_unique_token_ids"]),
        "training_natural_context_unique_token_fraction": float(inputs["natural_unique_token_fraction"]),
        "training_model_only_row_positions": synthetic_positions - special_positions,
        "training_special_token_positions": special_positions,
        "training_input_mode": ACTIVE_MODE,
        "training_forward_path_runtime_token_scale": float(inputs["runtime_token_scale"]),
        "training_gate_cache_materialization": (
            "host-resident layer-major fp16 memmap with sampled layer-position blocks transferred on demand"
        ),
        "fit_adapter": str(adapter_path),
        "fit_adapter_sha256": sha256_file(adapter_path),
        "fit_config": str(DEFAULT_CONFIG.resolve()),
        "fit_config_sha256": sha256_file(DEFAULT_CONFIG),
        "core_builder": str(Path(config["core_builder"]["path"]).resolve()),
        "core_builder_sha256": config["core_builder"]["sha256"],
    }
    return token_ids, gates, float(inputs["runtime_token_scale"]), metadata


def patch_generated_metadata(output_dir: Path, materialization: str) -> None:
    """Record how the training gate cache was materialized."""
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["training_gate_cache_materialization"] = materialization
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    temporary_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary_metrics.replace(metrics_path)

    artifact_path = output_dir / "compressed_ple.npz"
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["training_gate_cache_materialization"] = materialization
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary_artifact = artifact_path.with_suffix(".npz.tmp")
    with temporary_artifact.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_artifact.replace(artifact_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-mode", choices=MODES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    require(args.output_dir.resolve().is_relative_to(Path(config["storage"]["output_root"]).resolve()), "output directory is outside the declared storage root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "dense_ple_reconstruction.safetensors",
        args.output_dir / "compressed_ple.npz",
        args.output_dir / "metrics.json",
    ]
    require(not any(path.exists() for path in outputs), "refusing to overwrite an OPTIC output")

    global ACTIVE_CONFIG, ACTIVE_MODE, ACTIVE_DEVICE
    ACTIVE_CONFIG = config
    ACTIVE_MODE = args.input_mode
    ACTIVE_DEVICE = args.device
    builder_path = Path(config["core_builder"]["path"])
    spec = importlib.util.spec_from_file_location("optic_builder", builder_path)
    require(spec is not None and spec.loader is not None, "cannot load the OPTIC builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_exact_gate_cache_forward_path_training_inputs = load_broad_training_inputs
    recipe = config["shared_recipe"]
    model = config["model"]
    inputs = config["fit_inputs"]
    sys.argv = [
        str(builder_path),
        "--it-weights", model["weights"],
        "--num-layers", str(model["num_layers"]),
        "--d-ple", str(model["d_ple"]),
        "--latent-mode", "discrete",
        "--latent-dim", str(recipe["latent_dim"]),
        "--n-groups", str(recipe["n_groups"]),
        "--codebook-size", str(recipe["codebook_size"]),
        "--group-dim-allocation", recipe["group_dim_allocation"],
        "--alternations", str(recipe["alternations"]),
        "--steps-per-alternation", str(recipe["steps_per_alternation"]),
        "--assignment-icd-sweeps", str(recipe["assignment_icd_sweeps"]),
        "--assignment-ridge", str(recipe["assignment_ridge"]),
        "--assignment-chunk-size", str(recipe["assignment_chunk_size"]),
        "--dead-code-min-usage", str(recipe["dead_code_min_usage"]),
        "--kmeans-iterations", str(recipe["kmeans_iterations"]),
        "--metric-eigenvalue-floor", str(recipe["metric_eigenvalue_floor"]),
        "--lr", str(recipe["learning_rate"]),
        "--batch-size", str(recipe["batch_size"]),
        "--training-gate-cache", inputs["gate_file"],
        "--training-gate-cache-config", inputs["gate_manifest"],
        "--heldout-eval-size", str(recipe["heldout_eval_size"]),
        "--heldout-eval-seed", str(recipe["heldout_eval_seed"]),
        "--heldout-eval-interval-steps", str(recipe["heldout_eval_interval_steps"]),
        "--resident-roundtrip-atol", str(recipe["resident_roundtrip_atol"]),
        "--device", args.device,
        "--seed", str(recipe["seed"]),
        "--save-recon", str(outputs[0]),
        "--save-compressed-ple-npz", str(outputs[1]),
        "--output", str(outputs[2]),
    ]
    module.main()
    materialization = (
        "host-resident layer-major fp16 memmap with sampled layer-position blocks transferred on demand"
    )
    patch_generated_metadata(args.output_dir, materialization)
    metrics = json.loads(outputs[2].read_text(encoding="utf-8"))
    require(int(metrics["storage_bytes"]) == int(recipe["expected_resident_bytes"]), "resident byte count differs")
    require(abs(float(metrics["compression_ratio"]) - float(recipe["expected_compression_ratio"])) < 1e-9, "compression ratio differs")
    print(json.dumps({
        "input_mode": args.input_mode,
        "output_dir": str(args.output_dir.resolve()),
        "metrics_sha256": sha256_file(outputs[2]),
        "compressed_ple_sha256": sha256_file(outputs[1]),
        "reconstruction_sha256": sha256_file(outputs[0]),
        "storage_bytes": metrics["storage_bytes"],
        "compression_ratio": metrics["compression_ratio"],
        "elapsed_seconds": metrics["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
