#!/usr/bin/env python3
"""Fit OPTIC on the E2B BF16 checkpoint and on the QAT parents of the W4A16 checkpoints.

The runner applies the OPTIC builder to the broad calibration inputs prepared by
scripts/build_optic_e4b_bf16.py, with deterministic CUDA settings. Training runs
for alternations x steps_per_alternation optimizer steps and exports the final
state.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/fit/optic_e4b_w4a16.json"
BASE_RUNNER = ROOT / "scripts/build_optic_e4b_bf16.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    require(count == 1, f"builder anchor {label!r} occurs {count} times, expected once")
    return source.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reproducibility = config["reproducibility"]
    require(reproducibility["deterministic_algorithms"] is True, "deterministic algorithms must be enabled")
    require(reproducibility["cublas_workspace_config"] == ":4096:8", "cuBLAS workspace contract differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = reproducibility["cublas_workspace_config"]
    base = load_module("optic_input_adapter", BASE_RUNNER)
    # Check downloaded weight copies by content as well as cache symlinks.
    validation_config = json.loads(json.dumps(config))
    weights_path = Path(config["model"]["weights"])
    expected_weights_sha256 = str(config["model"]["weights_sha256"])
    if weights_path.resolve().name != expected_weights_sha256:
        require(
            sha256_file(weights_path) == expected_weights_sha256,
            "materialized model weight hash differs",
        )
        validation_config["model"]["weights_sha256"] = weights_path.resolve().name
    base.validate_config(validation_config)
    recipe = config["shared_recipe"]
    require(int(recipe["alternations"]) * int(recipe["steps_per_alternation"]) == int(recipe["optimizer_steps"]), "training horizon differs")
    require(recipe["lr_schedule"] == "constant", "learning-rate schedule differs")

    output_dir = Path(config["output"]["directory"])
    require(output_dir.resolve().is_relative_to(Path(config["storage"]["output_root"]).resolve()), "output directory is outside its declared storage root")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "reconstruction": output_dir / "dense_ple_reconstruction.safetensors",
        "artifact": output_dir / "compressed_ple.npz",
        "metrics": output_dir / "metrics.json",
    }
    require(not any(path.exists() for path in outputs.values()), "refusing to overwrite completed output")

    builder_path = Path(config["core_builder"]["path"])
    source = builder_path.read_text(encoding="utf-8")
    require(sha256_file(builder_path) == config["core_builder"]["sha256"], "OPTIC builder hash differs")
    transformed = replace_once(
        source,
        "    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(args.seed)\n    device = torch.device(args.device)\n",
        "    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(args.seed)\n"
        "    torch.use_deterministic_algorithms(True)\n"
        "    torch.backends.cudnn.benchmark = False\n"
        "    torch.backends.cudnn.deterministic = True\n"
        "    torch.backends.cuda.matmul.allow_tf32 = False\n"
        "    device = torch.device(args.device)\n",
        "deterministic CUDA settings",
    )
    module = types.ModuleType("optic_builder")
    module.__file__ = str(builder_path)
    exec(compile(transformed, str(builder_path), "exec"), module.__dict__)

    base.ACTIVE_CONFIG = config
    base.ACTIVE_MODE = config["training_input_mode"]
    base.ACTIVE_DEVICE = args.device
    base.DEFAULT_CONFIG = args.config.resolve()
    module.build_exact_gate_cache_forward_path_training_inputs = base.load_broad_training_inputs
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
        "--lr-schedule", recipe["lr_schedule"],
        "--batch-size", str(recipe["batch_size"]),
        "--training-gate-cache", inputs["gate_file"],
        "--training-gate-cache-config", inputs["gate_manifest"],
        "--heldout-eval-size", str(recipe["heldout_eval_size"]),
        "--heldout-eval-seed", str(recipe["heldout_eval_seed"]),
        "--heldout-eval-interval-steps", str(recipe["heldout_eval_interval_steps"]),
        "--resident-roundtrip-atol", str(recipe["resident_roundtrip_atol"]),
        "--device", args.device,
        "--seed", str(recipe["seed"]),
        "--save-recon", str(outputs["reconstruction"]),
        "--save-compressed-ple-npz", str(outputs["artifact"]),
        "--output", str(outputs["metrics"]),
    ]
    module.main()
    base.patch_generated_metadata(
        output_dir,
        "host-resident layer-major fp16 memmap with sampled layer-position blocks transferred on demand",
    )
    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    metrics.update({
        "schema_version": "optic-fit-v1",
        "method_full_name": "OPTIC",
        "reproducibility": reproducibility,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
    })
    temporary = outputs["metrics"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(outputs["metrics"])
    print(json.dumps({
        "method_full_name": metrics["method_full_name"],
        "compression_ratio": metrics["compression_ratio"],
        "metrics": str(outputs["metrics"].resolve()),
        "artifact": str(outputs["artifact"].resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
