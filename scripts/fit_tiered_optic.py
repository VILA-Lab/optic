#!/usr/bin/env python3
"""Fit frequency-tiered OPTIC with checkpoint-specific geometry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import os
import tempfile

# Directory for large intermediate files.
SCRATCH_DIR = os.path.realpath(os.environ.get("PLE_SCRATCH_DIR") or tempfile.gettempdir())

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_eval.tiered_optic_geometry import (  # noqa: E402
    CODEBOOK_SIZE,
    D_PLE,
    GROUP_DIM,
    LAYERS,
    RateGeometry,
    SPECIAL_TOKEN_IDS,
    VOCAB_SIZE,
    geometry_from_mapping,
)


SCHEMA_VERSION = "frequency-adaptive-variable-rate-multiratio-build-v1"
BASE_RUNNER = ROOT / (
    'scripts/fit_tiered_optic_e4b.py'
)
BROAD_RUNNER = ROOT / "scripts/build_optic_e4b_bf16.py"
CORE_BUILDER = ROOT / "builders/methods/optic.py"
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"


class BuildContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildContractError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_build_config(path: Path) -> tuple[dict[str, Any], RateGeometry]:
    config = load_json(path.resolve())
    require(config.get("schema_version") == SCHEMA_VERSION, "build schema differs")
    geometry = geometry_from_mapping(config["rate_geometry"])
    require(
        int(config.get("target_ratio", -1)) == geometry.target_ratio,
        "top-level target ratio differs",
    )
    require(
        geometry.target_ratio in (20, 40, 100, 500, 1000),
        "only the declared 20x/40x/100x/500x/1000x paths are supported",
    )
    fit_output = resolve_path(config["fit_output_directory"])
    compact_output = resolve_path(config["compact_output_root"])
    require(fit_output.is_relative_to(Path(SCRATCH_DIR)), "fit output must stay in PLE_SCRATCH_DIR")
    require(compact_output.is_relative_to(Path(SCRATCH_DIR)), "compact output must stay in PLE_SCRATCH_DIR")
    require(fit_output != compact_output, "fit and compact outputs must differ")
    return config, geometry


def load_base_recipe(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = resolve_path(config["base_fit_recipe_path"])
    return path, load_json(path)


def dependency_paths(config: Mapping[str, Any], recipe: Mapping[str, Any]) -> dict[str, Path]:
    inputs = recipe["fit_inputs"]
    model = recipe["model"]
    return {
        "base_runner": BASE_RUNNER,
        "broad_runner": BROAD_RUNNER,
        "core_builder": CORE_BUILDER,
        "checkpoint_weights": resolve_path(model["weights"]),
        "tokenizer_json": resolve_path(model["tokenizer_snapshot"]) / "tokenizer.json",
        "gate_file": resolve_path(inputs["gate_file"]["path"]),
        "gate_manifest": resolve_path(inputs["gate_manifest"]["path"]),
        "natural_token_file": resolve_path(inputs["natural_token_file"]["path"]),
        "complete_tokenizer_rows": resolve_path(
            inputs["complete_tokenizer_rows"]["path"]
        ),
        "fit_frequency": resolve_path(inputs["fit_only_frequency"]["path"]),
    }


def validate_recipe_geometry(
    recipe: Mapping[str, Any], geometry: RateGeometry
) -> dict[str, Any]:
    model = recipe["model"]
    require(
        int(model.get("vocab_size", -1)) == VOCAB_SIZE
        and int(model.get("num_layers", -1)) == geometry.n_layers
        and int(model.get("d_ple", -1)) == geometry.d_ple
        and int(model.get("dense_ple_bytes", -1)) == geometry.dense_bytes,
        "fit recipe model geometry differs from rate geometry",
    )
    return {
        "vocab_size": VOCAB_SIZE,
        "num_layers": geometry.n_layers,
        "d_ple": geometry.d_ple,
        "dense_ple_bytes": geometry.dense_bytes,
    }


def validate_source_lineage(
    recipe: Mapping[str, Any],
    dependencies: Mapping[str, Path],
    geometry: RateGeometry,
) -> dict[str, Any]:
    """Bind checkpoint rows and gate activations to one declared model lineage."""

    model = recipe["model"]
    inputs = recipe["fit_inputs"]
    with dependencies["checkpoint_weights"].open("rb") as handle:
        raw = handle.read(8)
        require(len(raw) == 8, "invalid checkpoint safetensors header")
        header_length = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(header_length))
    require(PLE_KEY in header, f"checkpoint lacks {PLE_KEY}")
    ple = header[PLE_KEY]
    offsets = [int(value) for value in ple["data_offsets"]]
    require(
        ple.get("dtype") == "BF16"
        and ple.get("shape") == [VOCAB_SIZE, geometry.n_layers * geometry.d_ple]
        and offsets[1] - offsets[0] == geometry.dense_bytes,
        "checkpoint PLE dtype, shape, or byte span differs",
    )

    gate = load_json(dependencies["gate_manifest"])
    expected_positions = int(inputs["natural_token_file"]["positions"])
    require(
        gate.get("model_identifier") == model["identifier"]
        and gate.get("model_revision") == model["revision"],
        "gate cache model lineage differs from the fit checkpoint",
    )
    require(
        resolve_path(gate["gate_file"]) == dependencies["gate_file"]
        and resolve_path(gate["token_file"]) == dependencies["natural_token_file"],
        "gate cache or calibration-token path differs",
    )
    require(
        gate.get("complete") is True
        and gate.get("quality_evaluation_allowed") is False
        and gate.get("gate_shape")
        == [geometry.n_layers, expected_positions, geometry.d_ple]
        and int(gate.get("gate_file_bytes", -1))
        == geometry.n_layers * expected_positions * geometry.d_ple * 2,
        "gate cache role, geometry, or byte count differs",
    )
    expected_token_sha = inputs["natural_token_file"].get("sha256")
    if expected_token_sha is not None:
        require(
            gate.get("token_file_sha256") == expected_token_sha,
            "gate cache calibration-token identity differs",
        )
    return {
        "model_identifier": model["identifier"],
        "model_revision": model["revision"],
        "ple_dtype": ple["dtype"],
        "ple_shape": ple["shape"],
        "gate_shape": gate["gate_shape"],
        "benchmark_inputs_used": False,
    }


def contract(path: Path) -> dict[str, Any]:
    config, geometry = load_build_config(path)
    recipe_path = resolve_path(config["base_fit_recipe_path"])
    dependencies: dict[str, Path] = {
        "base_fit_recipe": recipe_path,
        "base_runner": BASE_RUNNER,
        "broad_runner": BROAD_RUNNER,
        "core_builder": CORE_BUILDER,
    }
    if recipe_path.is_file():
        recipe = load_json(recipe_path)
        validate_recipe_geometry(recipe, geometry)
        dependencies.update(dependency_paths(config, recipe))
    missing = [str(value) for value in dependencies.values() if not value.is_file()]
    output = resolve_path(config["fit_output_directory"])
    if missing:
        status = "blocked_missing_dependency"
    elif output.exists():
        status = "blocked_fit_output_exists"
    else:
        status = "ready_to_fit"
    result = {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-fit-contract-v1",
        "status": status,
        "rate_geometry": geometry.as_dict(),
        "resident_byte_census": geometry.resident_census(),
        "fit_output_directory": str(output),
        "dependencies": {name: str(value) for name, value in dependencies.items()},
        "missing_dependencies": missing,
        "fit_or_evaluation_started": False,
    }
    if not missing:
        result["source_lineage"] = validate_source_lineage(
            load_json(recipe_path), dependencies, geometry
        )
    return result


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_high_rate_token_ids(
    frequency: np.ndarray, geometry: RateGeometry
) -> np.ndarray:
    require(
        frequency.dtype == np.dtype("<u8") and frequency.shape == (VOCAB_SIZE,),
        "fit frequency geometry differs",
    )
    ordinary_mask = np.ones(VOCAB_SIZE, dtype=bool)
    ordinary_mask[np.asarray(SPECIAL_TOKEN_IDS, dtype=np.int64)] = False
    ordinary_ids = np.arange(VOCAB_SIZE, dtype=np.int64)[ordinary_mask]
    order = np.lexsort((ordinary_ids, -frequency[ordinary_ids].astype(np.int64)))
    selected = np.sort(ordinary_ids[order[: geometry.high_rate_count]]).astype(np.uint32)
    require(
        selected.shape == (geometry.high_rate_count,)
        and np.unique(selected).size == geometry.high_rate_count,
        "high-rate selection geometry differs",
    )
    return np.ascontiguousarray(selected)


def run_fit(path: Path, device_name: str) -> dict[str, Any]:
    import torch

    config, geometry = load_build_config(path)
    _, recipe = load_base_recipe(config)
    validate_recipe_geometry(recipe, geometry)
    dependencies = dependency_paths(config, recipe)
    missing = [str(value) for value in dependencies.values() if not value.is_file()]
    require(not missing, f"missing dependencies: {missing}")
    source_lineage = validate_source_lineage(recipe, dependencies, geometry)
    output = resolve_path(config["fit_output_directory"])
    require(not output.exists(), f"refusing to reuse fit output: {output}")

    frequency_path = dependencies["fit_frequency"]
    frequency = np.fromfile(frequency_path, dtype="<u8")
    high_rate_ids = select_high_rate_token_ids(frequency, geometry)
    high_rate_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device_name)
    high_rate_mask[torch.from_numpy(high_rate_ids.astype(np.int64)).to(device_name)] = True

    base = _load_module(BASE_RUNNER, "frequency_adaptive_r100_algorithm")
    broad = _load_module(BROAD_RUNNER, "frequency_adaptive_broad_inputs_multiratio")
    builder = _load_module(CORE_BUILDER, "frequency_adaptive_core_builder_multiratio")
    base.BASE_GROUPS = geometry.base_groups
    base.MAX_GROUPS = geometry.maximum_groups
    base.EXTRA_GROUPS = geometry.extra_groups
    base.LATENT_DIM = geometry.latent_dim
    base.HIGH_RATE_COUNT = geometry.high_rate_count
    base.LAYERS = geometry.n_layers
    base.D_PLE = geometry.d_ple

    original_kmeans = builder.kmeans_per_group
    builder.kmeans_per_group = lambda torch_module, latents, n_groups, group_dim, codebook_size, iterations, generator, chunk: base.frequency_adaptive_kmeans(
        torch_module,
        latents,
        n_groups,
        group_dim,
        codebook_size,
        iterations,
        generator,
        chunk,
        high_rate_mask=high_rate_mask,
        original_kmeans=original_kmeans,
        base_groups=geometry.base_groups,
    )
    builder.assignment_step = lambda *args: base.frequency_adaptive_assignment_step(
        *args,
        high_rate_mask=high_rate_mask,
        base_groups=geometry.base_groups,
        z_star_storage_device="cpu" if geometry.target_ratio == 20 else None,
    )
    builder.reinitialize_dead_codes = lambda *args: base.frequency_adaptive_reinitialize_dead_codes(
        *args, high_rate_mask=high_rate_mask, base_groups=geometry.base_groups
    )

    inputs = recipe["fit_inputs"]
    fit_recipe = recipe["fit_recipe"]
    model = recipe["model"]
    natural_path = dependencies["natural_token_file"]
    complete_path = dependencies["complete_tokenizer_rows"]

    def load_training_inputs(
        torch_module: Any,
        _safe_open: Any,
        gate_cache: str,
        gate_manifest: str,
        vocab: int,
        layers: int,
        d: int,
    ):
        require(resolve_path(gate_cache) == dependencies["gate_file"], "gate cache differs")
        require(
            resolve_path(gate_manifest) == dependencies["gate_manifest"],
            "gate manifest differs",
        )
        require(
            (vocab, layers, d)
            == (VOCAB_SIZE, geometry.n_layers, geometry.d_ple),
            "PLE shape differs",
        )
        natural = np.memmap(
            natural_path,
            dtype="<u4",
            mode="r",
            shape=(int(inputs["natural_token_file"]["positions"]),),
        )
        complete = np.memmap(complete_path, dtype="<u4", mode="r", shape=(VOCAB_SIZE,))
        repeated = np.repeat(
            np.asarray(SPECIAL_TOKEN_IDS, dtype=np.int64),
            int(fit_recipe["special_token_repetitions_each"]),
        )
        tokens = torch_module.from_numpy(
            np.concatenate(
                [
                    np.asarray(natural, dtype=np.int64),
                    np.asarray(complete, dtype=np.int64),
                    repeated,
                ]
            )
        ).long().contiguous()
        require(
            int(tokens.numel()) == int(fit_recipe["total_training_positions"]),
            "training position count differs",
        )
        gates = [
            broad.LazyLayerGate(
                torch_module,
                dependencies["gate_file"],
                layer,
                int(inputs["natural_token_file"]["positions"]),
                geometry.n_layers,
                geometry.d_ple,
                int(tokens.numel()),
                device_name,
            )
            for layer in range(geometry.n_layers)
        ]
        return tokens, gates, float(fit_recipe["runtime_token_scale"]), {
            "training_input_mode": fit_recipe["input_mode"],
            "fit_config": str(path.resolve()),
            "frequency_adaptive_variable_rate": True,
            "target_ratio": geometry.target_ratio,
            "base_group_count": geometry.base_groups,
            "maximum_group_count": geometry.maximum_groups,
            "high_rate_token_count": geometry.high_rate_count,
            "low_rate_extra_code_constraint": "zero",
            "source_model_identifier": model["identifier"],
            "source_model_revision": model["revision"],
            "source_checkpoint_ple_key": PLE_KEY,
        }

    builder.build_exact_gate_cache_forward_path_training_inputs = load_training_inputs
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    old_argv = sys.argv
    try:
        sys.argv = [
            str(CORE_BUILDER),
            "--it-weights", str(dependencies["checkpoint_weights"]),
            "--num-layers", str(geometry.n_layers),
            "--d-ple", str(geometry.d_ple),
            "--latent-mode", "discrete",
            "--latent-dim", str(geometry.latent_dim),
            "--n-groups", str(geometry.maximum_groups),
            "--codebook-size", str(CODEBOOK_SIZE),
            "--group-dim-allocation", "contiguous",
            "--alternations", str(fit_recipe["alternations"]),
            "--steps-per-alternation", str(fit_recipe["steps_per_alternation"]),
            "--assignment-icd-sweeps", str(fit_recipe["assignment_icd_sweeps"]),
            "--assignment-ridge", str(fit_recipe["assignment_ridge"]),
            "--assignment-chunk-size", str(fit_recipe["assignment_chunk_size"]),
            "--dead-code-min-usage", str(fit_recipe["dead_code_min_usage"]),
            "--kmeans-iterations", str(fit_recipe["kmeans_iterations"]),
            "--metric-eigenvalue-floor", str(fit_recipe["metric_eigenvalue_floor"]),
            "--lr", str(fit_recipe["learning_rate"]),
            "--batch-size", str(fit_recipe["batch_size"]),
            "--training-gate-cache", str(dependencies["gate_file"]),
            "--training-gate-cache-config", str(dependencies["gate_manifest"]),
            "--heldout-eval-size", str(fit_recipe["heldout_eval_size"]),
            "--heldout-eval-seed", str(fit_recipe["heldout_eval_seed"]),
            "--heldout-eval-interval-steps", str(fit_recipe["heldout_eval_interval_steps"]),
            "--resident-roundtrip-atol", str(fit_recipe["resident_roundtrip_atol"]),
            "--device", device_name,
            "--seed", str(fit_recipe["seed"]),
            "--save-recon", str(output / "dense_ple_reconstruction.safetensors"),
            "--save-compressed-ple-npz", str(output / "full_code_fit_artifact.npz"),
            "--output", str(output / "fit_metrics.json"),
        ]
        builder.main()
    finally:
        sys.argv = old_argv

    artifact = output / "full_code_fit_artifact.npz"
    metrics_path = output / "fit_metrics.json"
    require(artifact.is_file() and metrics_path.is_file(), "fit outputs are incomplete")
    with np.load(artifact, allow_pickle=False) as archive:
        required_shapes = {
            "token_codes_packed_uint8": (VOCAB_SIZE * geometry.maximum_groups,),
            "codebooks_fp16": (
                geometry.maximum_groups,
                CODEBOOK_SIZE,
                GROUP_DIM,
            ),
            "decoder_weight_fp16": (
                geometry.n_layers,
                geometry.d_ple,
                geometry.latent_dim,
            ),
            "decoder_bias_fp16": (geometry.n_layers, geometry.d_ple),
        }
        require(set(required_shapes).issubset(archive.files), "fit arrays are incomplete")
        resident = sum(int(archive[name].nbytes) for name in required_shapes)
        for name, shape in required_shapes.items():
            require(tuple(archive[name].shape) == shape, f"fit {name} shape differs")
        codes = archive["token_codes_packed_uint8"].reshape(
            VOCAB_SIZE, geometry.maximum_groups
        )
        high_lookup = np.zeros(VOCAB_SIZE, dtype=bool)
        high_lookup[high_rate_ids.astype(np.int64)] = True
        require(
            bool((codes[~high_lookup, geometry.base_groups :] == 0).all()),
            "low-rate extra codes are not zero",
        )
    require(resident == geometry.raw_fit_resident_bytes, "raw fit resident bytes differ")
    metrics = load_json(metrics_path)
    require(
        int(metrics["storage_bytes"]) == geometry.raw_fit_resident_bytes,
        "fit metric resident bytes differ",
    )
    receipt = {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-fit-receipt-v1",
        "status": "fit_complete",
        "build_config_path": str(path.resolve()),
        "rate_geometry": geometry.as_dict(),
        "artifact_path": str(artifact),
        "metrics_path": str(metrics_path),
        "dense_reconstruction_path": str(output / "dense_ple_reconstruction.safetensors"),
        "low_rate_extra_codes_all_zero": True,
        "raw_full_code_fit_resident_bytes": resident,
        "source_lineage": source_lineage,
        "quality_evaluation_run": False,
    }
    receipt_path = output / "fit_receipt.json"
    require(not receipt_path.exists(), "refusing to overwrite fit receipt")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract_parser = subparsers.add_parser("contract")
    contract_parser.add_argument("--build-config", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--build-config", type=Path, required=True)
    run_parser.add_argument("--device", required=True)
    args = parser.parse_args()
    result = (
        contract(args.build_config.resolve())
        if args.command == "contract"
        else run_fit(args.build_config.resolve(), args.device)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
