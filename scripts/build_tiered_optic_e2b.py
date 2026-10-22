#!/usr/bin/env python3
"""Fit and export frequency-tiered OPTIC for BF16 E2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import os
import tempfile

# Directory for large intermediate files.
SCRATCH_DIR = os.path.realpath(os.environ.get("PLE_SCRATCH_DIR") or tempfile.gettempdir())

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
GATE_CONFIG = ROOT / "configs/calibration/gates_e2b_bf16.json"
GATE_OUTPUT = Path(SCRATCH_DIR, "ple-e2b-original-bf16-broad-v2-gate-cache-v1")
BUILD_CONFIG = ROOT / "configs/fit/tiered_optic_e2b_bf16.json"
FIT_OUTPUT = Path(SCRATCH_DIR, "ple-e2b-original-bf16-frequency-tiered-optic-r100-fit-v1/seed0")
COMPACT_OUTPUT = Path(SCRATCH_DIR, "ple-e2b-original-bf16-frequency-tiered-optic-r100-finalized-v1/seed0")
CAPABILITY_OUTPUT = Path(SCRATCH_DIR, "ple-e2b-original-bf16-frequency-tiered-optic-r100-capability-v1")
SOURCE_SNAPSHOT = Path(json.loads((ROOT / "configs/models/e2b_bf16.json").read_text())["snapshot"])
SOURCE_WEIGHTS = SOURCE_SNAPSHOT / "model.safetensors"
DESTINATION = ROOT / "artifacts/e2b-original-bf16-frequency-tiered-optic-r100/seed0"
METHOD_CONFIG = ROOT / "configs/methods/frequency_tiered_optic_e2b_bf16_r100.json"
METHOD_ID = "frequency_tiered_optic_e2b_bf16_r100"
SOURCE_REVISION = "b446025c61ecea876162774ee247706056963aba"
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
PLE_SHAPE = (262_144, 35 * 256)
DENSE_BYTES = 4_697_620_480
RESIDENT_BYTES = 46_754_082
ROUNDTRIP_ATOL = 0.002
REQUESTED_TOKEN_IDS = (0, 1, 2, 3, 4, 46, 98, 101, 1024, 131_072, 258_884)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def run_command(argv: list[str]) -> None:
    environment = {
        **os.environ,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(ROOT / "src"),
        "TRANSFORMERS_OFFLINE": "1",
    }
    subprocess.run(argv, cwd=ROOT, env=environment, check=True)


def validate_static_inputs() -> dict[str, Any]:
    gate = load_json(GATE_CONFIG)
    build = load_json(BUILD_CONFIG)
    recipe = load_json(ROOT / str(build["base_fit_recipe_path"]))
    model = recipe["model"]
    require(PYTHON.is_file() and SOURCE_WEIGHTS.is_file(), "Python runtime or source checkpoint is missing")
    require(
        gate["model"]["identifier"] == "google/gemma-4-E2B-it"
        and gate["model"]["revision"] == SOURCE_REVISION
        and model["identifier"] == "google/gemma-4-E2B-it"
        and model["revision"] == SOURCE_REVISION
        and Path(model["weights"]) == SOURCE_WEIGHTS,
        "original-BF16 E2B source lineage differs",
    )
    require(
        int(build["target_ratio"]) == 100
        and int(build["rate_geometry"]["num_layers"]) == 35
        and int(build["rate_geometry"]["d_ple"]) == 256
        and int(build["rate_geometry"]["expected_resident_bytes"]) == RESIDENT_BYTES,
        "100x E2B geometry differs",
    )
    inputs = recipe["fit_inputs"]
    for entry in ("natural_token_file", "complete_tokenizer_rows", "fit_only_frequency"):
        require(Path(inputs[entry]["path"]).is_file(), f"missing fit input: {entry}")
    require(inputs["benchmark_inputs_used"] is False and inputs["l1_or_l2_inputs_used"] is False,
            "fit recipe uses benchmark inputs")
    require(int(recipe["fit_recipe"]["seed"]) == 0, "fit seed differs")
    return {"gate": gate, "build": build, "recipe": recipe}


def validate_gate() -> None:
    manifest = load_json(GATE_OUTPUT / "dout_gate_cache_config.json")
    require(
        manifest.get("complete") is True
        and manifest.get("model_identifier") == "google/gemma-4-E2B-it"
        and manifest.get("model_revision") == SOURCE_REVISION
        and manifest.get("gate_shape") == [35, 786_432, 256]
        and manifest.get("quality_evaluation_allowed") is False,
        "original-BF16 E2B gate cache identity differs",
    )
    gate_file = Path(manifest["gate_file"])
    require(gate_file.is_file() and gate_file.stat().st_size == 35 * 786_432 * 256 * 2,
            "gate cache byte count differs")


def run_build_stages() -> None:
    validate_static_inputs()
    if not GATE_OUTPUT.exists():
        run_command([
            str(PYTHON), str(ROOT / "scripts/build_broad_gate_cache.py"),
            "--config", str(GATE_CONFIG), "--device", "cuda:0", "--batch-size", "4",
            "--output-dir", str(GATE_OUTPUT),
        ])
    validate_gate()
    if not (COMPACT_OUTPUT / "compressed_ple.npz").is_file():
        require(not FIT_OUTPUT.exists() and not COMPACT_OUTPUT.exists(),
                "partial fit/finalization output exists; refusing implicit cleanup")
        run_command([
            str(PYTHON), str(ROOT / "scripts/build_tiered_optic.py"),
            "run", "--build-config", str(BUILD_CONFIG), "--device", "cuda:0",
        ])
    if not (CAPABILITY_OUTPUT / "dense-ple-reconstruction.safetensors").is_file():
        require(not CAPABILITY_OUTPUT.exists(), "partial capability output exists; refusing implicit cleanup")
        run_command([
            str(PYTHON), str(ROOT / "scripts/materialize_tiered_optic.py"),
            "materialize", "--target-ratio", "100", "--build-config", str(BUILD_CONFIG),
            "--device", "cuda:0",
        ])


def persist_outputs() -> dict[str, Any]:
    import numpy as np
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(ROOT / "src"))
    from ple_eval.tiered_optic_adapter import load_compact_ple

    compact = COMPACT_OUTPUT / "compressed_ple.npz"
    finalization = COMPACT_OUTPUT / "finalization_receipt.json"
    capability = CAPABILITY_OUTPUT / "dense-ple-reconstruction.safetensors"
    require(compact.is_file() and finalization.is_file() and capability.is_file(), "build outputs are incomplete")
    final_receipt = load_json(finalization)
    require(
        final_receipt.get("source_model_identifier") == "google/gemma-4-E2B-it"
        and final_receipt.get("source_model_revision") == SOURCE_REVISION
        and int(final_receipt.get("resident_bytes", -1)) == RESIDENT_BYTES,
        "finalization source or resident-byte identity differs",
    )
    if not DESTINATION.exists():
        partial = DESTINATION.with_name(f"{DESTINATION.name}.partial-{os.getpid()}")
        require(not partial.exists(), f"refusing to reuse {partial}")
        partial.mkdir(parents=True)
        for source, name in (
            (compact, "compressed_ple.npz"),
            (finalization, "source_finalization_receipt.json"),
            (FIT_OUTPUT / "full_code_fit_artifact.npz", "full_code_fit_artifact.npz"),
            (FIT_OUTPUT / "fit_metrics.json", "fit_metrics.json"),
            (FIT_OUTPUT / "fit_receipt.json", "fit_receipt.json"),
            (GATE_OUTPUT / "dout_gate_cache_config.json", "gate_cache_manifest.json"),
        ):
            require(source.is_file(), f"missing persistence input: {source}")
            shutil.copy2(source, partial / name)
        artifact = partial / "compressed_ple.npz"
        module = load_compact_ple(artifact, device="cpu")
        require(tuple(module.ple_shape) == PLE_SHAPE, "compact PLE shape differs")
        require(int(module.persistent_resident_bytes()) == RESIDENT_BYTES, "compact resident bytes differ")
        ids = torch.tensor(REQUESTED_TOKEN_IDS, dtype=torch.long)
        started = time.perf_counter()
        with torch.inference_mode():
            rows = module(ids).cpu().contiguous()
        elapsed = time.perf_counter() - started
        with safe_open(capability, framework="pt", device="cpu") as handle:
            capability_rows = torch.cat([
                handle.get_slice("embed_tokens_per_layer.weight")[token_id:token_id + 1]
                for token_id in REQUESTED_TOKEN_IDS
            ])
        maximum = float((rows.float() - capability_rows.float()).abs().max().item())
        require(math.isfinite(maximum) and maximum <= ROUNDTRIP_ATOL,
                "compact rows differ from capability reconstruction")
        special_ids = tuple(int(value) for value in (0, 1, 2, 3, 4, 46, 98, 101))
        with torch.inference_mode():
            compact_special = module(torch.tensor(special_ids, dtype=torch.long)).cpu().contiguous()
        with safe_open(SOURCE_WEIGHTS, framework="pt", device="cpu") as handle:
            source = handle.get_slice(PLE_KEY)
            source_special = torch.cat([source[token_id:token_id + 1] for token_id in special_ids])
        require(torch.equal(compact_special.view(torch.uint16), source_special.view(torch.uint16)),
                "exact special rows differ from original-BF16 source")
        digest = sha256(artifact)
        final_artifact = DESTINATION / "compressed_ple.npz"
        runtime = {
            "schema_version": "ple-compact-requested-row-runtime-v1", "status": "passed",
            "artifact_path": str(final_artifact), "artifact_sha256": digest,
            "source_model_identifier": "google/gemma-4-E2B-it", "source_model_revision": SOURCE_REVISION,
            "device": "cpu", "requested_token_ids": list(REQUESTED_TOKEN_IDS),
            "requested_rows": len(REQUESTED_TOKEN_IDS), "output_shape": list(rows.shape),
            "finite": bool(torch.isfinite(rows.float()).all().item()), "elapsed_seconds": elapsed,
            "persistent_resident_bytes": RESIDENT_BYTES, "serialized_file_bytes": artifact.stat().st_size,
            "dense_ple_or_dense_cache_present": False, "full_table_materialized": False,
            "benchmark_inputs_used": False,
        }
        roundtrip = {
            "schema_version": "ple-compact-requested-row-roundtrip-v1", "status": "passed",
            "artifact_path": str(final_artifact), "artifact_sha256": digest,
            "source_model_identifier": "google/gemma-4-E2B-it", "source_model_revision": SOURCE_REVISION,
            "requested_token_ids": list(REQUESTED_TOKEN_IDS), "requested_rows": len(REQUESTED_TOKEN_IDS),
            "max_abs_diff_after_bfloat16_materialization": maximum, "atol": ROUNDTRIP_ATOL,
            "exact_special_rows_bitwise_equal_to_source_checkpoint": True,
            "persistent_resident_bytes": RESIDENT_BYTES, "full_table_materialized": False,
            "capability_reconstruction_path": str(capability),
            "benchmark_inputs_used": False,
        }
        write_json_new(partial / "requested_row_runtime_receipt.json", runtime)
        write_json_new(partial / "artifact_bound_roundtrip_receipt.json", roundtrip)
        os.replace(partial, DESTINATION)

    artifact = DESTINATION / "compressed_ple.npz"
    runtime_path = DESTINATION / "requested_row_runtime_receipt.json"
    roundtrip_path = DESTINATION / "artifact_bound_roundtrip_receipt.json"
    require(artifact.is_file() and runtime_path.is_file() and roundtrip_path.is_file(), "persistent outputs are incomplete")
    digest = sha256(artifact)
    require(load_json(runtime_path).get("artifact_sha256") == digest, "runtime receipt artifact differs")
    require(load_json(roundtrip_path).get("artifact_sha256") == digest, "round-trip receipt artifact differs")
    method = {
        "schema_version": "ple-method-config-v1", "status": "ready_for_evaluation",
        "method_id": METHOD_ID,
        "label": "Frequency-tiered cross-layer latent vector quantization for PLE (E2B original BF16; about 100x)",
        "publication_name": "Frequency-tiered cross-layer latent vector quantization for per-layer embeddings",
        "classification": "our_method", "publication_role": "our_method", "method_family": "cross_layer_latent_vq",
        "rate_allocation": "token_frequency_tiered",
        "data_access_regime": "fixed_decontaminated_broad_v2_calibration_and_checkpoint_tokenizer_rows",
        "ratio_bucket": "r100", "retention_reference_method_key": "dense_gemma4_e2b_it",
        "capability_mode": "dense_reconstruction_injection", "deployment_mode": "online_requested_token_row_decode",
        "reconstruction_path": str(capability), "compressed_artifact_path": str(artifact),
        "compressed_artifact_sha256": digest, "compressed_artifact_file_bytes": artifact.stat().st_size,
        "ple_resident_bytes": RESIDENT_BYTES, "ple_compression_ratio": DENSE_BYTES / RESIDENT_BYTES,
        "compression_ratio_scope": "all persistent online PLE tensors; BF16 capability materialization excluded",
        "source_model_config": str(ROOT / "configs/models/e2b_bf16.json"),
        "evaluation_model_config": str(ROOT / "configs/models/e2b_bf16.json"),
        "source_model_revision": SOURCE_REVISION,
        "build_config_path": str(BUILD_CONFIG), "builder_path": str(Path(__file__).resolve()),
        "training_metrics_path": str(DESTINATION / "fit_metrics.json"),
        "fit_receipt_path": str(DESTINATION / "fit_receipt.json"),
        "gate_cache_manifest_path": str(DESTINATION / "gate_cache_manifest.json"),
        "runtime_receipt_path": str(runtime_path), "roundtrip_receipt_path": str(roundtrip_path),
        "roundtrip_atol": ROUNDTRIP_ATOL, "seed": 0,
    }
    if METHOD_CONFIG.exists():
        existing = load_json(METHOD_CONFIG)
        require(existing.get("method_id") == METHOD_ID,
                "existing method config has another method identity")
    else:
        write_json_new(METHOD_CONFIG, method)
    return {
        "status": "ready_for_evaluation", "method_id": METHOD_ID,
        "artifact_path": str(artifact), "artifact_sha256": digest,
        "resident_bytes": RESIDENT_BYTES, "compression_ratio": DENSE_BYTES / RESIDENT_BYTES,
        "method_config": str(METHOD_CONFIG), "fit_metrics": str(DESTINATION / "fit_metrics.json"),
    }


def contract() -> dict[str, Any]:
    validate_static_inputs()
    return {
        "schema_version": "e2b-original-bf16-frequency-tiered-ple-vq-r100-build-contract-v1",
        "status": "ready_or_resumable",
        "source_model_identifier": "google/gemma-4-E2B-it", "source_model_revision": SOURCE_REVISION,
        "seed": 0, "benchmark_inputs_used": False, "resident_bytes": RESIDENT_BYTES,
        "stages": {
            "gate_cache": (GATE_OUTPUT / "dout_gate_cache_config.json").is_file(),
            "compact": (COMPACT_OUTPUT / "compressed_ple.npz").is_file(),
            "capability": (CAPABILITY_OUTPUT / "dense-ple-reconstruction.safetensors").is_file(),
            "persistent_artifact": (DESTINATION / "compressed_ple.npz").is_file(),
            "method_config": METHOD_CONFIG.is_file(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run"))
    args = parser.parse_args()
    if args.action == "contract":
        result = contract()
    else:
        run_build_stages()
        result = persist_outputs()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
