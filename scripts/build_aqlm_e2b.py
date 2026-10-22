#!/usr/bin/env python3
"""Fit the AQLM Euclidean beam encoder for BF16 E2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import os
import tempfile

# Directory for large intermediate files.
SCRATCH_DIR = os.path.realpath(os.environ.get("PLE_SCRATCH_DIR") or tempfile.gettempdir())

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MODEL_CONFIG = json.loads((ROOT / "configs/models/e2b_bf16.json").read_text())
WEIGHTS = Path(MODEL_CONFIG["snapshot"]) / MODEL_CONFIG["weight_file"]
SOURCE_METHOD = ROOT / "configs/methods/original_bf16_e2b_rvq_r100_seed0.json"
SOURCE_COMPACT = ROOT / "artifacts/original-bf16-e2b-baseline-matrix/rvq/r100/seed0/compressed_ple.npz"
ARTIFACT_ROOT = ROOT / "artifacts/original-bf16-e2b-baseline-matrix/aqlm/r100/seed0"
INITIALIZATION_ROOT = ARTIFACT_ROOT / "two-sweep-initialization"
INITIALIZATION_COMPACT = INITIALIZATION_ROOT / "compressed_ple.npz"
INITIALIZATION_SUMMARY = INITIALIZATION_ROOT / "summary.json"
INITIALIZATION_RECON = Path(SCRATCH_DIR, "original-bf16-e2b-baseline-matrix/aqlm/r100/seed0/two-sweep-initialization.safetensors")
COMPACT = ARTIFACT_ROOT / "compressed_ple.npz"
SUMMARY = ARTIFACT_ROOT / "summary.json"
RAW = ARTIFACT_ROOT / "author_beam_rows.jsonl"
RECON = Path(SCRATCH_DIR, "original-bf16-e2b-baseline-matrix/aqlm/r100/seed0/dense_ple_reconstruction.safetensors")
METHOD_KEY = "original_bf16_e2b_aqlm_euclidean_beam_encoder_r100_seed0"
METHOD_CONFIG = ROOT / f"configs/methods/{METHOD_KEY}.json"
AQLM_ROOT = ROOT / ".external/aqlm-beam-encoder-e79a896"
AQLM_COMMIT = "e79a896ed6656fe4ed06193d42d004e7d0bbdbb2"
AQLM_BEAM_SHA = "8dd39717dc9e44f7c8c8dba1426e616333878418dbe950ebeb7596c15848bdbf"
DENSE_BYTES = 4_697_620_480
RESIDENT_BYTES = 46_530_560


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_static() -> None:
    require(WEIGHTS.is_file(), "original BF16 E2B weights are absent")
    require(SOURCE_COMPACT.is_file(), "source-bound greedy RVQ initialization is absent")
    require(SOURCE_METHOD.is_file(), "source-bound greedy RVQ method config is absent")
    source_method = read_json(SOURCE_METHOD)
    require(source_method.get("method_id") == "original_bf16_e2b_rvq_r100_seed0", "source RVQ identity differs")
    require(source_method.get("compressed_artifact_sha256") == sha256(SOURCE_COMPACT), "source RVQ artifact binding differs")
    beam_source = AQLM_ROOT / "src/beam_search_l2.py"
    require(beam_source.is_file() and sha256(beam_source) == AQLM_BEAM_SHA, "audited AQLM beam source differs")


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"})
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    require(completed.returncode == 0, f"command failed with {completed.returncode}; see {log_path}")


def build_initialization(device: str) -> None:
    required = (INITIALIZATION_COMPACT, INITIALIZATION_SUMMARY, INITIALIZATION_RECON)
    if all(path.is_file() for path in required):
        summary = read_json(INITIALIZATION_SUMMARY)
        require(summary.get("sweeps") == 2, "AQLM initialization sweep count differs")
        require(summary.get("source_artifact_sha256") == sha256(SOURCE_COMPACT), "AQLM initialization source differs")
        return
    require(not any(path.exists() for path in required), "partial AQLM initialization exists")
    INITIALIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    INITIALIZATION_RECON.parent.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            str(PYTHON), str(ROOT / "scripts/refine_additive_quantization.py"),
            "--it-weights", str(WEIGHTS), "--source-artifact", str(SOURCE_COMPACT),
            "--num-layers", "35", "--d-ple", "256", "--sweeps", "2", "--batch-size", "8192",
            "--device", device, "--seed", "0", "--save-recon", str(INITIALIZATION_RECON),
            "--save-compressed-ple-npz", str(INITIALIZATION_COMPACT), "--output", str(INITIALIZATION_SUMMARY),
        ],
        INITIALIZATION_ROOT / "build.log",
    )


def build_aqlm(device: str) -> None:
    required = (COMPACT, SUMMARY, RAW, RECON)
    if all(path.is_file() for path in required):
        return
    require(not any(path.exists() for path in required), "partial AQLM adaptation outputs exist")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    RECON.parent.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            str(PYTHON), str(ROOT / "scripts/build_aqlm_e4b.py"),
            "--it-weights", str(WEIGHTS), "--initialization-artifact", str(INITIALIZATION_COMPACT),
            "--aqlm-root", str(AQLM_ROOT), "--num-layers", "35", "--d-ple", "256",
            "--beam-size", "2", "--row-chunk", "16384", "--author-chunk-size-bytes", "536870912",
            "--device", device, "--seed", "0", "--save-recon", str(RECON),
            "--save-compressed-ple-npz", str(COMPACT), "--raw-output", str(RAW), "--output", str(SUMMARY),
        ],
        ARTIFACT_ROOT / "build.log",
    )


def finalize() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from ple_eval.compact_adapter import artifact_sha256, compact_resident_bytes, load_compact_ple
    from refresh_compact_requested_row_receipts import refresh_one

    summary = read_json(SUMMARY)
    require(summary.get("classification") == "adapted_baseline", "AQLM classification differs")
    require(summary.get("aqlm_commit") == AQLM_COMMIT, "AQLM commit receipt differs")
    require(summary.get("aqlm_beam_search_sha256") == AQLM_BEAM_SHA, "AQLM source hash differs")
    require(summary.get("source_ple_sha256") == sha256(WEIGHTS), "AQLM source weights differ")
    require(summary.get("initialization_artifact_sha256") == sha256(INITIALIZATION_COMPACT), "AQLM initialization binding differs")
    require(float(summary["objective_nmse_after_fp32_decode"]) <= float(summary["objective_nmse_before"]), "AQLM objective increased")
    require(int(summary["resident_bytes"]) == RESIDENT_BYTES, "AQLM resident bytes differ")
    ratio = DENSE_BYTES / RESIDENT_BYTES
    require(math.isclose(float(summary["compression_ratio"]), ratio, rel_tol=0, abs_tol=1e-12), "AQLM ratio differs")
    module = load_compact_ple(COMPACT, device="cpu")
    require(int(compact_resident_bytes(module)) == RESIDENT_BYTES, "AQLM production adapter bytes differ")
    method = {
        "schema_version": "ple-method-config-v1",
        "status": "ready_for_evaluation",
        "method_id": METHOD_KEY,
        "label": f"AQLM Euclidean beam-encoder PLE adaptation ({ratio:.3f}x)",
        "publication_name": "AQLM Euclidean beam-encoder PLE adaptation",
        "classification": "adapted_baseline",
        "publication_role": "adapted_baseline",
        "method_family": "aqlm_euclidean_beam_encoder_ple_adaptation",
        "data_access_regime": "model_weights_only",
        "ratio_bucket": "r100",
        "capability_mode": "dense_reconstruction_injection",
        "deployment_mode": "online_requested_token_row_decode",
        "reconstruction_path": str(RECON),
        "reconstruction_sha256": sha256(RECON),
        "compressed_artifact_path": str(COMPACT),
        "compressed_artifact_sha256": artifact_sha256(COMPACT),
        "compressed_artifact_file_bytes": COMPACT.stat().st_size,
        "format_id": "joint_refined_additive_vq_npz",
        "builder_path": str(ROOT / "scripts/build_aqlm_e4b.py"),
        "builder_sha256": sha256(ROOT / "scripts/build_aqlm_e4b.py"),
        "training_metrics_path": str(SUMMARY),
        "training_metrics_sha256": sha256(SUMMARY),
        "reported_reconstruction_nmse": float(summary["decoded_source_ple_nmse"]),
        "ple_resident_bytes": RESIDENT_BYTES,
        "ple_compression_ratio": ratio,
        "compression_ratio_scope": "all persistent online PLE tensors; BF16 capability materialization excluded",
        "seed": 0,
        "source_model_config": str(ROOT / "configs/models/e2b_bf16.json"),
        "evaluation_model_config": str(ROOT / "configs/models/e2b_bf16.json"),
        "retention_reference_method_key": "dense_gemma4_e2b_it",
        "author_code": {
            "repository": "https://github.com/Vahe1994/AQLM",
            "commit": AQLM_COMMIT,
            "beam_search_l2_sha256": AQLM_BEAM_SHA,
        },
        "initialization_artifact": {
            "path": str(INITIALIZATION_COMPACT),
            "sha256": sha256(INITIALIZATION_COMPACT),
            "role": "fixed_two_sweep_additive_vq_initialization_only",
        },
    }
    write_json_atomic(METHOD_CONFIG, method)
    receipt = refresh_one(METHOD_CONFIG, write=True, overwrite=False)
    return {
        "status": "ready_for_evaluation",
        "method_key": METHOD_KEY,
        "resident_bytes": RESIDENT_BYTES,
        "compression_ratio": ratio,
        "method_config": str(METHOD_CONFIG),
        "receipt_refresh": receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "contract":
        require(WEIGHTS.is_file(), "original BF16 E2B weights are absent")
        require((AQLM_ROOT / "src/beam_search_l2.py").is_file(), "AQLM source is absent")
        result = {
            "status": "passed_waiting_for_source_rvq" if not SOURCE_COMPACT.exists() else "passed",
            "method_key": METHOD_KEY,
            "source_rvq_ready": SOURCE_COMPACT.exists(),
            "resident_bytes": RESIDENT_BYTES,
            "compression_ratio": DENSE_BYTES / RESIDENT_BYTES,
            "seed": 0,
        }
    else:
        validate_static()
        build_initialization(args.device)
        build_aqlm(args.device)
        result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
