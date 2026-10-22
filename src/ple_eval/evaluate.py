"""Evaluate the released OPTIC artifact for BF16 Gemma 4 E4B on the seven benchmarks."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import ROOT, file_sha256, read_json, write_json_atomic
from .orchestrator import run_level

METHOD = "optic_e4b_bf16_r100"
BENCHMARKS = {
    "L1": ["l1_quality_bundle"],
    "L2": ["gpqa_diamond", "aime_2026", "ifbench", "infographic_vqa", "mmlu_redux_30x100"],
}
MANIFEST = ROOT / "configs/artifact.json"


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _checked_file(path: Path, expected_sha256: str) -> Path:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")
    return path


def resolve_artifact(path: Path | None = None, manifest: dict[str, Any] | None = None) -> Path:
    """Download or locate the seed-0 artifact and check its SHA-256."""
    manifest = read_json(MANIFEST) if manifest is None else manifest
    entry = manifest["artifact"]
    if path is None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=entry["repo_id"], filename=entry["filename"],
                                    revision=entry["revision"]))
    artifact = path.expanduser().absolute()
    if not artifact.is_file():
        raise FileNotFoundError(f"OPTIC artifact is missing: {artifact}")
    return _checked_file(artifact, entry["sha256"])


def resolve_model_snapshot(path: Path | None = None, model: dict[str, Any] | None = None) -> Path:
    """Download or locate the fixed E4B snapshot and check its model weights."""
    model = read_json(ROOT / "configs/models/e4b_bf16.json") if model is None else model
    if path is None:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=model["model_id"], revision=model["revision"]))
    snapshot = path.expanduser().absolute()
    _checked_file(snapshot / model["weight_file"], model["weight_sha256"])
    return snapshot


def prepare(*, run_dir: Path, model_dir: Path | None, artifact: Path | None, gpu: int) -> dict[str, Any]:
    """Resolve the fixed inputs and prepare an audited reconstruction."""
    manifest = read_json(MANIFEST)
    if manifest["method"] != METHOD:
        raise ValueError(f"the released artifact uses {METHOD}")
    method = read_json(ROOT / "configs/methods" / f"{METHOD}.json")
    model = read_json(ROOT / "configs/models/e4b_bf16.json")
    artifact_path = resolve_artifact(artifact, manifest)
    if manifest["artifact"]["sha256"] != method["compressed_artifact_sha256"]:
        raise ValueError("artifact manifest and method configuration differ")
    snapshot = resolve_model_snapshot(model_dir, model)
    vllm = Path(sys.executable).parent / "vllm"
    if not vllm.is_file():
        raise FileNotFoundError(f"vLLM is missing from the active environment: {vllm}")
    model["snapshot"] = str(snapshot)
    model["runtime"]["python"] = sys.executable
    model["runtime"]["vllm"] = str(vllm)
    prepared_dir = run_dir / "prepared"
    model_config = prepared_dir / "model.json"
    method_dir = prepared_dir / "methods"
    write_json_atomic(model_config, model)

    reconstruction = ROOT / "artifacts/optic/e4b-bf16-r100-seed0/dense_ple_reconstruction.safetensors"
    materialization_receipt = reconstruction.with_name("materialization.json")
    if not reconstruction.is_file():
        subprocess.run([
            sys.executable, "-m", "ple_eval.materialize", "--artifact", str(artifact_path),
            "--output", str(reconstruction), "--receipt", str(materialization_receipt),
            "--device", f"cuda:{gpu}", "--chunk-rows", str(method["audit_chunk_rows"]),
        ], cwd=ROOT, check=True)
    _checked_file(reconstruction, method["reconstruction_sha256"])
    method["compressed_artifact_path"] = str(artifact_path)
    method["reconstruction_path"] = str(reconstruction)
    method["roundtrip_receipt_path"] = str(prepared_dir / "roundtrip.json")
    method["source_model_revision"] = model["revision"]
    for name in ("builder", "roundtrip_verifier", "online_adapter"):
        source = (ROOT / method[f"{name}_path"]).resolve()
        method[f"{name}_path"] = str(source)
        method[f"{name}_sha256"] = file_sha256(source)
    method.pop("training_metrics_path", None)
    method.pop("training_metrics_sha256", None)
    write_json_atomic(method_dir / f"{METHOD}.json", method)
    environment = {
        "PLE_EVAL_METHOD_CONFIG_DIR": str(method_dir),
        "PLE_EVALSCOPE_DATASET_DIR": os.environ.get("PLE_EVALSCOPE_DATASET_DIR", str(ROOT / ".cache/evalscope")),
        "MODELSCOPE_CACHE": os.environ.get("MODELSCOPE_CACHE", str(ROOT / ".cache/modelscope")),
        "NLTK_DATA": os.environ.get("NLTK_DATA", str(ROOT / ".cache/nltk_data")),
    }
    with _environment(environment):
        subprocess.run([
            sys.executable, "-m", "ple_eval.artifact_audit", "--method", METHOD,
            "--model-config", str(model_config), "--device", f"cuda:{gpu}",
            "--output", method["roundtrip_receipt_path"],
        ], cwd=ROOT, check=True)
    vllm_source = Path(importlib.util.find_spec("vllm").origin).parent / "model_executor/models/gemma4.py"
    prepared = {
        "method": METHOD,
        "model_config": str(model_config),
        "method_config": str(method_dir / f"{METHOD}.json"),
        "artifact_sha256": method["compressed_artifact_sha256"],
        "model_revision": model["revision"],
        "runtime": {
            "python": sys.executable,
            "package_versions": {name: importlib.metadata.version(name)
                                 for name in ("torch", "vllm", "transformers", "evalscope")},
            "gemma4_source_path": str(vllm_source),
            "gemma4_source_sha256": file_sha256(vllm_source),
        },
        "environment": environment,
    }
    write_json_atomic(prepared_dir / "inputs.json", prepared)
    return prepared


def _paper_helpers() -> Any:
    path = ROOT / "scripts/compute_paper_tables.py"
    spec = importlib.util.spec_from_file_location("optic_paper_tables", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load scoring helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aggregate(run_dir: Path) -> dict[str, Any]:
    """Read the seven measured scores and compute retention against the uncompressed reference."""
    tasks: dict[str, Path] = {}
    for level, benchmarks in BENCHMARKS.items():
        level_dir = run_dir / level.lower()
        plan = read_json(level_dir / "plan.json")
        if plan["smoke"]:
            raise ValueError("smoke runs have no seven-task retention result")
        for benchmark in benchmarks:
            task_dir = level_dir / METHOD / benchmark
            if read_json(task_dir / "status.json")["status"] != "completed":
                raise ValueError(f"benchmark is incomplete: {benchmark}")
            tasks[benchmark] = task_dir
    helpers = _paper_helpers()
    scores = helpers.run_scores(tasks)
    table_path = ROOT / "docs/figures/main-table.json"
    rows = json.loads(table_path.read_text(encoding="utf-8"))
    row = next(row for row in rows if row["row"] == "OPTIC" and row["column"] == "E4B BF16")
    reference = {name: value / 100.0 for name, value in row["reference_scores_pct"].items()}
    method = read_json(tasks["l1_quality_bundle"] / "method.json")
    model = read_json(tasks["l1_quality_bundle"] / "selected-model-config.json")
    return {
        "method": METHOD,
        "scores_pct": {name: 100.0 * scores[name] for name in helpers.TASK_METRICS},
        "retention_pct": helpers.retention_pct(scores, reference),
        "compression_ratio": method["ple_compression_ratio"],
        "reference": {
            "label": row["reference"],
            "scores_pct": row["reference_scores_pct"],
            "source": str(table_path),
            "source_sha256": file_sha256(table_path),
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "protocols": {name: read_json(path / "benchmark.json")["protocol"]["protocol_id"]
                          for name, path in tasks.items()},
        },
        "task_directories": {name: str(path) for name, path in tasks.items()},
    }


def evaluate_artifact(*, model_dir: Path | None = None, artifact: Path | None = None,
                        gpus: list[int] | None = None, run_id: str | None = None,
                        smoke: bool = False, prepare_only: bool = False) -> dict[str, Any]:
    gpus = [0] if gpus is None else gpus
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError("provide unique nonnegative GPU ordinals")
    run_id = run_id or time.strftime("optic-e4b-%Y%m%dT%H%M%SZ", time.gmtime())
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run ID must be a directory name")
    run_dir = ROOT / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not prepare_only and any((run_dir / level.lower()).exists() for level in BENCHMARKS):
        raise FileExistsError(f"benchmark outputs already exist in {run_dir}; choose another --run-id")
    started = time.monotonic()
    prepared = prepare(run_dir=run_dir, model_dir=model_dir, artifact=artifact, gpu=gpus[0])
    result: dict[str, Any] = {
        "ok": True, "run_id": run_id, "output_dir": str(run_dir),
        "smoke": smoke, "prepared": prepared, "executed": False,
    }
    if not prepare_only:
        levels = []
        with _environment(prepared["environment"]):
            for level, benchmarks in BENCHMARKS.items():
                outcome = run_level(
                    level_name=level, method_names=[METHOD], gpus=gpus,
                    requested_benchmarks=benchmarks, run_id=run_id,
                    plan_only=False, smoke=smoke, deep_hash=True,
                    model_config=prepared["model_config"],
                    protocol_lock_path=ROOT / "configs/evaluation/protocol.json",
                )
                levels.append(outcome)
                if not outcome["ok"]:
                    break
        result.update(ok=all(level["ok"] for level in levels), executed=True, levels=levels)
        if result["ok"] and not smoke:
            result["result"] = aggregate(run_dir)
    result["elapsed_seconds"] = time.monotonic() - started
    output_name = "prepared.json" if prepare_only else "smoke.json" if smoke else "result.json"
    write_json_atomic(run_dir / output_name, result)
    return result
