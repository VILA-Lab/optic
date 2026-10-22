from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from .config import (
    ROOT,
    load_benchmark,
    load_level,
    load_method,
    load_model,
    load_vlm,
    model_config_binding,
    object_sha256,
    resolve_model_config,
    write_json_atomic,
)
from .validation import validate


def _run_gpu_queue(
    gpu: int,
    jobs: Queue[tuple[str, str]],
    output_dir: Path,
    smoke: bool,
    model_config: Path,
    protocol_lock_binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows = []
    while True:
        try:
            method_name, benchmark_id = jobs.get_nowait()
        except Empty:
            break
        task_dir = output_dir / method_name / benchmark_id
        command = [
            sys.executable,
            "-m",
            "ple_eval.task_runner",
            "--benchmark",
            benchmark_id,
            "--method",
            method_name,
            "--gpu",
            str(gpu),
            "--port",
            str(8300 + gpu),
            "--output-dir",
            str(task_dir),
            "--model-config",
            str(model_config),
        ]
        if protocol_lock_binding is not None:
            command.extend(["--protocol-lock", protocol_lock_binding["path"],
                            "--protocol-lock-sha256", protocol_lock_binding["sha256"]])
        if smoke:
            command.append("--smoke")
        started = time.time()
        process = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
        rows.append({
            "method": method_name,
            "benchmark_id": benchmark_id,
            "gpu": gpu,
            "command": command,
            "returncode": process.returncode,
            "evaluation_returncode": process.returncode,
            "elapsed_seconds": time.time() - started,
            "output_dir": str(task_dir),
        })
        jobs.task_done()
    return {"gpu": gpu, "tasks": rows}


def run_level(
    *,
    level_name: str,
    method_names: list[str],
    gpus: list[int],
    requested_benchmarks: list[str],
    run_id: str | None,
    plan_only: bool,
    smoke: bool,
    deep_hash: bool,
    model_config: str | Path | None = None,
    protocol_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    model_config_path = resolve_model_config(model_config)
    validation = validate(
        method_names,
        deep_hash=deep_hash,
        suite_levels=[level_name],
        model_config=model_config_path,
        **({"protocol_lock_path": protocol_lock_path} if protocol_lock_path is not None else {}),
    )
    if not validation["ok"]:
        return {"ok": False, "stage": "preflight", "validation": validation}
    protocol_lock_binding = None
    if protocol_lock_path is not None:
        checked_lock = next(row for row in validation["findings"] if row["kind"] == "protocol_lock")
        protocol_lock_binding = {"path": checked_lock["lock_path"], "sha256": checked_lock["lock_sha256"]}

    level = load_level(level_name)
    allowed = set(level["benchmarks"])
    selected = requested_benchmarks or list(level["benchmarks"])
    unknown = sorted(set(selected) - allowed)
    if unknown:
        return {"ok": False, "stage": "planning", "error": f"benchmarks outside {level_name}: {unknown}"}

    approved = []
    skipped = []
    benchmark_records: dict[str, dict[str, Any]] = {}
    for benchmark_id in selected:
        record = load_benchmark(benchmark_id)
        benchmark_records[benchmark_id] = record
        if record["approved_for_paired_ple"] and record["status"] == "locked":
            approved.append(benchmark_id)
        else:
            skipped.append({
                "benchmark_id": benchmark_id,
                "status": record["status"],
                "reason": record.get("reason", "not approved"),
            })

    methods = {name: load_method(name) for name in method_names}
    model = load_model(model_config_path)
    vlm = load_vlm()
    effective_run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = ROOT / "results" / effective_run_id / level_name.lower()
    plan = {
        "schema_version": "ple-eval-plan-v1",
        "run_id": effective_run_id,
        "suite_level": level_name,
        "smoke": smoke,
        "model": model,
        "model_config": model_config_binding(model_config_path),
        "vlm": vlm,
        "methods": methods,
        "gpu_assignment": {
            "strategy": "dynamic_task_queue",
            "gpu_pool": gpus,
            "task_count": len(method_names) * len(approved),
        },
        "approved_benchmarks": approved,
        "skipped_benchmarks": skipped,
        "benchmark_records": benchmark_records,
    }
    if protocol_lock_binding is not None:
        plan["protocol_lock"] = protocol_lock_binding
    plan["plan_sha256"] = object_sha256(plan)
    if plan_only:
        return {"ok": True, "executed": False, "plan": plan}
    if not approved:
        return {"ok": False, "stage": "planning", "error": "no approved benchmarks", "plan": plan}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "plan.json", plan)
    for method_name in method_names:
        (output_dir / method_name).mkdir(parents=True, exist_ok=False)

    jobs: Queue[tuple[str, str]] = Queue()
    for benchmark_id in approved:
        for method_name in method_names:
            jobs.put((method_name, benchmark_id))

    worker_rows = []
    with ThreadPoolExecutor(max_workers=min(len(gpus), jobs.qsize())) as executor:
        futures = [
            executor.submit(
                _run_gpu_queue,
                gpu,
                jobs,
                output_dir,
                smoke,
                model_config_path,
                protocol_lock_binding,
            )
            for gpu in gpus
        ]
        for future in as_completed(futures):
            worker_rows.append(future.result())

    tasks_by_method: dict[str, list[dict[str, Any]]] = {name: [] for name in method_names}
    for worker in worker_rows:
        for task in worker["tasks"]:
            tasks_by_method[task["method"]].append(task)
    rows = []
    for method_name in method_names:
        tasks = tasks_by_method[method_name]
        tasks.sort(key=lambda row: approved.index(row["benchmark_id"]))
        used_gpus = sorted({int(row["gpu"]) for row in tasks})
        rows.append({
            "method": method_name,
            "gpu": used_gpus[0] if len(used_gpus) == 1 else None,
            "gpus": used_gpus,
            "ok": len(tasks) == len(approved) and all(row["returncode"] == 0 for row in tasks),
            "tasks": tasks,
        })
    result = {
        "ok": all(row["ok"] for row in rows),
        "executed": True,
        "output_dir": str(output_dir),
        "plan_sha256": plan["plan_sha256"],
        "methods": rows,
        "skipped_benchmarks": skipped,
    }
    write_json_atomic(output_dir / "run-summary.json", result)
    return result
