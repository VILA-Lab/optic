#!/usr/bin/env python3
"""Run matched CPU prefill/decode cases with the pinned native llama bench."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def cpu_list(values):
    values = sorted(values)
    spans = []
    for value in values:
        if spans and spans[-1][1] + 1 == value:
            spans[-1][1] = value
        else:
            spans.append([value, value])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def snapshot():
    threads = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            tasks = list((process / "task").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for task in tasks:
            try:
                raw = (task / "stat").read_text()
                fields = raw.rsplit(")", 1)[1].split()
                tid = int(task.name)
                threads[str(tid)] = {
                    "pid": int(process.name), "comm": raw.split("(", 1)[1].rsplit(")", 1)[0],
                    "ticks": int(fields[11]) + int(fields[12]), "last_cpu": int(fields[36]),
                    "allowed_cpus": cpu_list(os.sched_getaffinity(tid)),
                }
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    return {"utc": now(), "loadavg": Path("/proc/loadavg").read_text().strip(),
            "proc_stat": Path("/proc/stat").read_text(), "threads": threads}


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def summarize(report):
    result = {}
    for case in report["cases"]:
        rows = [r for pair in report["pairs"] if pair["case"]["id"] == case["id"] for r in pair["runs"]]
        if not rows:
            continue
        item = {}
        for variant in ("official", "optic"):
            runs = [r for r in rows if r["variant"] == variant]
            if not runs:
                continue
            rates = [r["process_median_tps"] for r in runs]
            item[variant] = {"processes": len(runs), "median_tps": statistics.median(rates),
                             "process_medians": rates, "min_process_tps": min(rates),
                             "max_process_tps": max(rates),
                             "peak_rss_bytes": max(r["peak_rss_bytes"] for r in runs)}
        paired = [p for p in report["pairs"] if p["case"]["id"] == case["id"] and p["complete"]]
        ratios = []
        for pair in paired:
            by_model = {r["variant"]: r for r in pair["runs"]}
            ratios.append(by_model["optic"]["process_median_tps"] / by_model["official"]["process_median_tps"])
        if ratios:
            item["paired_optic_over_official"] = ratios
            item["median_paired_ratio"] = statistics.median(ratios)
        result[case["id"]] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--optic", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=202610291645)
    parser.add_argument("--first-cpu", type=int, default=36)
    parser.add_argument("--monitor-cpu", type=int, default=0)
    parser.add_argument("--hold-file", type=Path,
                        help="finish the current pair and stop if this file exists")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if min(args.blocks, args.repetitions) < 1:
        parser.error("blocks and repetitions must be positive")
    models = {"official": args.official.resolve(), "optic": args.optic.resolve()}
    binary = args.binary.resolve()
    plan = json.loads(args.plan.read_text())
    cases = plan["cases"][:1] if args.pilot else plan["cases"]
    blocks, repetitions = (1, 1) if args.pilot else (args.blocks, args.repetitions)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_DYNAMIC="FALSE")
    path = args.output / "results.json"
    os.sched_setaffinity(0, {args.first_cpu})
    identities = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
                  for p in [binary, *models.values(), args.plan.resolve(), Path(__file__).resolve()]]
    if args.resume:
        report = json.loads(path.read_text())
        if report["cases"] != cases or report["blocks"] != blocks or report["repetitions"] != repetitions or report["seed"] != args.seed:
            parser.error("resume settings differ from retained run")
        if report["input_identities"] != identities or report["first_cpu"] != args.first_cpu or report["monitor_cpu"] != args.monitor_cpu:
            parser.error("resume input identities or CPU placement differ from retained run")
        report.update(status="running", resumed_utc=now())
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        report = {"started_utc": now(), "status": "running", "pilot": args.pilot,
                  "cases": cases, "blocks": blocks, "repetitions": repetitions,
                  "first_cpu": args.first_cpu, "monitor_cpu": args.monitor_cpu,
                  "seed": args.seed, "input_identities": identities,
                  "conditions": plan, "pairs": [], "summary": {}}
    os.sched_setaffinity(0, {args.monitor_cpu})
    report["controller_affinity"] = cpu_list(os.sched_getaffinity(0))
    rng = random.Random(args.seed)
    model_orders = {}
    for case in cases:
        orders = [("official", "optic") if b % 2 == 0 else ("optic", "official") for b in range(blocks)]
        rng.shuffle(orders)
        model_orders[case["id"]] = orders
    schedule = []
    for block in range(blocks):
        shuffled = list(cases)
        rng.shuffle(shuffled)
        schedule.extend((block, case) for case in shuffled)
    for index, (block, case) in enumerate(schedule):
        if index < len(report["pairs"]) and report["pairs"][index]["complete"]:
            continue
        if args.hold_file and args.hold_file.exists():
            report.update(status="held_at_completed_pair", held_utc=now())
            atomic_json(path, report)
            print(json.dumps({"status": report["status"], "completed_pairs": index}), flush=True)
            return
        order = model_orders[case["id"]][block]
        if index == len(report["pairs"]):
            report["pairs"].append({"index": index, "block": block, "case": case, "order": order, "runs": [], "complete": False})
        pair = report["pairs"][index]
        for variant in order[len(pair["runs"]):]:
            cpus = list(range(args.first_cpu, args.first_cpu + case["threads"]))
            mask = format(sum(1 << c for c in cpus), "x")
            command = ["taskset", "-c", cpu_list(cpus), str(binary), "bench", "-m", str(models[variant]),
                       "-ngl", "0", "-nopo", "1", "-nkvo", "1", "-lm", "mmap", "-lzm", "on",
                       "-p", str(case["prompt"]), "-n", str(case["generation"]), "-d", str(case["depth"]),
                       "-t", str(case["threads"]), "-C", mask, "--cpu-strict", "1", "--poll", "50",
                       "-b", "512", "-ub", "512", "-ctk", "f16", "-ctv", "f16", "-fa", "on",
                       "-r", str(repetitions), "-o", "json"]
            prefix = args.output / f"{index:02d}-{case['id']}-{variant}"
            atomic_json(prefix.with_suffix(".before.json"), snapshot())
            started = time.monotonic()
            owned = []
            with prefix.with_suffix(".stdout.json").open("wb") as stdout, prefix.with_suffix(".stderr.log").open("wb") as stderr:
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
                while True:
                    finished, status, usage = os.wait4(child.pid, os.WNOHANG)
                    if finished:
                        child.returncode = os.waitstatus_to_exitcode(status)
                        break
                    try:
                        owned.append({"seconds": time.monotonic() - started,
                                      "main_affinity": cpu_list(os.sched_getaffinity(child.pid)),
                                      "task_affinities": sorted({cpu_list(os.sched_getaffinity(int(t.name))) for t in Path(f"/proc/{child.pid}/task").iterdir()})})
                        maps = Path(f"/proc/{child.pid}/numa_maps").read_text()
                        prefix.with_suffix(".numa_maps").write_text(maps)
                        if ".gguf" in maps:
                            prefix.with_suffix(".model-numa_maps").write_text(maps)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    time.sleep(0.5)
            wall = time.monotonic() - started
            atomic_json(prefix.with_suffix(".after.json"), snapshot())
            if child.returncode:
                report.update(status="failed", failed_command=command, failed_prefix=str(prefix))
                atomic_json(path, report)
                raise RuntimeError(f"native bench failed: {prefix}")
            native = json.loads(prefix.with_suffix(".stdout.json").read_text())
            if len(native) != 1:
                raise RuntimeError(f"expected one native case: {prefix}")
            row = native[0]
            for key, expected in [("n_prompt", case["prompt"]), ("n_gen", case["generation"]), ("n_depth", case["depth"]), ("n_threads", case["threads"])]:
                if row[key] != expected:
                    raise RuntimeError(f"native configuration differs: {key}")
            samples = row["samples_ts"]
            if len(samples) != repetitions or any(s <= 0 for s in samples):
                raise RuntimeError(f"invalid native sample count/rate: {prefix}")
            entry = {"variant": variant, "pid": child.pid, "command": command,
                     "started_utc": row.get("test_time"), "completed_utc": now(),
                     "process_wall_seconds": wall, "peak_rss_bytes": usage.ru_maxrss * 1024,
                     "process_median_tps": statistics.median(samples), "native": row,
                     "major_faults": usage.ru_majflt, "minor_faults": usage.ru_minflt,
                     "observed_affinities": owned, "prefix": str(prefix)}
            pair["runs"].append(entry)
            report["summary"] = summarize(report)
            atomic_json(path, report)
            print(json.dumps({"pair": index, "case": case["id"], "model": variant,
                              "median_tps": entry["process_median_tps"], "utc": now()}), flush=True)
        pair.update(complete=True, completed_utc=now())
        report["summary"] = summarize(report)
        atomic_json(path, report)
        print(json.dumps({"completed_pair": index, "case": case["id"], "utc": now()}), flush=True)
    report.update(status="complete", completed_utc=now())
    report["summary"] = summarize(report)
    atomic_json(path, report)


if __name__ == "__main__":
    main()
