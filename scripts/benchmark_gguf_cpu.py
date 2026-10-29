#!/usr/bin/env python3
"""Measure official and OPTIC GGUFs with the same Linux CPU configuration.

Uses the OPTIC-patched llama.cpp CLI. Saves every process's native timings,
stdout/stderr, peak RSS, model hashes and NUMA snapshot in the output directory.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--optic", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path,
                        default=Path(__file__).resolve().parents[1] / "examples/prompts/gguf-cpu.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cpus", required=True, help="taskset CPU list on one NUMA node, e.g. 36-71")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--lazy-modes", choices=("on", "auto"), nargs="+", default=["on", "auto"])
    parser.add_argument("--hold-file", type=Path,
                        help="stop before the next block if this file exists")
    parser.add_argument("--seed", type=int, default=202610291435)
    args = parser.parse_args()
    if args.threads < 1 or args.blocks < 1 or args.predict < 1:
        parser.error("threads, blocks and predict must be positive")
    args.binary = args.binary.resolve()
    args.official = args.official.resolve()
    args.optic = args.optic.resolve()
    args.prompt_file = args.prompt_file.resolve()
    prompt = args.prompt_file.read_text().strip()
    if not prompt or "\n" in prompt:
        parser.error("the streaming probe currently requires a single-line user prompt")
    args.output.mkdir(parents=True, exist_ok=False)
    variants = {f"{model}_lazy_{lazy}": (path, lazy)
                for model, path in [("official", args.official), ("optic", args.optic)]
                for lazy in args.lazy_modes}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_DYNAMIC="FALSE")
    report = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "primary_comparison": "official_lazy_on vs optic_lazy_on",
        "secondary_comparison": "official_lazy_auto vs optic_lazy_auto" if "auto" in args.lazy_modes else None,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "script_sha256": digest(Path(__file__)),
        "input_identities": [{"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}
                             for p in [args.binary, args.official, args.optic, args.prompt_file]],
        "prompt": prompt,
        "cpu_environment": {k: v for k, v in environment.items()
                            if k.startswith(("OMP_", "GOMP_", "KMP_")) or k == "CUDA_VISIBLE_DEVICES"},
        "cache_condition": "Inputs checksummed before runs; one untimed CLI process per variant; no eviction or global cache change",
        "warmup": "CLI --warmup enabled for both; untimed process per variant at verbosity4 for loading receipts; measured processes at verbosity3",
        "runs": [],
    }
    rng = random.Random(args.seed)
    order = []
    warmup = list(variants)
    rng.shuffle(warmup)
    order.extend((-1, name) for name in warmup)
    for block in range(args.blocks):
        names = list(variants)
        rng.shuffle(names)
        order.extend((block, name) for name in names)
    report["order"] = order
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    previous_block = None
    for index, (block, variant) in enumerate(order):
        if block != previous_block and args.hold_file and args.hold_file.exists():
            report["status"] = "held_at_completed_block"
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            return
        previous_block = block
        model, lazy = variants[variant]
        prefix = args.output / f"{index:02d}-block{block}-{variant}"
        command = ["taskset", "-c", args.cpus, str(args.binary), "cli", "-m", str(model),
                   "-ngl", "0", "-t", str(args.threads), "-tb", str(args.threads),
                   "--load-mode", "auto", "--lazy-mode", lazy,
                   "-c", "512", "-b", "512", "-ub", "512", "-n", str(args.predict),
                   "--cache-type-k", "f16", "--cache-type-v", "f16", "--flash-attn", "on",
                   "--single-turn", "--simple-io", "--reasoning", "off", "--temp", "0", "--seed", "0",
                   "--warmup", "--no-display-prompt", "--log-verbosity", "4" if block < 0 else "3", "-p", prompt]
        first_visible = None
        response_offset = None
        read_events = []
        output = bytearray()
        numa_path = Path(str(prefix) + ".numa_maps")
        load_before = Path("/proc/loadavg").read_text().strip()
        started = time.monotonic()
        with Path(str(prefix) + ".stderr.log").open("wb") as err:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, env=environment)
            while True:
                chunk = os.read(child.stdout.fileno(), 4096)
                if not chunk:
                    break
                elapsed = time.monotonic() - started
                read_events.append({"seconds": elapsed, "offset": len(output), "bytes": len(chunk)})
                output.extend(chunk)
                if response_offset is None:
                    position = output.find(b"\n> ")
                    if position >= 0:
                        end = output.find(b"\n", position + 3)
                        if end >= 0:
                            response_offset = end + 1
                if first_visible is None and response_offset is not None and output[response_offset:].strip():
                    first_visible = elapsed
                    try:
                        numa_path.write_text(Path(f"/proc/{child.pid}/numa_maps").read_text())
                    except (FileNotFoundError, ProcessLookupError):
                        pass
            child.stdout.close()
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
        wall = time.monotonic() - started
        Path(str(prefix) + ".stdout.log").write_bytes(output)
        stderr = Path(str(prefix) + ".stderr.log").read_text()
        slot_timings = "\n".join(line for line in stderr.splitlines() if "slot " in line)
        prompt_metrics = re.findall(r"prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens", slot_timings)
        decode_metrics = re.findall(r"(?<!prompt )eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens", slot_timings)
        native_rates = re.findall(rb"\[ Prompt:\s*([\d.]+) t/s \| Generation:\s*([\d.]+) t/s \]", output)
        if child.returncode or not prompt_metrics or not decode_metrics or len(native_rates) != 1:
            raise RuntimeError(f"run failed or metrics missing: {prefix}")
        generated = int(decode_metrics[-1][1])
        if response_offset is None or first_visible is None:
            raise RuntimeError(f"response echo boundary or first visible byte missing: {prefix}")
        response = bytes(output[response_offset:]).split(b"\n[ Prompt:", 1)[0].strip()
        Path(str(prefix) + ".response.txt").write_bytes(response)
        entry = {
            "block": block, "measured": block >= 0, "variant": variant, "command": command,
            "pid": child.pid, "returncode": child.returncode, "wall_seconds": wall,
            "first_visible_response_seconds": first_visible,
            "first_response_measurement": "first non-whitespace stdout byte after the UI prompt echo line, which may be truncated; includes initialization and pipe buffering",
            "response_offset": response_offset, "stdout_read_events": read_events,
            "actual_output_tokens": generated, "output_count_matches_limit": generated == args.predict,
            "actual_decode_steps": max(0, generated - 1),
            "stop_reason": "CLI info log does not expose final stop enum; actual token count retained",
            "prompt_tokens": int(prompt_metrics[-1][1]),
            "native_prompt_ms": float(prompt_metrics[-1][0]),
            "native_decode_ms": float(decode_metrics[-1][0]),
            "generation_tps_from_native_ms": max(0, generated - 1) / (float(decode_metrics[-1][0]) / 1000),
            "decode_rate_convention": "native llama.cpp: first output token comes from prefill; decode steps = output tokens - 1",
            "native_generation_tps_rounded": float(native_rates[0][1]),
            "peak_rss_bytes": usage.ru_maxrss * 1024,
            "major_page_faults": usage.ru_majflt, "minor_page_faults": usage.ru_minflt,
            "filesystem_input_blocks": usage.ru_inblock,
            "response_sha256": hashlib.sha256(response).hexdigest(),
            "stdout": str(prefix) + ".stdout.log", "stderr": str(prefix) + ".stderr.log",
            "load_mode_log": re.findall(r"load_mode\s*=\s*(\w+)", stderr),
            "lazy_enabled_log_lines": [line for line in stderr.splitlines() if "lazy read enabled" in line],
            "numa_maps": str(numa_path) if numa_path.exists() else None,
            "loadavg_before": load_before, "loadavg_after": Path("/proc/loadavg").read_text().strip(),
        }
        report["runs"].append(entry)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: entry[k] for k in ["block", "variant", "wall_seconds", "actual_output_tokens"]}), flush=True)
        if generated != args.predict:
            raise RuntimeError("Actual generation length differs from fixed-work protocol; do not claim a matched speedup")
    report["summary"] = {}
    for variant in variants:
        runs = [r for r in report["runs"] if r["measured"] and r["variant"] == variant]
        report["summary"][variant] = {
            "wall_seconds_median": statistics.median(r["wall_seconds"] for r in runs),
            "first_visible_response_seconds_median": statistics.median(r["first_visible_response_seconds"] for r in runs),
            "generation_tps_median": statistics.median(r["generation_tps_from_native_ms"] for r in runs),
            "max_peak_rss_bytes": max(r["peak_rss_bytes"] for r in runs),
        }
    report["paired_blocks"] = []
    for lazy in args.lazy_modes:
        for block in range(args.blocks):
            rows = {r["variant"]: r for r in report["runs"] if r["block"] == block}
            official = rows[f"official_lazy_{lazy}"]
            optic = rows[f"optic_lazy_{lazy}"]
            report["paired_blocks"].append({
                "lazy": lazy, "block": block,
                "official_wall_seconds": official["wall_seconds"], "optic_wall_seconds": optic["wall_seconds"],
                "optic_over_official_wall": optic["wall_seconds"] / official["wall_seconds"],
            })
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["status"] = "complete"
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
