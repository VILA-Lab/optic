#!/usr/bin/env python3
"""Check available disk space and optional cache and output limits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs/storage.json"


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for current_root, _, files in os.walk(path):
        base = Path(current_root)
        for name in files:
            candidate = base / name
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                continue
    return total


def evaluate_storage_policy(
    policy: dict[str, Any], *, reserve_bytes: int = 0,
    reserve_scope: str = "outputs", phase: str = "unspecified"
) -> dict[str, Any]:
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be non-negative")
    if reserve_scope not in {"corpus", "outputs"}:
        raise ValueError("reserve_scope must be 'corpus' or 'outputs'")
    filesystem_path = Path(policy["filesystem_path"])
    filesystem = os.statvfs(filesystem_path)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    corpus_rows = [
        {"path": value, "bytes": directory_size_bytes(Path(value))}
        for value in policy["corpus_cache_paths"]
    ]
    output_rows = [
        {"path": value, "bytes": directory_size_bytes(Path(value))}
        for value in policy["new_output_paths"]
    ]
    corpus_bytes = sum(row["bytes"] for row in corpus_rows)
    output_bytes = sum(row["bytes"] for row in output_rows)
    minimum_available_bytes = int(policy.get("minimum_available_bytes", 0))
    maximum_corpus_bytes = policy.get("maximum_corpus_cache_bytes")
    maximum_output_bytes = policy.get("maximum_new_output_bytes")
    checks = {
        "available_after_reserve_meets_floor": (
            available_bytes - reserve_bytes >= minimum_available_bytes
        ),
        "corpus_cache_within_limit": (
            maximum_corpus_bytes is None
            or corpus_bytes + (reserve_bytes if reserve_scope == "corpus" else 0)
            <= int(maximum_corpus_bytes)
        ),
        "new_output_within_limit": (
            maximum_output_bytes is None
            or output_bytes + reserve_bytes <= int(maximum_output_bytes)
        ),
    }
    return {
        "schema_version": "ple-r100-storage-check-v1",
        "phase": phase,
        "reserve_bytes": reserve_bytes,
        "reserve_scope": reserve_scope,
        "filesystem_path": str(filesystem_path),
        "available_bytes": available_bytes,
        "available_after_reserve_bytes": available_bytes - reserve_bytes,
        "corpus_cache_bytes": corpus_bytes,
        "new_output_bytes": output_bytes,
        "limits": {
            "minimum_available_bytes": minimum_available_bytes,
            "maximum_corpus_cache_bytes": maximum_corpus_bytes,
            "maximum_new_output_bytes": maximum_output_bytes,
        },
        "corpus_cache_paths": corpus_rows,
        "new_output_paths": output_rows,
        "checks": checks,
        "ok": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--reserve-scope", choices=("corpus", "outputs"), default="outputs")
    parser.add_argument("--phase", default="unspecified")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    report = evaluate_storage_policy(
        policy,
        reserve_bytes=args.reserve_bytes,
        reserve_scope=args.reserve_scope,
        phase=args.phase,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
