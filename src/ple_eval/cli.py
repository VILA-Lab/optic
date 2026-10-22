from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import run_level
from .evaluate import evaluate_artifact
from .validation import validate, validate_protocol_lock


DEFAULT_METHODS = "dense,optic_e4b_bf16_r100"
DEFAULT_GPUS = "0"


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="ple-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--methods", default=DEFAULT_METHODS)
    validate_parser.add_argument("--deep-hash", action="store_true")
    validate_parser.add_argument("--model-config", default="")
    validate_parser.add_argument("--protocol-lock", type=Path)

    protocol_parser = subparsers.add_parser("validate-protocols")
    protocol_parser.add_argument("--protocol-lock", type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--level", choices=["L1", "L2"], required=True)
    run_parser.add_argument("--methods", default=DEFAULT_METHODS)
    run_parser.add_argument("--gpus", default=DEFAULT_GPUS)
    run_parser.add_argument("--benchmarks", default="")
    run_parser.add_argument("--run-id", default="")
    run_parser.add_argument("--plan-only", action="store_true")
    run_parser.add_argument("--smoke", action="store_true")
    run_parser.add_argument("--deep-hash", action="store_true")
    run_parser.add_argument("--model-config", default="")
    run_parser.add_argument("--protocol-lock", type=Path)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the released BF16 E4B OPTIC artifact")
    evaluate_parser.add_argument("--model-dir", type=Path, help="Exact Hugging Face snapshot directory")
    evaluate_parser.add_argument("--artifact", type=Path, help="Seed-0 compressed_ple.npz")
    evaluate_parser.add_argument("--gpus", default=DEFAULT_GPUS)
    evaluate_parser.add_argument("--run-id")
    evaluate_parser.add_argument("--smoke", action="store_true", help="Run a small text and image check")
    evaluate_parser.add_argument("--prepare-only", action="store_true", help="Prepare inputs without running benchmarks")

    args = parser.parse_args()
    if args.command == "evaluate":
        try:
            result = evaluate_artifact(
                model_dir=args.model_dir, artifact=args.artifact,
                gpus=[int(value) for value in _csv(args.gpus)], run_id=args.run_id,
                smoke=args.smoke, prepare_only=args.prepare_only,
            )
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result["ok"] else 1)
    if args.command == "validate-protocols":
        result = validate_protocol_lock(args.protocol_lock)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result["ok"] else 1)
    if args.command == "validate":
        result = validate(
            _csv(args.methods),
            deep_hash=args.deep_hash,
            model_config=args.model_config or None,
            **({"protocol_lock_path": args.protocol_lock} if args.protocol_lock is not None else {}),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result["ok"] else 1)

    methods = _csv(args.methods)
    gpus = [int(value) for value in _csv(args.gpus)]
    benchmarks = _csv(args.benchmarks)
    if not methods or not gpus:
        parser.error("provide at least one method and one GPU")
    if len(gpus) != len(set(gpus)):
        parser.error("GPU ordinals must be unique")
    result = run_level(
        level_name=args.level,
        method_names=methods,
        gpus=gpus,
        requested_benchmarks=benchmarks,
        run_id=args.run_id or None,
        plan_only=args.plan_only,
        smoke=args.smoke,
        deep_hash=args.deep_hash,
        model_config=args.model_config or None,
        **({"protocol_lock_path": args.protocol_lock} if args.protocol_lock is not None else {}),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
