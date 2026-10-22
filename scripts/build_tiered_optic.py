#!/usr/bin/env python3
"""Fit and export frequency-tiered OPTIC."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIT_WRAPPER = ROOT / "scripts/fit_tiered_optic.py"
FINALIZER = ROOT / "scripts/finalize_tiered_optic.py"


class BuildTaskError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildTaskError(message)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def contract(config_path: Path) -> dict[str, Any]:
    fit = load_module(FIT_WRAPPER, "multiratio_fit_build_task_contract")
    finalizer = load_module(FINALIZER, "multiratio_finalizer_build_task_contract")
    config, geometry = fit.load_build_config(config_path)
    return {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-build-task-contract-v1",
        "status": "passed_not_started",
        "target_ratio": geometry.target_ratio,
        "rate_geometry": geometry.as_dict(),
        "fit": fit.contract(config_path),
        "finalization": finalizer.contract(config_path),
        "fit_output_directory": str(fit.resolve_path(config["fit_output_directory"])),
        "compact_output_root": str(fit.resolve_path(config["compact_output_root"])),
        "execution_order": ["fit", "matching_finalization"],
        "quality_evaluation_run": False,
    }


def run(config_path: Path, device: str) -> dict[str, Any]:
    fit = load_module(FIT_WRAPPER, "multiratio_fit_build_task_run")
    finalizer = load_module(FINALIZER, "multiratio_finalizer_build_task_run")
    config, geometry = fit.load_build_config(config_path)
    fit_output = fit.resolve_path(config["fit_output_directory"])
    compact_output = fit.resolve_path(config["compact_output_root"])

    if not fit_output.exists():
        fit_result = fit.run_fit(config_path, device)
    else:
        receipt_path = fit_output / "fit_receipt.json"
        artifact_path = fit_output / "full_code_fit_artifact.npz"
        require(
            receipt_path.is_file() and artifact_path.is_file(),
            "existing fit output is incomplete; refusing implicit cleanup or retry",
        )
        receipt = load_json(receipt_path)
        declared = receipt.get("rate_geometry", {})
        require(
            receipt.get("status") == "fit_complete"
            and int(declared.get("target_ratio", -1)) == geometry.target_ratio
            and int(declared.get("base_group_count", -1)) == geometry.base_groups
            and int(declared.get("extra_group_count", -1)) == geometry.extra_groups,
            "existing fit receipt geometry differs",
        )
        fit_result = {
            "status": "existing_complete_fit_reused",
            "artifact_path": str(artifact_path),
        }

    require(
        (fit_output / "fit_receipt.json").is_file()
        and (fit_output / "full_code_fit_artifact.npz").is_file(),
        "fit did not produce its required outputs",
    )
    if not compact_output.exists():
        finalization_result = finalizer.finalize(config_path)
    else:
        receipt_path = compact_output / "finalization_receipt.json"
        artifact_path = compact_output / "compressed_ple.npz"
        require(
            receipt_path.is_file() and artifact_path.is_file(),
            "existing compact output is incomplete; refusing implicit cleanup or retry",
        )
        receipt = load_json(receipt_path)
        declared = receipt.get("rate_geometry", {})
        require(
            receipt.get("status") == "finalized_in_dev_shm"
            and int(declared.get("target_ratio", -1)) == geometry.target_ratio
            and int(receipt.get("resident_bytes", -1)) == geometry.resident_bytes,
            "existing finalization receipt geometry differs",
        )
        finalization_result = {
            "status": "existing_complete_finalization_reused",
            "artifact_path": str(artifact_path),
        }
    return {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-build-task-result-v1",
        "status": "fit_and_matching_finalization_complete",
        "target_ratio": geometry.target_ratio,
        "rate_geometry": geometry.as_dict(),
        "fit": fit_result,
        "finalization": finalization_result,
        "quality_evaluation_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract_parser = subparsers.add_parser("contract")
    contract_parser.add_argument("--build-config", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--build-config", type=Path, required=True)
    run_parser.add_argument("--device", required=True)
    args = parser.parse_args()
    config_path = args.build_config.resolve()
    result = (
        contract(config_path)
        if args.command == "contract"
        else run(config_path, args.device)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
