#!/usr/bin/env python3
"""Decode a frequency-tiered OPTIC artifact for evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import os
import tempfile

# Directory for large intermediate files.
SCRATCH_DIR = os.path.realpath(os.environ.get("PLE_SCRATCH_DIR") or tempfile.gettempdir())

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_eval.tiered_optic_geometry import (  # noqa: E402
    RateGeometry,
    VOCAB_SIZE,
    geometry_from_mapping,
)
from ple_eval.tiered_optic_adapter import (  # noqa: E402
    load_compact_ple,
)


DENSE_RECONSTRUCTION_FILE_OVERHEAD_BYTES = 112
CHUNK_ROWS = 4096
RATIO_CONFIGS = {
    100: ROOT / "configs/fit/tiered_optic_e4b_w4a16.json",
}


class MaterializationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_build_config(
    target_ratio: int, build_config: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    if build_config is None:
        require(
            target_ratio in RATIO_CONFIGS,
            "a build config is required for the selected main-table setting",
        )
        config_path = RATIO_CONFIGS[target_ratio]
    else:
        config_path = build_config.resolve()
    config = load_json(config_path)
    require(
        config.get("schema_version")
        == "frequency-adaptive-variable-rate-multiratio-build-v1"
        and int(config.get("target_ratio", -1)) == target_ratio,
        "build config identity differs",
    )
    return config_path, config


def paths_for(
    target_ratio: int, build_config: Path | None = None
) -> dict[str, Path]:
    config_path, config = load_build_config(target_ratio, build_config)
    compact_root = resolve_path(config["compact_output_root"])
    output_root = resolve_path(
        config.get(
            "capability_output_root",
            SCRATCH_DIR + "/frequency-adaptive-variable-rate-cross-layer-latent-vector-"
            f"quantization-r{target_ratio}-capability",
        )
    )
    return {
        "config": config_path,
        "compact_root": compact_root,
        "artifact": compact_root / "compressed_ple.npz",
        "finalization_receipt": compact_root / "finalization_receipt.json",
        "output_root": output_root,
        "reconstruction": output_root / "dense-ple-reconstruction.safetensors",
        "receipt": output_root / "materialization-receipt.json",
    }


def geometry_for(target_ratio: int, build_config: Path | None = None) -> RateGeometry:
    _, config = load_build_config(target_ratio, build_config)
    return geometry_from_mapping(config["rate_geometry"])


def reconstruction_file_bytes(geometry: RateGeometry) -> int:
    return geometry.dense_bytes + DENSE_RECONSTRUCTION_FILE_OVERHEAD_BYTES


def validate_compact(
    target_ratio: int, build_config: Path | None = None
) -> tuple[dict[str, Path], RateGeometry]:
    paths = paths_for(target_ratio, build_config)
    geometry = geometry_for(target_ratio, build_config)
    require(paths["artifact"].is_file(), f"compact artifact is missing: {paths['artifact']}")
    require(
        paths["finalization_receipt"].is_file(),
        f"finalization receipt is missing: {paths['finalization_receipt']}",
    )
    receipt = load_json(paths["finalization_receipt"])
    declared = receipt.get("rate_geometry", {})
    require(
        receipt.get("status") == "finalized_in_dev_shm"
        and int(declared.get("target_ratio", -1)) == target_ratio
        and int(declared.get("base_group_count", -1)) == geometry.base_groups
        and int(declared.get("extra_group_count", -1)) == geometry.extra_groups
        and int(receipt.get("resident_bytes", -1)) == geometry.resident_bytes
        and receipt.get("dense_ple_or_dense_cache_present") is False
        and receipt.get("quality_evaluation_run") is False,
        "compact finalization geometry or pre-evaluation boundary differs",
    )
    return paths, geometry


def validate_existing_materialization(
    paths: Mapping[str, Path], geometry: RateGeometry
) -> None:
    require(paths["reconstruction"].is_file(), "dense reconstruction is missing")
    require(
        paths["reconstruction"].stat().st_size
        == reconstruction_file_bytes(geometry),
        "dense reconstruction byte count differs",
    )
    require(paths["receipt"].is_file(), "materialization receipt is missing")
    receipt = load_json(paths["receipt"])
    require(
        receipt.get("status") == "passed"
        and int(receipt.get("target_ratio", -1)) == geometry.target_ratio
        and int(receipt.get("persistent_resident_bytes", -1)) == geometry.resident_bytes
        and int(receipt.get("reconstruction_file_bytes", -1))
        == reconstruction_file_bytes(geometry)
        and receipt.get("capability_only") is True,
        "materialization receipt differs",
    )


def contract(target_ratio: int, build_config: Path | None = None) -> dict[str, Any]:
    paths = paths_for(target_ratio, build_config)
    geometry = geometry_for(target_ratio, build_config)
    compact_ready = paths["artifact"].is_file() and paths["finalization_receipt"].is_file()
    materialized = paths["output_root"].exists()
    if materialized:
        validate_existing_materialization(paths, geometry)
        status = "already_materialized"
    elif compact_ready:
        validate_compact(target_ratio, build_config)
        status = "ready_to_materialize"
    else:
        status = "waiting_for_compact_artifact"
    return {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-capability-contract-v1",
        "status": status,
        "target_ratio": target_ratio,
        "rate_geometry": geometry.as_dict(),
        "compact_artifact_path": str(paths["artifact"]),
        "finalization_receipt_path": str(paths["finalization_receipt"]),
        "reconstruction_path": str(paths["reconstruction"]),
        "materialization_receipt_path": str(paths["receipt"]),
        "expected_reconstruction_file_bytes": reconstruction_file_bytes(geometry),
        "capability_only": True,
        "quality_evaluation_run": False,
    }


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def materialize(
    target_ratio: int,
    device: str,
    chunk_rows: int,
    build_config: Path | None = None,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    require(device == "cuda:0", "Scheduler-bound materialization uses logical cuda:0")
    require(chunk_rows == CHUNK_ROWS, f"chunk rows must remain {CHUNK_ROWS}")
    paths, geometry = validate_compact(target_ratio, build_config)
    require(
        int(shutil.disk_usage(Path(SCRATCH_DIR)).free)
        >= reconstruction_file_bytes(geometry) + 1024**3,
        "shared memory cannot hold the reconstruction plus 1 GiB reserve",
    )
    require(not paths["output_root"].exists(), f"refusing to reuse {paths['output_root']}")
    staging = paths["output_root"].with_name(
        paths["output_root"].name + f".partial-{os.getpid()}"
    )
    require(not staging.exists(), f"refusing to reuse {staging}")
    staging.mkdir(parents=True)
    started = time.perf_counter()
    module = load_compact_ple(paths["artifact"], device=device)
    require(
        tuple(module.ple_shape)
        == (VOCAB_SIZE, geometry.n_layers * geometry.d_ple)
        and int(module.persistent_resident_bytes()) == geometry.resident_bytes
        and int(module.geometry.target_ratio) == target_ratio,
        "loaded compact artifact geometry differs",
    )
    reconstruction = torch.empty(
        (VOCAB_SIZE, geometry.n_layers * geometry.d_ple),
        dtype=torch.bfloat16,
        device="cpu",
    )
    with torch.inference_mode():
        for start in range(0, VOCAB_SIZE, chunk_rows):
            stop = min(VOCAB_SIZE, start + chunk_rows)
            ids = torch.arange(start, stop, device=device, dtype=torch.long)
            reconstruction[start:stop] = module(ids).to(device="cpu", dtype=torch.bfloat16)
    reconstruction_staged = staging / paths["reconstruction"].name
    save_file(
        {"embed_tokens_per_layer.weight": reconstruction.contiguous()},
        str(reconstruction_staged),
    )
    require(
        reconstruction_staged.stat().st_size == reconstruction_file_bytes(geometry),
        "dense reconstruction byte count differs",
    )
    receipt = {
        "schema_version": "frequency-adaptive-variable-rate-multiratio-capability-materialization-v1",
        "status": "passed",
        "target_ratio": target_ratio,
        "rate_geometry": geometry.as_dict(),
        "artifact_path": str(paths["artifact"]),
        "finalization_receipt_path": str(paths["finalization_receipt"]),
        "reconstruction_path": str(paths["reconstruction"]),
        "reconstruction_file_bytes": int(reconstruction_staged.stat().st_size),
        "ple_shape": [VOCAB_SIZE, geometry.n_layers * geometry.d_ple],
        "dtype": "torch.bfloat16",
        "persistent_resident_bytes": geometry.resident_bytes,
        "chunk_rows": chunk_rows,
        "device": device,
        "source_model_identifier": module.artifact_metadata.get(
            "source_model_identifier"
        ),
        "source_model_revision": module.artifact_metadata.get("source_model_revision"),
        "capability_only": True,
        "benchmark_inputs_used": False,
        "quality_evaluation_run": False,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_new(staging / paths["receipt"].name, receipt)
    os.replace(staging, paths["output_root"])
    validate_existing_materialization(paths, geometry)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract_parser = subparsers.add_parser("contract")
    contract_parser.add_argument("--target-ratio", type=int, required=True)
    contract_parser.add_argument("--build-config", type=Path)
    run_parser = subparsers.add_parser("materialize")
    run_parser.add_argument("--target-ratio", type=int, required=True)
    run_parser.add_argument("--build-config", type=Path)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    args = parser.parse_args()
    result = (
        contract(args.target_ratio, args.build_config)
        if args.command == "contract"
        else materialize(
            args.target_ratio, args.device, args.chunk_rows, args.build_config
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
