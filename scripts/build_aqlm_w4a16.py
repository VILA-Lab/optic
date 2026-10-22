#!/usr/bin/env python3
"""Build the W4A16 AQLM Euclidean beam-encoder PLE baselines.

The greedy-RVQ compact of the same checkpoint initializes the codes, two local
block-coordinate sweeps refine them, and one complete assignment pass of the
AQLM Euclidean beam encoder (author code) produces the final codes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "fit/aqlm_w4a16.json"
)
MODEL_KEYS = ("e2b", "e4b")
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
SAFEGUARD_TRACE_FIELDS = {
    "accepted_author_beam_rows",
    "rejected_worsening_rows",
    "maximum_rejected_proposal_sse_increase",
    "proposal_objective_sse_after",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected a JSON object: {source}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_file(path: str | Path, expected_sha256: str, role: str) -> Path:
    source = Path(path)
    require(source.is_file(), f"missing {role}: {source}")
    actual = sha256_file(source)
    require(actual == expected_sha256, f"{role} SHA-256 differs: {actual} != {expected_sha256}")
    return source


def packed_geometry(model: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    rows = int(model["vocab_size"]) * int(model["num_layers"])
    width = int(model["ple_dimension"])
    codebooks = int(recipe["codebooks"])
    clusters = int(recipe["clusters"])
    bits = int(recipe["bits"])
    code_bytes = codebooks * math.ceil(rows * bits / 8)
    codebook_bytes = codebooks * clusters * width * 2
    resident_bytes = code_bytes + codebook_bytes
    return {
        "rows": rows,
        "code_bytes": code_bytes,
        "codebook_bytes": codebook_bytes,
        "resident_bytes": resident_bytes,
        "compression_ratio": int(model["dense_ple_bytes"]) / resident_bytes,
    }


def output_paths(config: dict[str, Any], model_key: str) -> dict[str, Path]:
    model = config["models"][model_key]
    cell = f"{model_key}-r100-seed0"
    persistent = Path(config["artifacts"]["persistent_root"]) / cell
    ephemeral = Path(config["artifacts"]["ephemeral_root"]) / cell
    return {
        "root": persistent,
        "logs": persistent / "logs",
        "stages": persistent / "stages",
        "block_compact": persistent / "stages" / "two-sweep-block-coordinate-initialization.npz",
        "block_metrics": persistent / "stages" / "two-sweep-block-coordinate-metrics.json",
        "block_reconstruction": ephemeral / "two-sweep-block-coordinate-reconstruction.safetensors",
        "compact": persistent / "compressed_ple.npz",
        "summary": persistent / "summary.json",
        "raw_trace": persistent / "beam-encoder-trace.jsonl",
        "reconstruction": ephemeral / "dense_ple_reconstruction.safetensors",
        "build_receipt": persistent / "build-receipt.json",
        "method_config": Path(config["artifacts"]["method_config_root"]) / f"{model['method_id']}.json",
    }


def _validate_bound_model(declared: dict[str, Any], role: str, dense_bytes: int) -> None:
    config_path = verify_file(
        declared["model_config_path"], declared["model_config_sha256"], f"{role} model config"
    )
    bound = read_json(config_path)
    require(bound.get("model_id") == declared.get("model_id"), f"{role} model id differs")
    require(bound.get("revision") == declared.get("revision"), f"{role} revision differs")
    require(int(bound.get("ple_dense_bytes", -1)) == dense_bytes, f"{role} PLE byte count differs")
    require(Path(declared["weights_path"]).is_file(), f"missing {role} weights: {declared['weights_path']}")


def _validate_rvq_initializer(model: dict[str, Any], recipe: dict[str, Any], role: str) -> None:
    rvq = model["rvq_initializer"]
    method_path = verify_file(
        rvq["method_config_path"], rvq["method_config_sha256"], f"{role} RVQ method config"
    )
    method = read_json(method_path)
    require(method.get("method_id") == rvq["method_id"], f"{role} RVQ method id differs")
    require(method.get("compressed_artifact_path") == rvq["compact_path"], f"{role} RVQ path differs")
    require(method.get("compressed_artifact_sha256") == rvq["compact_sha256"], f"{role} RVQ hash binding differs")
    require(method.get("ratio_bucket") == "r100", f"{role} RVQ ratio differs")
    verify_file(rvq["compact_path"], rvq["compact_sha256"], f"{role} RVQ compact")
    metrics_path = verify_file(rvq["metrics_path"], rvq["metrics_sha256"], f"{role} RVQ metrics")
    metrics = read_json(metrics_path)
    require(metrics.get("method") == "residual_vq", f"{role} initializer is not greedy RVQ")
    require(int(metrics.get("codebooks", -1)) == int(recipe["codebooks"]), f"{role} RVQ codebooks differ")
    require(int(metrics.get("clusters", -1)) == int(recipe["clusters"]), f"{role} RVQ clusters differ")
    require(int(metrics.get("bits_per_code", -1)) == int(recipe["bits"]), f"{role} RVQ bits differ")
    expected_shape = [int(model["vocab_size"]), int(model["num_layers"]) * int(model["ple_dimension"])]
    require(metrics.get("ple_shape") == expected_shape, f"{role} RVQ PLE shape differs")
    geometry = packed_geometry(model, recipe)
    require(int(metrics.get("storage_bytes", -1)) == geometry["resident_bytes"], f"{role} RVQ bytes differ")


def validate_direct_builder_source(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    require("new = new_grouped[:, 0, :]" in text, "beam-encoder builder does not use the encoder output directly")
    require("retain_nonworsening" not in text, "beam-encoder builder must use the encoder output directly")
    require("torch.where" not in text, "beam-encoder builder must use the encoder output directly")
    require(text.count("new_grouped = beam_search(") == 1, "beam-encoder builder must make exactly one author beam call")
    return {
        "status": "passed",
        "author_beam_calls": 1,
        "proposal_acceptance": "direct_unfiltered_author_output",
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    require(
        config.get("schema_version")
        == "w4a16-aqlm-euclidean-beam-encoder-ple-r100-build-v1",
        "schema version differs",
    )
    require(config.get("status") == "fixed", "build contract is not fixed")
    require(config.get("classification") == "adapted_baseline", "classification differs")
    require(config.get("data_access_regime") == "model_weights_only", "data access differs")
    require(int(config.get("seed", -1)) == 0, "seed must be zero")
    recipe = config["recipe"]
    expected_recipe = {
        "target_ratio": 100,
        "codebooks": 5,
        "clusters": 256,
        "bits": 8,
        "full_vector_group_size": 256,
        "block_coordinate_sweeps": 2,
        "author_beam_passes": 1,
        "beam_size": 2,
        "per_vector_scales": False,
        "proposal_acceptance": "direct_unfiltered_author_output",
        "retain_nonworsening_safeguard": False,
        "calibration_corpus_used": False,
    }
    for key, expected in expected_recipe.items():
        require(recipe.get(key) == expected, f"recipe {key} differs")
    dependencies = config["dependencies"]
    verify_file(
        dependencies["wrapper"]["path"],
        dependencies["wrapper"]["sha256"],
        "beam-encoder wrapper",
    )
    refiner = verify_file(
        dependencies["block_coordinate_refiner"]["path"],
        dependencies["block_coordinate_refiner"]["sha256"],
        "two-sweep block-coordinate refiner",
    )
    direct_builder = verify_file(
        dependencies["beam_encoder_builder"]["path"],
        dependencies["beam_encoder_builder"]["sha256"],
        "beam-encoder builder",
    )
    source_audit = validate_direct_builder_source(direct_builder)
    author = config["author_code"]
    beam_path = verify_file(author["beam_search_path"], author["beam_search_sha256"], "audited author beam search")
    require(beam_path == Path(author["local_root"]) / "src" / "beam_search_l2.py", "author beam path escaped root")
    require(set(config.get("models", {})) == set(MODEL_KEYS), "model matrix must be exactly E2B and E4B")
    cells = []
    for model_key in MODEL_KEYS:
        model = config["models"][model_key]
        require(int(model["ple_dimension"]) == 256, f"{model_key} PLE dimension differs")
        require(
            int(model["dense_ple_bytes"])
            == int(model["vocab_size"]) * int(model["num_layers"]) * int(model["ple_dimension"]) * 2,
            f"{model_key} dense byte count differs",
        )
        _validate_bound_model(model["fit_source"], f"{model_key} fit source", int(model["dense_ple_bytes"]))
        _validate_bound_model(
            model["evaluation_target"], f"{model_key} evaluation target", int(model["dense_ple_bytes"])
        )
        _validate_rvq_initializer(model, recipe, model_key)
        geometry = packed_geometry(model, recipe)
        require(
            geometry["resident_bytes"] <= int(model["dense_ple_bytes"]) // 100,
            f"{model_key} exceeds the nominal 100x budget",
        )
        cells.append(
            {
                "model": model_key,
                "method_id": model["method_id"],
                "rvq_compact": model["rvq_initializer"]["compact_path"],
                "resident_bytes": geometry["resident_bytes"],
                "compression_ratio": geometry["compression_ratio"],
                "outputs": {name: str(path) for name, path in output_paths(config, model_key).items()},
            }
        )
    return {
        "status": "passed_not_started",
        "refiner": str(refiner),
        "direct_builder": str(direct_builder),
        "direct_builder_audit": source_audit,
        "cells": cells,
    }


def verify_matched_parent_child_ple(model: dict[str, Any], token_chunk: int = 1024) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    parent_path = Path(model["fit_source"]["weights_path"])
    child_path = Path(model["evaluation_target"]["weights_path"])
    expected_shape = [int(model["vocab_size"]), int(model["num_layers"]) * int(model["ple_dimension"])]
    digest = hashlib.sha256()
    with safe_open(parent_path, framework="pt", device="cpu") as parent, safe_open(
        child_path, framework="pt", device="cpu"
    ) as child:
        require(PLE_KEY in set(parent.keys()) and PLE_KEY in set(child.keys()), "parent/child PLE is missing")
        parent_slice = parent.get_slice(PLE_KEY)
        child_slice = child.get_slice(PLE_KEY)
        require(list(parent_slice.get_shape()) == expected_shape, "parent PLE shape differs")
        require(list(child_slice.get_shape()) == expected_shape, "child PLE shape differs")
        require(parent_slice.get_dtype() == "BF16" and child_slice.get_dtype() == "BF16", "PLE dtype differs")
        for start in range(0, expected_shape[0], token_chunk):
            stop = min(expected_shape[0], start + token_chunk)
            parent_chunk = parent_slice[start:stop]
            child_chunk = child_slice[start:stop]
            require(bool(torch.equal(parent_chunk, child_chunk)), f"parent/child PLE differs at {start}:{stop}")
            digest.update(parent_chunk.contiguous().view(torch.uint8).numpy().tobytes())
    actual = digest.hexdigest()
    require(actual == model["matched_parent_child_logical_ple_sha256"], "logical PLE hash differs")
    return {
        "status": "passed",
        "comparison": "bitwise_equal_complete_bfloat16_ple_tensor",
        "logical_ple_sha256": actual,
        "ple_shape": expected_shape,
        "parent_weights_path": str(parent_path),
        "child_weights_path": str(child_path),
    }


def build_commands(config: dict[str, Any], model_key: str, device: str) -> dict[str, list[str]]:
    require(model_key in MODEL_KEYS, f"unsupported model: {model_key}")
    model = config["models"][model_key]
    recipe = config["recipe"]
    paths = output_paths(config, model_key)
    python = str(config["runtime"]["python"])
    refiner = str(config["dependencies"]["block_coordinate_refiner"]["path"])
    direct_builder = str(config["dependencies"]["beam_encoder_builder"]["path"])
    common = ["--it-weights", str(model["fit_source"]["weights_path"]), "--num-layers", str(model["num_layers"]), "--d-ple", str(model["ple_dimension"])]
    return {
        "two_sweep_block_coordinate_initialization": [
            python,
            refiner,
            *common,
            "--source-artifact",
            str(model["rvq_initializer"]["compact_path"]),
            "--sweeps",
            "2",
            "--batch-size",
            str(recipe["block_coordinate_batch_size"]),
            "--device",
            device,
            "--seed",
            "0",
            "--save-recon",
            str(paths["block_reconstruction"]),
            "--save-compressed-ple-npz",
            str(paths["block_compact"]),
            "--output",
            str(paths["block_metrics"]),
        ],
        "beam_encoder": [
            python,
            direct_builder,
            *common,
            "--initialization-artifact",
            str(paths["block_compact"]),
            "--aqlm-root",
            str(config["author_code"]["local_root"]),
            "--beam-size",
            "2",
            "--row-chunk",
            str(recipe["beam_row_chunk"]),
            "--author-chunk-size-bytes",
            str(recipe["author_chunk_size_bytes"]),
            "--device",
            device,
            "--seed",
            "0",
            "--save-recon",
            str(paths["reconstruction"]),
            "--save-compressed-ple-npz",
            str(paths["compact"]),
            "--raw-output",
            str(paths["raw_trace"]),
            "--output",
            str(paths["summary"]),
        ],
    }


def validate_direct_trace(path: Path, expected_rows: int) -> dict[str, Any]:
    cursor = 0
    records = 0
    changed = 0
    objective_increased_chunks = 0
    maximum_objective_sse_increase = 0.0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        unexpected = SAFEGUARD_TRACE_FIELDS.intersection(record)
        require(not unexpected, f"safeguard fields in the beam-encoder trace at line {line_number}: {sorted(unexpected)}")
        require(int(record["row_start"]) == cursor, f"direct trace row gap at line {line_number}")
        cursor = int(record["row_stop"])
        require(cursor > int(record["row_start"]), f"empty direct trace row range at line {line_number}")
        require("objective_sse_before" in record and "objective_sse_after" in record, "objective values are missing")
        delta = float(record["objective_sse_after"]) - float(record["objective_sse_before"])
        if delta > max(1e-6, 1e-10 * float(record["objective_sse_before"])):
            objective_increased_chunks += 1
            maximum_objective_sse_increase = max(maximum_objective_sse_increase, delta)
        changed += int(record["changed_vectors"])
        records += 1
    require(cursor == expected_rows, f"direct trace coverage differs: {cursor} != {expected_rows}")
    return {
        "status": "passed",
        "records": records,
        "rows": cursor,
        "changed_vectors": changed,
        "objective_increased_chunks": objective_increased_chunks,
        "maximum_objective_sse_increase": maximum_objective_sse_increase,
        "proposal_acceptance": "direct_unfiltered_author_output",
    }


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    require(not log_path.exists(), f"refusing to overwrite log: {log_path}")
    with log_path.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    require(completed.returncode == 0, f"command failed with {completed.returncode}; see {log_path}")


def validate_outputs(config: dict[str, Any], model_key: str, pairing: dict[str, Any]) -> dict[str, Any]:
    model = config["models"][model_key]
    paths = output_paths(config, model_key)
    for name in ("block_compact", "block_metrics", "compact", "summary", "raw_trace", "reconstruction"):
        require(paths[name].is_file(), f"missing build output: {paths[name]}")
    block = read_json(paths["block_metrics"])
    require(int(block.get("sweeps", -1)) == 2, "block-coordinate initialization did not complete two sweeps")
    require(block.get("source_artifact_sha256") == model["rvq_initializer"]["compact_sha256"], "RVQ source binding differs")
    summary = read_json(paths["summary"])
    require(summary.get("classification") == "adapted_baseline", "final classification differs")
    require(summary.get("source_ple") == model["fit_source"]["weights_path"], "fit source path differs")
    require(summary.get("initialization_artifact") == str(paths["block_compact"]), "block initializer path differs")
    require(summary.get("initialization_artifact_sha256") == sha256_file(paths["block_compact"]), "block initializer hash differs")
    require(summary.get("aqlm_beam_search_sha256") == config["author_code"]["beam_search_sha256"], "author source differs")
    geometry = packed_geometry(model, config["recipe"])
    require(int(summary.get("resident_bytes", -1)) == geometry["resident_bytes"], "resident bytes differ")
    trace = validate_direct_trace(paths["raw_trace"], geometry["rows"])
    receipt = {
        "schema_version": "w4a16-aqlm-euclidean-beam-encoder-build-receipt-v1",
        "status": "built_pending_runtime_check",
        "completed_utc": utc_now(),
        "model": model_key,
        "method_id": model["method_id"],
        "fit_source": model["fit_source"],
        "evaluation_target": model["evaluation_target"],
        "matched_parent_child_ple_verification": pairing,
        "rvq_initializer": model["rvq_initializer"],
        "two_sweep_block_coordinate_initialization": {
            "compact_path": str(paths["block_compact"]),
            "compact_sha256": sha256_file(paths["block_compact"]),
            "metrics_path": str(paths["block_metrics"]),
            "metrics_sha256": sha256_file(paths["block_metrics"]),
            "sweeps": 2,
            "role": "fixed_local_initialization_only",
        },
        "author_beam_encoder": {
            **config["author_code"],
            "passes": 1,
            "proposal_acceptance": "direct_unfiltered_author_output",
            "retain_nonworsening_safeguard": False,
        },
        "direct_trace": {**trace, "path": str(paths["raw_trace"]), "sha256": sha256_file(paths["raw_trace"])},
        "compact_path": str(paths["compact"]),
        "compact_sha256": sha256_file(paths["compact"]),
        "resident_bytes": geometry["resident_bytes"],
        "compression_ratio": geometry["compression_ratio"],
        "summary_path": str(paths["summary"]),
        "summary_sha256": sha256_file(paths["summary"]),
        "reconstruction_path": str(paths["reconstruction"]),
        "reconstruction_sha256": sha256_file(paths["reconstruction"]),
    }
    write_json_atomic(paths["build_receipt"], receipt)
    method_config = {
        "schema_version": "ple-method-config-v1",
        "status": "built_pending_runtime_check",
        "method_id": model["method_id"],
        "label": f"AQLM Euclidean beam-encoder PLE adaptation ({geometry['compression_ratio']:.6f}x; {model_key.upper()})",
        "publication_name": config["publication_name"],
        "classification": "adapted_baseline",
        "publication_role": "adapted_baseline",
        "method_family": "aqlm_beam_encoder",
        "data_access_regime": "model_weights_only",
        "ratio_bucket": "r100",
        "seed": 0,
        "capability_mode": "dense_reconstruction_injection",
        "deployment_mode": "online_requested_token_row_decode",
        "fit_source_model_id": model["fit_source"]["model_id"],
        "fit_source_model_revision": model["fit_source"]["revision"],
        "source_qat_parent_model_config": model["fit_source"]["model_config_path"],
        "evaluation_model_id": model["evaluation_target"]["model_id"],
        "evaluation_model_revision": model["evaluation_target"]["revision"],
        "evaluation_w4a16_model_config": model["evaluation_target"]["model_config_path"],
        "dense_reference_method_key": model["evaluation_target"]["dense_reference_method_key"],
        "retention_reference_method_key": model["evaluation_target"]["dense_reference_method_key"],
        "compressed_artifact_path": receipt["compact_path"],
        "compressed_artifact_sha256": receipt["compact_sha256"],
        "compressed_artifact_file_bytes": paths["compact"].stat().st_size,
        "format_id": "joint_refined_additive_vq_npz",
        "ple_resident_bytes": receipt["resident_bytes"],
        "ple_compression_ratio": receipt["compression_ratio"],
        "compression_ratio_scope": "all persistent online PLE tensors; BF16 capability materialization excluded",
        "reconstruction_path": receipt["reconstruction_path"],
        "reconstruction_sha256": receipt["reconstruction_sha256"],
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "training_metrics_path": receipt["summary_path"],
        "training_metrics_sha256": receipt["summary_sha256"],
        "reported_reconstruction_nmse": float(summary["decoded_source_ple_nmse"]),
        "build_receipt_path": str(paths["build_receipt"]),
        "rvq_initialization_artifact_sha256": model["rvq_initializer"]["compact_sha256"],
        "proposal_acceptance": "direct_unfiltered_author_output",
        "retain_nonworsening_safeguard": False,
        "roundtrip_atol": 0.0,
    }
    write_json_atomic(paths["method_config"], method_config)
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from refresh_compact_requested_row_receipts import refresh_one

    runtime = refresh_one(paths["method_config"], write=True, overwrite=False)
    method_config = read_json(paths["method_config"])
    method_config["status"] = "ready_for_evaluation"
    write_json_atomic(paths["method_config"], method_config)
    receipt["status"] = "ready_for_evaluation"
    receipt["method_config_path"] = str(paths["method_config"])
    receipt["method_config_sha256"] = sha256_file(paths["method_config"])
    receipt["requested_row_receipts"] = runtime["after"]
    write_json_atomic(paths["build_receipt"], receipt)
    return receipt


def run_cell(config: dict[str, Any], model_key: str, device: str) -> dict[str, Any]:
    require(device.startswith("cuda:"), "full build requires an explicit CUDA device")
    validate_config(config)
    paths = output_paths(config, model_key)
    protected = [path for name, path in paths.items() if name not in {"root", "logs", "stages", "method_config"}]
    require(not any(path.exists() for path in protected), "partial or completed beam-encoder outputs already exist")
    paths["stages"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    paths["block_reconstruction"].parent.mkdir(parents=True, exist_ok=True)
    pairing = verify_matched_parent_child_ple(config["models"][model_key])
    commands = build_commands(config, model_key, device)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"})
    run_logged(commands["two_sweep_block_coordinate_initialization"], paths["logs"] / "01-two-sweep-block-coordinate.log", env)
    run_logged(commands["beam_encoder"], paths["logs"] / "02-beam-encoder.log", env)
    return validate_outputs(config, model_key, pairing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", choices=MODEL_KEYS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = read_json(args.config)
    contract = validate_config(config)
    if args.command == "plan":
        require(args.model is not None, "plan requires --model")
        result = {
            "contract": contract,
            "selected_model": args.model,
            "execution_order": [
                "two_sweep_block_coordinate_initialization",
                "beam_encoder",
            ],
            "commands": build_commands(config, args.model, args.device),
        }
    else:
        require(args.model is not None, "run requires --model")
        result = run_cell(config, args.model, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
