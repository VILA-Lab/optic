#!/usr/bin/env python3
"""Build and validate the matched W4A16 PLE baseline matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os
import tempfile

# Directory for large intermediate files.
SCRATCH_DIR = os.path.realpath(os.environ.get("PLE_SCRATCH_DIR") or tempfile.gettempdir())

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "fit/baselines_w4a16.json"
PYTHON = Path("python")
RATIOS = (20, 40, 100)
MODELS = ("e2b", "e4b")
METHODS = ("pq", "rvq", "opq_parametric", "opq_nonparametric", "aqlm", "additive_quantization", "low_rank_svd")
L2_BENCHMARKS = ("gpqa_diamond", "aime_2026", "ifbench", "infographic_vqa", "mmlu_redux_30x100")
PROPOSED_W4_METHOD_KEYS = (
    "e2b_qat_w4a16_frequency_tiered_cross_layer_latent_vector_quantization_r20",
    "e2b_qat_w4a16_frequency_tiered_cross_layer_latent_vector_quantization_r40",
    "frequency_tiered_optic_e2b_w4a16_r100",
    "e4b_qat_w4a16_frequency_tiered_cross_layer_latent_vector_quantization_r20",
    "e4b_qat_w4a16_frequency_tiered_cross_layer_latent_vector_quantization_r40",
    "frequency_tiered_optic_e4b_w4a16_r100",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def bind_retention_reference(
    config: dict[str, Any], model: str, method_config_path: Path,
) -> dict[str, Any]:
    """Bind aggregation to the dense reference from the same W4A16 model revision."""
    method_config = read_json(method_config_path)
    expected = str(config["models"][model]["dense_reference_method_key"])
    actual = method_config.get("retention_reference_method_key")
    require(actual in (None, expected), f"retention reference differs: {actual} != {expected}")
    if actual is None:
        method_config["retention_reference_method_key"] = expected
        write_json_atomic(method_config_path, method_config)
    return method_config


def configured_models(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(config.get("selected_models", MODELS))


def configured_methods(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(config.get("selected_methods", METHODS))


def configured_ratios(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in config.get("selected_ratios", RATIOS))


def parse_cell(config: dict[str, Any], value: str) -> tuple[str, str, int]:
    parts = value.split(":")
    require(len(parts) == 3, "cell must be MODEL:METHOD:RATIO")
    model, method, ratio_text = parts
    require(model in configured_models(config), f"unsupported model: {model}")
    require(method in configured_methods(config), f"unsupported method: {method}")
    ratio = int(ratio_text)
    require(ratio in configured_ratios(config), f"unsupported ratio: {ratio}")
    return model, method, ratio


def cell_id(model: str, method: str, ratio: int) -> str:
    return f"{model}:{method}:{ratio}"


def method_key(model: str, method: str, ratio: int, *, config: dict[str, Any] | None = None) -> str:
    if config is not None:
        template = config["methods"][method].get("method_key_template")
        if template is not None:
            return str(template).format(model=model, ratio=ratio)
    if method == "aqlm":
        return f"aqlm_euclidean_beam_encoder_ple_matched_qat_parent_{model}_r{ratio}_seed0"
    if method == "additive_quantization":
        if config is not None:
            template = config["methods"][method].get("method_key_template")
            if template is not None:
                return str(template).format(model=model, ratio=ratio)
        return f"full_additive_quantization_ple_{model}_r{ratio}_seed0"
    return f"w4a16_{model}_{method}_r{ratio}_seed0"


def paths(
    model: str, method: str, ratio: int, *, config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    key = method_key(model, method, ratio, config=config)
    family = {} if config is None else config["methods"][method]
    artifact_template = family.get("artifact_root_template")
    reconstruction_template = family.get("reconstruction_template")
    method_config_template = family.get("method_config_template")
    roundtrip_receipt_template = family.get("roundtrip_receipt_template")
    runtime_receipt_template = family.get("runtime_receipt_template")
    if artifact_template and reconstruction_template and method_config_template:
        artifact = Path(str(artifact_template).format(model=model, ratio=ratio))
        reconstruction = Path(str(reconstruction_template).format(model=model, ratio=ratio))
        method_config = Path(str(method_config_template).format(model=model, ratio=ratio))
    elif method == "aqlm":
        artifact = ROOT / "artifacts" / "w4a16-matched-qat-parent-aqlm-euclidean-beam-encoder-ple" / f"{model}-r{ratio}-seed0"
        reconstruction = Path(SCRATCH_DIR, "w4a16-matched-qat-parent-aqlm-euclidean-beam-encoder-ple") / f"{model}-r{ratio}-seed0" / "dense_ple_reconstruction.safetensors"
        method_config = ROOT / "configs" / "methods" / f"{key}.json"
    elif method == "additive_quantization":
        artifact_template = family.get("artifact_root_template")
        reconstruction_template = family.get("reconstruction_template")
        method_config_template = family.get("method_config_template")
        artifact = (
            ROOT / "artifacts" / "full-additive-quantization-ple-w4" / model / f"r{ratio}" / "seed0"
            if artifact_template is None
            else Path(str(artifact_template).format(model=model, ratio=ratio))
        )
        reconstruction = (
            Path(SCRATCH_DIR, "full-additive-quantization-ple-w4") / model / f"r{ratio}" / "seed0" / "dense_ple_reconstruction.safetensors"
            if reconstruction_template is None
            else Path(str(reconstruction_template).format(model=model, ratio=ratio))
        )
        method_config = (
            ROOT / "configs" / "methods" / f"full_additive_quantization_ple_{model}_r{ratio}.json"
            if method_config_template is None
            else Path(str(method_config_template).format(model=model, ratio=ratio))
        )
    else:
        artifact = ROOT / "artifacts" / "w4a16-baseline-matrix" / model / method / f"r{ratio}" / "seed0"
        reconstruction = Path(SCRATCH_DIR, "w4a16-baseline-matrix") / model / method / f"r{ratio}" / "seed0" / "dense_ple_reconstruction.safetensors"
        method_config = ROOT / "configs" / "methods" / f"{key}.json"
    output_root = (
        ROOT / "results" / "w4a16-baseline-matrix"
        if config is None or config.get("output_root") is None
        else Path(str(config["output_root"]))
    )
    collection_root = (
        ROOT / "receipts" / "w4a16-baseline-matrix"
        if config is None or config.get("collection_root") is None
        else Path(str(config["collection_root"]))
    )
    output = output_root / key
    summary_name = "metrics.json" if method.startswith("opq_") else "summary.json"
    build_log_name = "build.log"
    return {
        "artifact_root": artifact,
        "compact": artifact / "compressed_ple.npz",
        # The OPQ builder rejects any output basename other than metrics.json.
        "summary": artifact / summary_name,
        "build_log": artifact / build_log_name,
        "identity_weights": artifact / "identity_coord_weights.npy",
        "roundtrip_receipt": (
            artifact / "requested_row_roundtrip_receipt.json"
            if roundtrip_receipt_template is None
            else Path(str(roundtrip_receipt_template).format(model=model, ratio=ratio))
        ),
        "runtime_receipt": (
            artifact / "requested_row_runtime_receipt.json"
            if runtime_receipt_template is None
            else Path(str(runtime_receipt_template).format(model=model, ratio=ratio))
        ),
        "reconstruction": reconstruction,
        "method_config": method_config,
        "output_root": output,
        "l1_output": output / "l1" / "l1_quality_bundle",
        "collection": collection_root / f"{key}.json",
    }


def geometry(config: dict[str, Any], model: str, method: str, ratio: int) -> dict[str, int | float]:
    spec = config["models"][model]
    family = config["methods"][method]
    vocab, layers, dimension = int(spec["vocab_size"]), int(spec["num_layers"]), int(spec["d_ple"])
    rows = vocab * layers
    dense = int(spec["dense_ple_bytes"])
    if family.get("builder_interface") == "prebuilt_source_bound_artifact":
        declared = family["ratios"][str(ratio)]
        resident = int(declared["resident_bytes"])
        require(resident > 0, f"{cell_id(model, method, ratio)} resident bytes must be positive")
        extra = {"geometry_source": "prebuilt_source_bound_artifact"}
    elif method in {"pq", "opq_parametric", "opq_nonparametric"}:
        rate = family["ratios"][str(ratio)]
        subspaces, clusters = int(rate["num_subspaces"]), int(rate["clusters"])
        require(dimension % subspaces == 0, f"{cell_id(model, method, ratio)} subspaces do not divide d_ple")
        bits = math.ceil(math.log2(clusters))
        code_bytes = math.ceil(rows * subspaces * bits / 8)
        codebook_bytes = subspaces * clusters * (dimension // subspaces) * 2
        resident = code_bytes + codebook_bytes
        extra: dict[str, int | float] = {"bits": bits, "code_bytes": code_bytes, "codebook_bytes": codebook_bytes}
        if method.startswith("opq_"):
            rotation_bytes = dimension * dimension * 2
            mean_bytes = dimension * 2
            coordinate_scale_bytes = layers * dimension * 2
            resident += rotation_bytes + mean_bytes + coordinate_scale_bytes
            extra.update(rotation_bytes=rotation_bytes, mean_bytes=mean_bytes, coordinate_scale_bytes=coordinate_scale_bytes)
    elif method in {"rvq", "aqlm", "additive_quantization"}:
        rate = family["ratios"][str(ratio)]
        codebooks, clusters = int(rate["codebooks"]), int(rate["clusters"])
        bits = math.ceil(math.log2(clusters))
        code_bytes = math.ceil(rows * codebooks * bits / 8)
        codebook_bytes = codebooks * clusters * dimension * 2
        resident = code_bytes + codebook_bytes
        extra = {"bits": bits, "code_bytes": code_bytes, "codebook_bytes": codebook_bytes}
    else:
        rank = int(spec["low_rank"][str(ratio)])
        factor_f_bytes = vocab * rank
        factor_g_bytes = rank * layers * dimension
        scale_f_bytes = 2 * rank
        scale_g_bytes = 2 * rank * layers
        resident = factor_f_bytes + factor_g_bytes + scale_f_bytes + scale_g_bytes
        extra = {
            "rank": rank,
            "factor_f_bytes": factor_f_bytes,
            "factor_g_bytes": factor_g_bytes,
            "scale_f_bytes": scale_f_bytes,
            "scale_g_bytes": scale_g_bytes,
        }
    return {**extra, "resident_bytes": resident, "compression_ratio": dense / resident}


def validate_config(config: dict[str, Any]) -> None:
    require(
        config.get("schema_version") in {"w4a16-ple-baseline-matrix-v1", "ple-baseline-matrix-v1"},
        "matrix schema differs",
    )
    require(config.get("status") == "fixed", "matrix config is not fixed")
    require(int(config.get("seed", -1)) == 0, "matrix seed must be zero")
    protocol = read_json(Path(config["protocol_lock"]))
    require(protocol.get("status") == "locked", "protocol lock status differs")
    require(protocol.get("level_entries", {}).get("L1") == ["l1_quality_bundle"], "L1 protocol differs")
    require(set(protocol.get("level_entries", {}).get("L2", [])) == set(L2_BENCHMARKS), "L2 protocol differs")
    for model in configured_models(config):
        spec = config["models"][model]
        require(int(spec["dense_ple_bytes"]) == int(spec["vocab_size"]) * int(spec["num_layers"]) * int(spec["d_ple"]) * 2, f"{model} dense bytes differ")
        for key in ("parent_model_config", "evaluation_model_config"):
            model_config = read_json(Path(spec[key]))
            require(int(model_config["ple_dense_bytes"]) == int(spec["dense_ple_bytes"]), f"{model} {key} PLE bytes differ")
        require(Path(spec["parent_weights"]).is_file(), f"{model} QAT-parent weights are missing")
    for method in configured_methods(config):
        family = config["methods"][method]
        if family.get("availability") == "blocked":
            continue
        require(Path(family["builder"]).is_file(), f"{method} builder is missing")
        for model in configured_models(config):
            for ratio in configured_ratios(config):
                measured = geometry(config, model, method, ratio)
                require(float(measured["compression_ratio"]) >= ratio, f"{cell_id(model, method, ratio)} misses byte budget")


def matrix_contract(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for model in configured_models(config):
        for method in configured_methods(config):
            for ratio in configured_ratios(config):
                family = config["methods"][method]
                blocked = family.get("availability") == "blocked"
                rows.append({
                    "cell": cell_id(model, method, ratio),
                    "method_key": method_key(model, method, ratio, config=config),
                    "availability": "blocked" if blocked else "ready",
                    "blocked_reason": family.get("blocked_reason") if blocked else None,
                    "geometry": geometry(config, model, method, ratio),
                    "artifacts": {name: str(path) for name, path in paths(model, method, ratio, config=config).items() if name in {"compact", "summary", "reconstruction", "method_config"}},
                })
    return {
        "schema_version": "w4a16-ple-baseline-matrix-contract-v1",
        "status": "passed_not_enqueued",
        "seed": 0,
        "ready_cells": sum(row["availability"] == "ready" for row in rows),
        "blocked_cells": sum(row["availability"] == "blocked" for row in rows),
        "blocked_policy": "Never substitute a non-matching source artifact.",
        "rows": rows,
    }


def build_command(config: dict[str, Any], model: str, method: str, ratio: int, device: str) -> list[str]:
    family, spec = config["methods"][method], config["models"][model]
    require(family.get("availability") != "blocked", f"{cell_id(model, method, ratio)} is blocked: {family.get('blocked_reason')}")
    output = paths(model, method, ratio, config=config)
    if family.get("builder_interface") == "prebuilt_source_bound_artifact":
        return [str(PYTHON), str(family["builder"]), "run"]
    if method == "aqlm":
        if family.get("builder_interface") == "original_bf16_e2b_aqlm_pipeline":
            return [str(PYTHON), str(family["builder"]), "run", "--device", device]
        if family.get("builder_interface") == "w4a16_aqlm_euclidean_beam_encoder_pipeline":
            return [
                str(PYTHON), str(family["builder"]), "run",
                "--config", str(family["builder_config"]), "--model", model,
                "--device", device,
            ]
        return [
            str(PYTHON), str(family["builder"]), "run",
            "--config", str(family["builder_config"]), "--model", model,
            "--ratio", f"r{ratio}", "--device", device,
        ]
    if method == "additive_quantization":
        return [
            str(PYTHON), str(family["builder"]), "--config", str(family["builder_config"]),
            "--model", model, "--ratio", str(ratio), "--device", device,
            "--output-root", str(output["artifact_root"]),
            "--save-recon", str(output["reconstruction"]),
            "--resume",
        ]
    common = [
        str(PYTHON), str(family["builder"]), "--it-weights", str(spec["parent_weights"]),
        "--num-layers", str(spec["num_layers"]), "--d-ple", str(spec["d_ple"]),
    ]
    if method == "pq":
        rate = family["ratios"][str(ratio)]
        common += ["--num-subspaces", str(rate["num_subspaces"]), "--clusters", str(rate["clusters"]), "--iters", str(rate["iterations"])]
    elif method == "rvq":
        rate = family["ratios"][str(ratio)]
        common += ["--codebooks", str(rate["codebooks"]), "--clusters", str(rate["clusters"]), "--iters", str(rate["iterations"])]
    elif method.startswith("opq_"):
        rate = family["ratios"][str(ratio)]
        common += [
            "--num-subspaces", str(rate["num_subspaces"]), "--clusters", str(rate["clusters"]), "--iters", str(rate["iterations"]),
            "--exception-count", "0", "--exception-scale", "0.0", "--exception-selector", "ppl_text_frequency",
            "--coord-weights", str(output["identity_weights"]), "--coord-weights-provenance", "model_only_identity_control",
            "--coord-weight-alpha", "1.0",
        ]
    elif method == "low_rank_svd":
        common += [
            "--rank", str(spec["low_rank"][str(ratio)]),
            "--power-iters", str(family["power_iterations"]), "--oversample", str(family["oversample"]),
        ]
    else:
        raise RuntimeError(f"unsupported build method: {method}")
    common += [
        "--batch-size", "8192", "--eval-batch-size", "8192",
        "--device", device, "--seed", "0", "--save-recon", str(output["reconstruction"]),
        "--save-compressed-ple-npz", str(output["compact"]), "--output", str(output["summary"]),
    ]
    if method == "low_rank_svd":
        # This builder does not expose the common k-means batch options.
        del common[common.index("--batch-size"):common.index("--device")]
    return common


def prepare_identity_weights(path: Path, layers: int, dimension: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        value = np.load(path, allow_pickle=False)
        require(tuple(value.shape) == (layers, dimension) and bool(np.all(value == 1.0)), "identity weights differ")
    else:
        np.save(path, np.ones((layers, dimension), dtype=np.float32), allow_pickle=False)


def _bfloat16_comparison(actual: Any, expected: Any) -> dict[str, int | float]:
    import torch

    require(tuple(actual.shape) == tuple(expected.shape), "BF16 comparison shape differs")
    require(actual.dtype == torch.bfloat16 and expected.dtype == torch.bfloat16,
            "round trip must compare BF16 materializations")
    absolute = (actual.float() - expected.float()).abs()
    code_distance = (
        actual.contiguous().view(torch.int16).to(torch.int32)
        - expected.contiguous().view(torch.int16).to(torch.int32)
    ).abs()
    return {
        "different_elements": int((actual != expected).sum().item()),
        "max_abs_diff_after_bfloat16_materialization": float(absolute.max().item()),
        "mean_abs_diff_after_bfloat16_materialization": float(absolute.mean().item()),
        "max_bfloat16_code_distance": int(code_distance.max().item()),
    }


def validate_quantized_low_rank_requested_rows(
    compact_path: Path,
    reconstruction_path: Path,
    requested: Any,
    *,
    compact_rows: Any | None = None,
) -> dict[str, Any]:
    """Validate persisted QLR factors against formula and canonical capability."""
    import numpy as np
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(ROOT / "src"))
    from ple_eval.compact_adapter import load_compact_ple

    if compact_rows is None:
        with torch.inference_mode():
            compact_rows = load_compact_ple(compact_path, device="cpu")(requested).cpu()
    else:
        compact_rows = compact_rows.cpu()

    # This is deliberately independent of CompactQuantizedLowRank.forward: it
    # reads the persisted arrays and evaluates the documented factor formula.
    with np.load(compact_path, allow_pickle=False) as archive:
        q_f = torch.from_numpy(np.array(archive["q_f_int8"], copy=True))
        f_scale = torch.from_numpy(np.array(archive["f_scale_fp16"], copy=True))
        q_g = torch.from_numpy(np.array(archive["q_g_int8"], copy=True))
        g_scale = torch.from_numpy(np.array(archive["g_scale_fp16"], copy=True))
    left = q_f.index_select(0, requested.cpu().to(torch.long)).to(torch.float32)
    left.mul_(f_scale.to(torch.float32))
    right = q_g.to(torch.float32)
    right.mul_(g_scale.to(torch.float32).unsqueeze(2))
    independent_rows = (left @ right.reshape(right.shape[0], -1)).to(torch.bfloat16)

    with safe_open(reconstruction_path, framework="pt", device="cpu") as handle:
        keys = [name for name in handle.keys() if name.endswith("embed_tokens_per_layer.weight")]
        require(len(keys) == 1, "capability reconstruction PLE key is ambiguous")
        source = handle.get_slice(keys[0])
        capability_rows = torch.cat(
            [source[int(index):int(index) + 1] for index in requested.cpu()], dim=0
        )

    independent = _bfloat16_comparison(compact_rows, independent_rows)
    capability = _bfloat16_comparison(compact_rows, capability_rows)
    maximum_code_distance = 1
    require(
        int(independent["max_bfloat16_code_distance"]) <= maximum_code_distance,
        "persisted-factor independent formula exceeds the BF16 code-distance rule: "
        f"{independent['max_bfloat16_code_distance']} > {maximum_code_distance}",
    )
    require(
        int(capability["max_bfloat16_code_distance"]) <= maximum_code_distance,
        "canonical capability subset exceeds the BF16 code-distance rule: "
        f"{capability['max_bfloat16_code_distance']} > {maximum_code_distance}",
    )
    return {
        "comparison_rule": "persisted_compact_formula_and_canonical_capability_bfloat16_adjacent_or_equal",
        "maximum_bfloat16_code_distance": maximum_code_distance,
        "independent_formula": independent,
        "capability_subset": capability,
    }


def validate_and_record_build(config: dict[str, Any], model: str, method: str, ratio: int) -> dict[str, Any]:
    if config["methods"][method].get("builder_interface") == "prebuilt_source_bound_artifact":
        output = paths(model, method, ratio, config=config)
        expected = geometry(config, model, method, ratio)
        require(output["compact"].is_file(), "prebuilt compact artifact is missing")
        require(output["reconstruction"].is_file(), "prebuilt capability reconstruction is missing")
        require(output["method_config"].is_file(), "prebuilt method config is missing")
        selected = read_json(output["method_config"])
        require(selected.get("status") == "ready_for_evaluation", "prebuilt method is not ready")
        require(selected.get("method_id") == method_key(model, method, ratio, config=config), "prebuilt method id differs")
        require(int(selected.get("ple_resident_bytes", -1)) == int(expected["resident_bytes"]), "prebuilt resident bytes differ")
        require(Path(str(selected.get("compressed_artifact_path"))) == output["compact"], "prebuilt artifact path differs")
        require(Path(str(selected.get("reconstruction_path"))) == output["reconstruction"], "prebuilt reconstruction path differs")
        require(selected.get("retention_reference_method_key") == config["models"][model]["dense_reference_method_key"], "prebuilt dense reference differs")
        require(selected.get("compressed_artifact_sha256") == file_sha256(output["compact"]), "prebuilt artifact digest differs")
        require(output["reconstruction"].stat().st_size == int(config["models"][model]["dense_ple_bytes"]) + 112, "prebuilt reconstruction bytes differ")
        for receipt_key in ("runtime_receipt_path", "roundtrip_receipt_path"):
            receipt_path = Path(str(selected.get(receipt_key)))
            require(receipt_path.is_file() and read_json(receipt_path).get("status") == "passed", f"prebuilt {receipt_key} is not passed")
        return {
            "status": "ready_for_evaluation",
            "cell": cell_id(model, method, ratio),
            "method_config": str(output["method_config"]),
            "resident_bytes": int(expected["resident_bytes"]),
            "compression_ratio": float(expected["compression_ratio"]),
        }
    if method == "aqlm":
        if config["methods"][method].get("builder_interface") == "original_bf16_e2b_aqlm_pipeline":
            return validate_original_bf16_aqlm_build(config, model, ratio)
        if config["methods"][method].get("builder_interface") == "w4a16_aqlm_euclidean_beam_encoder_pipeline":
            return validate_w4a16_euclidean_beam_encoder_aqlm_build(config, model, ratio)
        return validate_matched_aqlm_build(config, model, ratio)
    if method == "additive_quantization":
        return validate_full_aq_build(config, model, ratio)
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(ROOT / "src"))
    from ple_eval.compact_adapter import artifact_sha256, compact_resident_bytes, load_compact_ple

    output = paths(model, method, ratio, config=config)
    for name in ("compact", "summary", "reconstruction"):
        require(output[name].is_file(), f"missing build output: {output[name]}")
    summary = read_json(output["summary"])
    spec, family = config["models"][model], config["methods"][method]
    expected = geometry(config, model, method, ratio)
    measured = int(summary.get("storage_bytes", -1))
    measured_ratio = float(summary.get("compression_ratio", -1))
    require(int(summary.get("seed", -1)) == 0, "builder seed differs")
    require(summary.get("ple_shape") == [int(spec["vocab_size"]), int(spec["num_layers"]) * int(spec["d_ple"])], "builder PLE shape differs")
    require(measured == int(expected["resident_bytes"]), f"builder storage bytes differ: {measured} != {expected['resident_bytes']}")
    require(measured <= int(spec["dense_ple_bytes"]) // ratio, "builder exceeded nominal byte budget")
    require(math.isclose(measured_ratio, int(spec["dense_ple_bytes"]) / measured, rel_tol=0, abs_tol=1e-12), "builder ratio differs")

    requested = torch.tensor([0, 1, 2, 1024, 131072, 258884], dtype=torch.long)
    started = time.perf_counter()
    module = load_compact_ple(output["compact"], device="cpu")
    load_seconds = time.perf_counter() - started
    resident = int(compact_resident_bytes(module))
    require(resident == measured, f"compact adapter resident bytes differ: {resident} != {measured}")
    started = time.perf_counter()
    with torch.inference_mode():
        actual = module(requested).cpu()
    decode_seconds = time.perf_counter() - started
    expected_rows = None
    if method != "low_rank_svd":
        with safe_open(output["reconstruction"], framework="pt", device="cpu") as handle:
            keys = [name for name in handle.keys() if name.endswith("embed_tokens_per_layer.weight")]
            require(len(keys) == 1, "capability reconstruction PLE key is ambiguous")
            source = handle.get_slice(keys[0])
            expected_rows = torch.cat([source[int(index):int(index) + 1] for index in requested], dim=0)
        require(tuple(actual.shape) == tuple(expected_rows.shape), "requested-row shape differs")
    max_abs_diff = 0.0
    atol = 0.002 if method == "rvq" else 0.0
    comparison_rule = "absolute_tolerance"
    max_bfloat16_code_distance = 0
    low_rank_checks = None
    if method == "low_rank_svd":
        low_rank_checks = validate_quantized_low_rank_requested_rows(
            output["compact"], output["reconstruction"], requested, compact_rows=actual,
        )
        comparison_rule = str(low_rank_checks["comparison_rule"])
        max_bfloat16_code_distance = max(
            int(low_rank_checks["independent_formula"]["max_bfloat16_code_distance"]),
            int(low_rank_checks["capability_subset"]["max_bfloat16_code_distance"]),
        )
        max_abs_diff = max(
            float(low_rank_checks["independent_formula"]["max_abs_diff_after_bfloat16_materialization"]),
            float(low_rank_checks["capability_subset"]["max_abs_diff_after_bfloat16_materialization"]),
        )
    else:
        assert expected_rows is not None
        max_abs_diff = float((actual.float() - expected_rows.float()).abs().max().item())
        require(max_abs_diff <= atol, f"requested-row round trip differs: {max_abs_diff} > {atol}")
    artifact_hash = artifact_sha256(output["compact"])
    common_receipt = {
        "status": "passed", "method": method_key(model, method, ratio, config=config), "artifact_path": str(output["compact"]),
        "artifact_sha256": artifact_hash, "completed_utc": utc_now(), "requested_token_ids": requested.tolist(),
        "requested_rows": len(requested), "output_shape": list(actual.shape), "persistent_resident_bytes": resident,
        "finite": bool(torch.isfinite(actual.float()).all().item()), "full_table_materialized": False,
        "dense_ple_tensor_resident": False, "benchmark_inputs_used": False,
    }
    runtime = {
        "schema_version": "ple-compact-requested-row-runtime-v1", **common_receipt, "device": "cpu",
        "module_load_seconds": load_seconds, "requested_row_decode_seconds": decode_seconds,
    }
    roundtrip = {
        "schema_version": "ple-compact-requested-row-roundtrip-v1", **common_receipt,
        "max_abs_diff_after_bfloat16_materialization": max_abs_diff, "atol": atol,
        "comparison_rule": comparison_rule,
        "max_bfloat16_code_distance": max_bfloat16_code_distance,
        "low_rank_checks": low_rank_checks,
        "cross_backend_requested_row_telemetry": summary.get("cross_backend_requested_row_telemetry"),
        "reconstruction_path": str(output["reconstruction"]), "reconstruction_sha256": file_sha256(output["reconstruction"]),
    }
    write_json_atomic(output["runtime_receipt"], runtime)
    write_json_atomic(output["roundtrip_receipt"], roundtrip)
    method_config = {
        "schema_version": "ple-method-config-v1", "status": "ready_for_evaluation",
        "method_id": method_key(model, method, ratio, config=config), "label": f"{family['publication_name']} ({measured_ratio:.3f}x)",
        "publication_name": family["publication_name"], "classification": family["classification"],
        "publication_role": family["classification"], "method_family": method,
        "data_access_regime": "model_weights_only", "ratio_bucket": f"r{ratio}",
        "capability_mode": "dense_reconstruction_injection", "deployment_mode": "online_requested_token_row_decode",
        "reconstruction_path": str(output["reconstruction"]), "reconstruction_sha256": roundtrip["reconstruction_sha256"],
        "compressed_artifact_path": str(output["compact"]), "compressed_artifact_sha256": artifact_hash,
        "compressed_artifact_file_bytes": output["compact"].stat().st_size, "format_id": family["format_id"],
        "builder_path": family["builder"], "builder_sha256": file_sha256(Path(family["builder"])),
        "training_metrics_path": str(output["summary"]), "training_metrics_sha256": file_sha256(output["summary"]),
        "reported_reconstruction_nmse": float(summary["decoded_source_ple_nmse"]),
        "ple_resident_bytes": measured, "ple_compression_ratio": measured_ratio,
        "compression_ratio_scope": "all persistent online PLE tensors; BF16 capability materialization excluded",
        "roundtrip_receipt_path": str(output["roundtrip_receipt"]), "runtime_receipt_path": str(output["runtime_receipt"]),
        "seed": 0, 
        "retention_reference_method_key": spec["dense_reference_method_key"],
    }
    if config.get("schema_version") == "w4a16-ple-baseline-matrix-v1":
        method_config["source_qat_parent_model_config"] = spec["parent_model_config"]
        method_config["evaluation_w4a16_model_config"] = spec["evaluation_model_config"]
    else:
        method_config["source_model_config"] = spec["parent_model_config"]
        method_config["evaluation_model_config"] = spec["evaluation_model_config"]
    if output["method_config"].is_file():
        # Evaluation outputs bind the existing method config, so a capability
        # rematerialization keeps it unchanged.
        existing_method = read_json(output["method_config"])
        require(existing_method.get("status") == "ready_for_evaluation", "existing method config is not ready")
        require(existing_method.get("method_id") == method_config["method_id"], "existing method id differs")
        require(existing_method.get("compressed_artifact_sha256") == artifact_hash, "existing artifact binding differs")
        require(int(existing_method.get("ple_resident_bytes", -1)) == measured, "existing resident bytes differ")
        require(
            math.isclose(float(existing_method.get("ple_compression_ratio", -1)), measured_ratio, rel_tol=0, abs_tol=1e-12),
            "existing compression ratio differs",
        )
        require(existing_method.get("reconstruction_sha256") == roundtrip["reconstruction_sha256"], "existing reconstruction binding differs")
    else:
        write_json_atomic(output["method_config"], method_config)
    return {"status": "ready_for_evaluation", "cell": cell_id(model, method, ratio), "method_config": str(output["method_config"]), "resident_bytes": measured, "compression_ratio": measured_ratio}


def validate_original_bf16_aqlm_build(config: dict[str, Any], model: str, ratio: int) -> dict[str, Any]:
    """Check the AQLM beam-encoder artifact."""
    sys.path.insert(0, str(ROOT / "src"))
    from ple_eval.compact_adapter import artifact_sha256, compact_resident_bytes, load_compact_ple

    output = paths(model, "aqlm", ratio, config=config)
    for name in ("compact", "summary", "reconstruction", "runtime_receipt", "roundtrip_receipt", "method_config"):
        require(output[name].is_file(), f"missing original-BF16 AQLM output: {output[name]}")
    summary = read_json(output["summary"])
    method_config = read_json(output["method_config"])
    expected = geometry(config, model, "aqlm", ratio)
    require(summary.get("classification") == "adapted_baseline", "AQLM classification differs")
    require(summary.get("objective_update") == "author_aqlm_l2_beam_search", "AQLM author objective differs")
    require(float(summary["objective_nmse_after_fp32_decode"]) <= float(summary["objective_nmse_before"]), "AQLM objective increased")
    require(int(summary["resident_bytes"]) == int(expected["resident_bytes"]), "AQLM resident bytes differ")
    require(math.isclose(float(summary["compression_ratio"]), float(expected["compression_ratio"]), rel_tol=0, abs_tol=1e-12), "AQLM ratio differs")
    require(method_config.get("method_id") == method_key(model, "aqlm", ratio, config=config), "AQLM method id differs")
    require(method_config.get("status") == "ready_for_evaluation", "AQLM method config is not ready")
    require(method_config.get("compressed_artifact_sha256") == artifact_sha256(output["compact"]), "AQLM artifact binding differs")
    require(read_json(output["runtime_receipt"]).get("status") == "passed", "AQLM requested-row runtime failed")
    require(read_json(output["roundtrip_receipt"]).get("status") == "passed", "AQLM requested-row round trip failed")
    module = load_compact_ple(output["compact"], device="cpu")
    require(int(compact_resident_bytes(module)) == int(expected["resident_bytes"]), "AQLM compact adapter bytes differ")
    return {
        "status": "ready_for_evaluation", "cell": cell_id(model, "aqlm", ratio),
        "method_config": str(output["method_config"]), "resident_bytes": int(expected["resident_bytes"]),
        "compression_ratio": float(expected["compression_ratio"]),
    }


def validate_w4a16_euclidean_beam_encoder_aqlm_build(
    config: dict[str, Any], model: str, ratio: int,
) -> dict[str, Any]:
    """Validate the W4A16 AQLM beam-encoder adaptation."""
    sys.path.insert(0, str(ROOT / "src"))
    from ple_eval.compact_adapter import artifact_sha256, compact_resident_bytes, load_compact_ple

    output = paths(model, "aqlm", ratio, config=config)
    for name in ("compact", "summary", "reconstruction", "runtime_receipt", "roundtrip_receipt", "method_config"):
        require(output[name].is_file(), f"missing AQLM beam-encoder output: {output[name]}")
    receipt_path = output["artifact_root"] / "build-receipt.json"
    require(receipt_path.is_file(), "AQLM beam-encoder build receipt is missing")
    receipt = read_json(receipt_path)
    summary = read_json(output["summary"])
    method_config = read_json(output["method_config"])
    expected = geometry(config, model, "aqlm", ratio)
    measured = int(expected["resident_bytes"])
    measured_ratio = float(expected["compression_ratio"])
    require(receipt.get("status") == "ready_for_evaluation", "direct AQLM build is not ready")
    require(receipt.get("author_beam_encoder", {}).get("passes") == 1, "direct AQLM beam schedule differs")
    require(
        receipt.get("author_beam_encoder", {}).get("proposal_acceptance") == "direct_unfiltered_author_output"
        and receipt.get("author_beam_encoder", {}).get("retain_nonworsening_safeguard") is False,
        "direct AQLM proposal was filtered",
    )
    require(summary.get("classification") == "adapted_baseline", "direct AQLM classification differs")
    require(int(summary.get("resident_bytes", -1)) == measured, "direct AQLM resident bytes differ")
    require(math.isclose(float(summary.get("compression_ratio", -1)), measured_ratio, rel_tol=0, abs_tol=1e-12), "direct AQLM ratio differs")
    require(method_config.get("status") == "ready_for_evaluation", "direct AQLM method is not ready")
    require(method_config.get("method_id") == method_key(model, "aqlm", ratio, config=config), "direct AQLM method id differs")
    require(method_config.get("retain_nonworsening_safeguard") is False, "direct AQLM method enables safeguard")
    require(method_config.get("compressed_artifact_sha256") == artifact_sha256(output["compact"]), "direct AQLM artifact binding differs")
    require(read_json(output["runtime_receipt"]).get("status") == "passed", "direct AQLM requested-row runtime failed")
    require(read_json(output["roundtrip_receipt"]).get("status") == "passed", "direct AQLM requested-row round trip failed")
    module = load_compact_ple(output["compact"], device="cpu")
    require(int(compact_resident_bytes(module)) == measured, "direct AQLM compact adapter bytes differ")
    return {
        "status": "ready_for_evaluation", "cell": cell_id(model, "aqlm", ratio),
        "method_config": str(output["method_config"]), "resident_bytes": measured,
        "compression_ratio": measured_ratio,
    }


def validate_matched_aqlm_build(config: dict[str, Any], model: str, ratio: int) -> dict[str, Any]:
    """Validate the matched-QAT-parent AQLM Euclidean beam-encoder adaptation."""
    sys.path.insert(0, str(ROOT / "src"))
    from ple_eval.compact_adapter import artifact_sha256, compact_resident_bytes, load_compact_ple

    output = paths(model, "aqlm", ratio, config=config)
    for name in ("compact", "summary", "reconstruction", "runtime_receipt", "roundtrip_receipt", "method_config"):
        require(output[name].is_file(), f"missing matched AQLM build output: {output[name]}")
    summary = read_json(output["summary"])
    method = bind_retention_reference(config, model, output["method_config"])
    runtime = read_json(output["runtime_receipt"])
    roundtrip = read_json(output["roundtrip_receipt"])
    expected = geometry(config, model, "aqlm", ratio)
    measured = int(summary.get("actual_packed_resident_bytes", -1))
    measured_ratio = float(summary.get("compression_ratio", -1))
    require(summary.get("status") == "completed_author_beam_fixed_point", "AQLM author beam encoder did not converge")
    require(summary.get("classification") == "adapted_baseline", "AQLM classification differs")
    require(summary.get("author_beam_convergence", {}).get("converged") is True, "AQLM fixed-point receipt differs")
    require(measured == int(expected["resident_bytes"]), "AQLM packed resident bytes differ")
    require(measured <= int(config["models"][model]["dense_ple_bytes"]) // ratio, "AQLM exceeded nominal byte budget")
    require(math.isclose(measured_ratio, float(expected["compression_ratio"]), rel_tol=0, abs_tol=1e-12), "AQLM ratio differs")
    require(method.get("status") == "ready_for_evaluation", "AQLM method config is not ready")
    require(method.get("method_id") == method_key(model, "aqlm", ratio, config=config), "AQLM method id differs")
    require(method.get("compressed_artifact_sha256") == artifact_sha256(output["compact"]), "AQLM artifact hash differs")
    require(runtime.get("status") == "passed", "AQLM requested-row runtime failed")
    require(int(runtime.get("actual_packed_persistent_resident_bytes", -1)) == measured, "AQLM runtime bytes differ")
    require(roundtrip.get("status") == "passed", "AQLM requested-row round trip failed")
    module = load_compact_ple(output["compact"], device="cpu")
    require(int(compact_resident_bytes(module)) == measured, "AQLM compact adapter bytes differ")
    return {
        "status": "ready_for_evaluation", "cell": cell_id(model, "aqlm", ratio),
        "method_config": str(output["method_config"]), "resident_bytes": measured,
        "compression_ratio": measured_ratio,
    }


def validate_full_aq_build(config: dict[str, Any], model: str, ratio: int) -> dict[str, Any]:
    """Check the Additive Quantization artifact."""
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from aq_codec import (
        artifact_sha256, compact_resident_bytes, load_full_aq_ple,
    )

    output = paths(model, "additive_quantization", ratio, config=config)
    for name in ("compact", "summary", "reconstruction", "roundtrip_receipt", "method_config"):
        require(output[name].is_file(), f"missing full-AQ build output: {output[name]}")
    summary = read_json(output["summary"])
    method = bind_retention_reference(config, model, output["method_config"])
    roundtrip = read_json(output["roundtrip_receipt"])
    expected = geometry(config, model, "additive_quantization", ratio)
    measured = int(summary.get("resident_bytes", -1))
    measured_ratio = float(summary.get("compression_ratio", -1))
    require(summary.get("method_family") == "full_additive_quantization", "full-AQ summary family differs")
    required_training_status = str(config["methods"]["additive_quantization"].get("required_training_status", "converged"))
    require(summary.get("training_status") == required_training_status, "full-AQ completion contract did not pass")
    require(summary.get("final_artifact_emitted") is True, "full-AQ final artifact was not emitted")
    require(measured == int(expected["resident_bytes"]), "full-AQ packed resident bytes differ")
    require(measured <= int(config["models"][model]["dense_ple_bytes"]) // ratio, "full-AQ exceeded nominal byte budget")
    require(math.isclose(measured_ratio, float(expected["compression_ratio"]), rel_tol=0, abs_tol=1e-12), "full-AQ ratio differs")
    require(method.get("status") == "ready_for_evaluation", "full-AQ method config is not ready")
    require(method.get("method_id") == method_key(model, "additive_quantization", ratio, config=config), "full-AQ method id differs")
    require(Path(method["compressed_artifact_path"]).resolve() == output["compact"].resolve(), "full-AQ artifact binding differs")
    require(Path(method["reconstruction_path"]).resolve() == output["reconstruction"].resolve(), "full-AQ reconstruction binding differs")
    require(method.get("compressed_artifact_sha256") == artifact_sha256(output["compact"]), "full-AQ artifact hash differs")
    require(
        roundtrip.get("status") == "passed"
        and roundtrip.get("exact_bfloat16") is True
        and float(roundtrip.get("max_abs_diff_after_bfloat16_materialization", float("inf"))) == 0.0,
        "full-AQ requested-row round trip failed",
    )
    adapter = Path(str(method.get("online_adapter_path", "")))
    require(adapter.resolve() == (ROOT / "scripts/aq_codec.py").resolve(), "full-AQ online adapter path differs")
    require(method.get("online_adapter_sha256") == file_sha256(adapter), "full-AQ online adapter binding differs")
    require(Path(str(method.get("runtime_receipt_path", ""))).resolve() == output["runtime_receipt"].resolve(), "full-AQ runtime receipt path differs")

    started = time.perf_counter()
    module = load_full_aq_ple(output["compact"], device="cpu")
    load_seconds = time.perf_counter() - started
    require(int(compact_resident_bytes(module)) == measured, "full-AQ compact adapter bytes differ")
    requested = torch.tensor([0, 1, 2, 1024, 131072, 258884], dtype=torch.long)
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = module(requested)
    decode_seconds = time.perf_counter() - started
    require(bool(torch.isfinite(decoded.float()).all().item()), "full-AQ requested rows are non-finite")
    runtime = {
        "schema_version": "ple-compact-requested-row-runtime-v1", "status": "passed",
        "method": method_key(model, "additive_quantization", ratio, config=config),
        "artifact_path": str(output["compact"]), "artifact_sha256": artifact_sha256(output["compact"]),
        "device": "cpu", "requested_token_ids": requested.tolist(),
        "requested_rows": len(requested), "output_shape": list(decoded.shape),
        "persistent_resident_bytes": measured, "finite": True,
        "full_table_materialized": False, "dense_ple_tensor_resident": False,
        "benchmark_inputs_used": False, "module_load_seconds": load_seconds,
        "requested_row_decode_seconds": decode_seconds, "completed_utc": utc_now(),
    }
    write_json_atomic(output["runtime_receipt"], runtime)
    return {
        "status": "ready_for_evaluation", "cell": cell_id(model, "additive_quantization", ratio),
        "method_config": str(output["method_config"]), "resident_bytes": measured,
        "compression_ratio": measured_ratio,
    }


def run_build(config: dict[str, Any], model: str, method: str, ratio: int, device: str) -> dict[str, Any]:
    output = paths(model, method, ratio, config=config)
    if config["methods"][method].get("builder_interface") == "prebuilt_source_bound_artifact":
        if output["compact"].is_file() and output["method_config"].is_file() and output["reconstruction"].is_file():
            return {"action": "reuse", **validate_and_record_build(config, model, method, ratio)}
        command = build_command(config, model, method, ratio, device)
        build_log = Path(config["state_root"]) / "build-logs" / f"{method_key(model, method, ratio, config=config)}.log"
        build_log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"})
        with build_log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        require(completed.returncode == 0, f"prebuilt source-bound pipeline failed with {completed.returncode}; see {build_log}")
        return {"action": "built", "build_log": str(build_log), **validate_and_record_build(config, model, method, ratio)}
    required = (output["compact"], output["summary"], output["reconstruction"])
    if output["compact"].is_file() and output["summary"].is_file() and not output["reconstruction"].exists():
        # Recreate a missing dense reconstruction from the fitted compact artifact.
        sys.path.insert(0, str(ROOT / "src"))
        from ple_eval.materialize import materialize

        output["reconstruction"].parent.mkdir(parents=True, exist_ok=True)
        receipt = materialize(
            output["compact"], output["reconstruction"], device=device, chunk_rows=4096,
        )
        write_json_atomic(
            output["reconstruction"].parent / "capability-materialization-receipt.json", receipt,
        )
        if output["method_config"].is_file():
            existing_method = read_json(output["method_config"])
            expected_hash = existing_method.get("reconstruction_sha256")
            if expected_hash is not None:
                require(
                    receipt["reconstruction_sha256"] == expected_hash,
                    "fresh capability reconstruction differs from the method binding",
                )
        return {"action": "rematerialized", **validate_and_record_build(config, model, method, ratio)}
    if any(path.exists() for path in required):
        require(all(path.is_file() for path in required), "partial build outputs exist; preserve and inspect them before recovery")
        return {"action": "reuse", **validate_and_record_build(config, model, method, ratio)}
    output["artifact_root"].mkdir(parents=True, exist_ok=True)
    output["reconstruction"].parent.mkdir(parents=True, exist_ok=True)
    if method.startswith("opq_"):
        spec = config["models"][model]
        prepare_identity_weights(output["identity_weights"], int(spec["num_layers"]), int(spec["d_ple"]))
    command = build_command(config, model, method, ratio, device)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"})
    with output["build_log"].open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    require(completed.returncode == 0, f"builder failed with {completed.returncode}; see {output['build_log']}")
    return {"action": "built", **validate_and_record_build(config, model, method, ratio)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "build-command", "run-build", "finalize-build"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cell")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    if args.action == "contract": result = matrix_contract(config)
    else:
        model, method, ratio = parse_cell(config, args.cell)
        if args.action == "build-command": result = build_command(config, model, method, ratio, args.device)
        elif args.action == "run-build": result = run_build(config, model, method, ratio, args.device)
        else: result = validate_and_record_build(config, model, method, ratio)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
