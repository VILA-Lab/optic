#!/usr/bin/env python3
"""Shared fitting helpers for the OPTIC builder."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def set_learning_rate(optimizer: Any, base_lr: float, schedule: str, step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        lr = base_lr * float(step + 1) / float(warmup_steps)
    elif schedule == "constant":
        lr = base_lr
    elif schedule == "cosine":
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps + 1) / denom, 0.0), 1.0)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def benchmark_id_for_metrics_output(output_path: Path) -> str:
    for parent in (output_path.parent, *output_path.parents):
        record_path = parent / "experiment.json"
        if not record_path.is_file():
            continue
        try:
            experiment = load_json(record_path)
        except SystemExit:
            continue
        benchmark_id = experiment.get("benchmark_id")
        if isinstance(benchmark_id, str) and benchmark_id:
            return benchmark_id
    return "build_full_ple_reconstruction"


def load_projection_weights(torch: Any, safe_open: Any, weights_path: str, layers: int, d: int, device):
    key_pattern = "model.language_model.layers.{layer}.per_layer_projection.weight"
    projection_weights = []
    output_dim = None
    projection_digest = hashlib.sha256()
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for layer_idx in range(layers):
            key = key_pattern.format(layer=layer_idx)
            if key not in keys:
                raise SystemExit(f"missing per-layer projection weight key: {key}")
            weight = handle.get_tensor(key).to(torch.float32).contiguous()
            if weight.ndim != 2 or int(weight.shape[1]) != d:
                raise SystemExit(f"projection weight {key} shape {tuple(weight.shape)} must be [out_dim, {d}]")
            if output_dim is None:
                output_dim = int(weight.shape[0])
            elif int(weight.shape[0]) != output_dim:
                raise SystemExit("per-layer projection output dimensions are not consistent")
            projection_digest.update(weight.cpu().numpy().tobytes(order="C"))
            projection_weights.append(weight.to(device=device, dtype=torch.float32))
    return projection_weights, {
        "training_forward_path_projection_weight_file": str(weights_path),
        "training_forward_path_projection_weight_key_pattern": key_pattern,
        "training_forward_path_projection_weight_float32_sha256": projection_digest.hexdigest(),
        "training_forward_path_projection_output_dim": int(output_dim),
    }


def build_exact_gate_cache_forward_path_training_inputs(*_args: Any, **_kwargs: Any):
    raise SystemExit(
        "training inputs are supplied by the fitting script: run scripts/build_optic_e4b_bf16.py, "
        "scripts/build_optic.py or a frequency-tiered OPTIC fitting script"
    )
