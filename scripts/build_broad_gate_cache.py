#!/usr/bin/env python3
"""Materialize original-model Gemma-4 PLE gates for the broad token stream."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/calibration/broad.json"
sys.path.insert(0, str(ROOT / "scripts"))
from build_eval_contamination_signatures import sha256_file  # noqa: E402
from check_disk_space import evaluate_storage_policy  # noqa: E402


DEFAULT_LAYERS = 42
DEFAULT_D_PLE = 256


def expected_gate_bytes(
    positions: int,
    layers: int = DEFAULT_LAYERS,
    d_ple: int = DEFAULT_D_PLE,
) -> int:
    return positions * layers * d_ple * np.dtype("<f2").itemsize


def resolve_token_asset(
    token_dir: Path, token_manifest: dict[str, Any]
) -> tuple[Path, str, int, int]:
    """Resolve the broad calibration stream or the coverage-optimized broad calibration stream."""
    if "selected_token_file" in token_manifest:
        if not token_manifest.get("gate_cache_allowed", False):
            raise RuntimeError("coverage-optimized manifest has not passed its gate-cache threshold")
        return (
            Path(token_manifest["selected_token_file"]),
            str(token_manifest["selected_token_file_sha256"]),
            int(token_manifest["selected_sequence_count"]),
            int(token_manifest["selected_sequence_length"]),
        )
    return (
        token_dir / "calibration_token_ids_le_u32.bin",
        str(token_manifest["token_file_sha256"]),
        int(token_manifest["sequence_count"]),
        int(token_manifest["sequence_length"]),
    )


def check_storage(config: dict[str, Any], phase: str, reserve_bytes: int = 0) -> dict[str, Any]:
    policy = json.loads(Path(config["storage"]["policy"]).read_text(encoding="utf-8"))
    report = evaluate_storage_policy(
        policy, reserve_bytes=reserve_bytes, reserve_scope="corpus", phase=phase
    )
    if not report["ok"]:
        raise RuntimeError(f"storage policy failed at {phase}: {report['checks']}")
    return report


def load_text_model(snapshot: str, device: str):
    import torch
    from transformers import Gemma4ForConditionalGeneration

    full_model = Gemma4ForConditionalGeneration.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    full_model.to(device)
    language_model = full_model.model.language_model
    full_model.model.language_model = None
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    language_model.eval()
    return language_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit-sequences", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    layers_count = int(model_config.get("num_layers", DEFAULT_LAYERS))
    d_ple = int(model_config.get("d_ple", DEFAULT_D_PLE))
    if layers_count <= 0 or d_ple <= 0:
        raise SystemExit("model num_layers and d_ple must be positive")
    token_dir = Path(config["storage"]["output_directory"])
    token_manifest_path = token_dir / "calibration_token_manifest.json"
    if not token_manifest_path.is_file():
        raise SystemExit("broad calibration token assets are missing")
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    try:
        token_path, expected_token_sha256, total_sequences, sequence_length = (
            resolve_token_asset(token_dir, token_manifest)
        )
    except (KeyError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not token_path.is_file():
        raise SystemExit("selected broad calibration token file is missing")
    if expected_token_sha256 != sha256_file(token_path):
        raise SystemExit("broad calibration token hash mismatch")
    sequence_count = args.limit_sequences or total_sequences
    if sequence_count <= 0 or sequence_count > total_sequences:
        raise SystemExit("--limit-sequences must be in the materialized token range")
    complete = sequence_count == total_sequences and args.limit_sequences is None
    gate_cache_version = "v2" if "selected_token_file" in token_manifest else "v1"
    output_dir = args.output_dir or token_dir / f"gate-cache-fp16-{gate_cache_version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "dout_gate_cache_layer_major_f16.bin"
    manifest_path = output_dir / "dout_gate_cache_config.json"
    partial_path = gate_path.with_suffix(gate_path.suffix + ".partial")
    if gate_path.exists() or manifest_path.exists() or partial_path.exists():
        raise SystemExit("refusing to overwrite an existing gate-cache asset or partial")

    positions = sequence_count * sequence_length
    gate_bytes = expected_gate_bytes(positions, layers_count, d_ple)
    check_storage(config, "pre_gate_cache", gate_bytes + 1024 * 1024 * 1024)
    token_ids = np.memmap(
        token_path, dtype="<u4", mode="r", shape=(total_sequences, sequence_length)
    )[:sequence_count]
    if int(token_ids.max()) >= int(config["token_coverage"]["vocab_size"]):
        raise RuntimeError("token stream contains an out-of-vocabulary id")

    import torch
    import transformers

    started = time.time()
    language_model = load_text_model(config["model"]["local_snapshot"], args.device)
    layers = list(language_model.layers)
    if len(layers) != layers_count:
        raise RuntimeError(
            f"decoder layer count differs: {len(layers)} != {layers_count}"
        )
    if any(layer.hidden_size_per_layer_input != d_ple for layer in layers):
        raise RuntimeError("per-layer gate width differs")

    captured: list[Any] = [None] * layers_count
    handles = []
    for layer_index, layer in enumerate(layers):
        def capture_gate(_module, _inputs, output, *, index=layer_index, act_fn=layer.act_fn):
            captured[index] = act_fn(output).detach()

        handles.append(layer.per_layer_input_gate.register_forward_hook(capture_gate))

    with partial_path.open("wb") as handle:
        handle.truncate(gate_bytes)
    gate_memmap = np.memmap(
        partial_path,
        dtype="<f2",
        mode="r+",
        shape=(layers_count, positions, d_ple),
    )
    minimum = float("inf")
    maximum = float("-inf")
    total_sum = 0.0
    total_square_sum = 0.0
    total_values = 0
    try:
        with torch.inference_mode():
            for sequence_start in range(0, sequence_count, args.batch_size):
                sequence_stop = min(sequence_count, sequence_start + args.batch_size)
                batch_numpy = np.asarray(token_ids[sequence_start:sequence_stop], dtype=np.int64)
                batch = torch.from_numpy(batch_numpy).to(device=args.device, dtype=torch.long)
                attention_mask = torch.ones_like(batch)
                captured[:] = [None] * layers_count
                output = language_model(
                    input_ids=batch,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                if any(value is None for value in captured):
                    raise RuntimeError("one or more gate hooks did not execute")
                stacked = torch.stack(captured, dim=2).to(dtype=torch.float16).cpu().numpy()
                batch_positions = (sequence_stop - sequence_start) * sequence_length
                position_start = sequence_start * sequence_length
                position_stop = position_start + batch_positions
                layer_major = np.transpose(stacked, (2, 0, 1, 3)).reshape(
                    layers_count, batch_positions, d_ple
                )
                if not np.isfinite(layer_major).all():
                    raise RuntimeError("non-finite value in activated gate cache")
                gate_memmap[:, position_start:position_stop, :] = layer_major
                values_f32 = layer_major.astype(np.float32, copy=False)
                minimum = min(minimum, float(values_f32.min()))
                maximum = max(maximum, float(values_f32.max()))
                total_sum += float(values_f32.sum(dtype=np.float64))
                total_square_sum += float(np.square(values_f32).sum(dtype=np.float64))
                total_values += int(values_f32.size)
                del output, stacked, layer_major, values_f32, attention_mask, batch
                if sequence_stop % 32 == 0 or sequence_stop == sequence_count:
                    gate_memmap.flush()
                    check_storage(config, f"gate_cache_sequences_{sequence_stop}")
                    print(
                        json.dumps({
                            "sequence_stop": sequence_stop,
                            "sequence_count": sequence_count,
                            "elapsed_seconds": time.time() - started,
                        }),
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
        gate_memmap.flush()
        del gate_memmap, language_model
        gc.collect()
        torch.cuda.empty_cache()

    if partial_path.stat().st_size != gate_bytes:
        raise RuntimeError("gate cache byte count differs")
    partial_path.replace(gate_path)
    mean = total_sum / total_values
    variance = max(0.0, total_square_sum / total_values - mean * mean)
    manifest = {
        "schema_version": "ple-broad-gate-cache-v1",
        "protocol_id": config["protocol_id"],
        "complete": complete,
        "artifact_role": "calibration_based_fitting_only_gate_cache",
        "quality_evaluation_allowed": False,
        "model_identifier": config["model"]["identifier"],
        "model_revision": config["model"]["revision"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "token_manifest": str(token_manifest_path.resolve()),
        "token_manifest_sha256": sha256_file(token_manifest_path),
        "token_file": str(token_path.resolve()),
        "token_file_sha256": sha256_file(token_path),
        "gate_file": str(gate_path.resolve()),
        "gate_file_sha256": sha256_file(gate_path),
        "gate_file_bytes": gate_path.stat().st_size,
        "gate_storage_order": "layer_position_coordinate",
        "gate_dtype": "float16_little_endian",
        "gate_shape": [layers_count, positions, d_ple],
        "n_layers": layers_count,
        "d_ple": d_ple,
        "sequence_count": sequence_count,
        "sequence_length": sequence_length,
        "positions": positions,
        "gate_activation": "gelu_pytorch_tanh",
        "gate_definition": "decoder_layer.act_fn(raw per_layer_input_gate output)",
        "runtime_token_scale": math.sqrt(d_ple) * (2.0**-0.5),
        "runtime_token_scale_formula": f"sqrt({d_ple}) * 2**-0.5",
        "statistics": {
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "finite": True,
            "value_count": total_values,
        },
        "device": args.device,
        "batch_size": args.batch_size,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    check_storage(config, "post_gate_cache")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
