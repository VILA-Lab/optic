#!/usr/bin/env python3
"""Fit frequency-tiered OPTIC on BF16 E4B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import os
import tempfile

# Directory for large intermediate files.
SCRATCH_DIR = os.path.realpath(os.environ.get("PLE_SCRATCH_DIR") or tempfile.gettempdir())

ROOT = Path(__file__).resolve().parents[1]
NAME = (
    "frequency-adaptive-variable-rate-cross-layer-latent-vector-quantization-"
    "with-int8-decoding-and-exact-tokenizer-special-rows"
)
METHOD_NAME = (
    "frequency-adaptive variable-rate cross-layer latent vector quantization "
    "with int8 decoding and exact tokenizer-special rows"
)
RECIPE = ROOT / "configs/fit/tiered_optic_e4b_bf16_recipe.json"
BROAD_RUNNER = ROOT / "scripts/build_optic_e4b_bf16.py"
CORE_BUILDER = ROOT / "builders/methods/optic.py"
LAUNCH_SCHEMA = f"{NAME}-fit-launch-v1"
RECEIPT_SCHEMA = f"{NAME}-fit-receipt-v1"
VOCAB = 262_144
LAYERS = 42
D_PLE = 256
BASE_GROUPS = 160
MAX_GROUPS = 200
EXTRA_GROUPS = MAX_GROUPS - BASE_GROUPS
GROUP_DIM = 4
LATENT_DIM = MAX_GROUPS * GROUP_DIM
CODEBOOK_SIZE = 256
HIGH_RATE_COUNT = 110_411
RAW_FP16_RESIDENT_BYTES = 70_063_104


class FitContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FitContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def select_high_rate_token_ids(
    frequency: np.ndarray, special_token_ids: tuple[int, ...], count: int = HIGH_RATE_COUNT,
) -> np.ndarray:
    require(frequency.dtype == np.dtype("<u8") and frequency.shape == (VOCAB,), "fit frequency differs")
    ordinary_mask = np.ones(VOCAB, dtype=bool)
    ordinary_mask[np.asarray(special_token_ids, dtype=np.int64)] = False
    ordinary_ids = np.arange(VOCAB, dtype=np.int64)[ordinary_mask]
    order = np.lexsort((ordinary_ids, -frequency[ordinary_ids].astype(np.int64)))
    selected = np.sort(ordinary_ids[order[:count]]).astype(np.uint32)
    require(selected.shape == (count,) and np.unique(selected).size == count, "high-rate selection differs")
    require(not np.isin(selected, np.asarray(special_token_ids, dtype=np.uint32)).any(), "special ID selected")
    return np.ascontiguousarray(selected)


def frequency_adaptive_kmeans(
    torch: Any, latents: Any, n_groups: int, group_dim: int, codebook_size: int,
    iterations: int, generator: Any, chunk: int, *, high_rate_mask: Any,
    original_kmeans: Any, base_groups: int = BASE_GROUPS,
) -> tuple[Any, Any]:
    require(n_groups == MAX_GROUPS and group_dim == GROUP_DIM, "variable-rate k-means geometry differs")
    require(codebook_size == CODEBOOK_SIZE, "variable-rate codebook size differs")
    vocab = int(latents.shape[0])
    require(vocab == int(high_rate_mask.numel()), "high-rate mask geometry differs")
    base_books, base_codes = original_kmeans(
        torch, latents[:, : base_groups * group_dim], base_groups, group_dim,
        codebook_size, iterations, generator, chunk,
    )
    device = latents.device
    high_ids = high_rate_mask.nonzero(as_tuple=False).squeeze(1)
    low_ids = (~high_rate_mask).nonzero(as_tuple=False).squeeze(1)
    require(int(high_ids.numel()) == HIGH_RATE_COUNT, "high-rate mask count differs")
    codebooks = torch.empty((n_groups, codebook_size, group_dim), dtype=torch.float32, device=device)
    codes = torch.zeros((vocab, n_groups), dtype=torch.int64, device=device)
    codebooks[:base_groups] = base_books
    codes[:, :base_groups] = base_codes
    for group in range(base_groups, n_groups):
        data = latents[:, group * group_dim : (group + 1) * group_dim].contiguous()
        high_data = data.index_select(0, high_ids)
        low_data = data.index_select(0, low_ids)
        centroids = torch.empty((codebook_size, group_dim), dtype=torch.float32, device=device)
        centroids[0] = low_data.mean(dim=0)
        init = torch.randperm(int(high_ids.numel()), generator=generator)[: codebook_size - 1].to(device)
        centroids[1:] = high_data.index_select(0, init)
        assign = torch.empty(int(high_ids.numel()), dtype=torch.int64, device=device)
        for _ in range(iterations):
            minimum = torch.empty(int(high_ids.numel()), dtype=torch.float32, device=device)
            centroid_norm = centroids.square().sum(dim=1)
            for start in range(0, int(high_ids.numel()), chunk):
                stop = min(int(high_ids.numel()), start + chunk)
                rows = high_data[start:stop]
                distance = centroid_norm[None, :] - 2.0 * (rows @ centroids.T)
                best = distance.argmin(dim=1)
                assign[start:stop] = best
                minimum[start:stop] = distance.gather(1, best[:, None]).squeeze(1) + rows.square().sum(dim=1)
            counts = torch.bincount(assign, minlength=codebook_size).to(torch.float32)
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, assign, high_data)
            counts[0] += float(low_ids.numel())
            sums[0] += low_data.sum(dim=0)
            nonempty = counts > 0
            centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
            empty = (~nonempty).nonzero(as_tuple=False).squeeze(1)
            if int(empty.numel()) > 0:
                far_rows = torch.topk(minimum, int(empty.numel())).indices
                centroids[empty] = high_data.index_select(0, far_rows)
        centroid_norm = centroids.square().sum(dim=1)
        for start in range(0, int(high_ids.numel()), chunk):
            stop = min(int(high_ids.numel()), start + chunk)
            distance = centroid_norm[None, :] - 2.0 * (high_data[start:stop] @ centroids.T)
            assign[start:stop] = distance.argmin(dim=1)
        codebooks[group] = centroids
        codes[high_ids, group] = assign
    require(bool((codes[low_ids, base_groups:] == 0).all().item()), "low-rate k-means code drift")
    return codebooks, codes


def frequency_adaptive_assignment_step(
    torch: Any, packed_device: Any, codes: Any, codebooks: Any, decoder_weight: Any,
    decoder_bias: Any, output_metric: Any, layers: int, d: int, n_groups: int,
    group_dim: int, icd_sweeps: int, ridge: float, chunk: int, *, high_rate_mask: Any,
    base_groups: int = BASE_GROUPS, z_star_storage_device: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    require(n_groups == MAX_GROUPS and group_dim == GROUP_DIM, "variable-rate ICD geometry differs")
    vocab = int(packed_device.shape[0])
    latent_dim = n_groups * group_dim
    device = packed_device.device
    # Chunk independent token rows to bound GPU temporary storage.
    working_row_chunk = min(chunk, 4096) if z_star_storage_device == "cpu" else chunk
    with torch.no_grad():
        codes[~high_rate_mask, base_groups:] = 0
        weight = decoder_weight.detach()
        bias_rows = decoder_bias.detach()
        books = codebooks.detach()
        latent_metric = torch.zeros((latent_dim, latent_dim), dtype=torch.float32, device=device)
        metric_weight = torch.empty((layers, d, latent_dim), dtype=torch.float32, device=device)
        for layer in range(layers):
            metric_weight[layer] = output_metric[layer] @ weight[layer]
            latent_metric += weight[layer].T @ metric_weight[layer]
        latent_metric = 0.5 * (latent_metric + latent_metric.T)
        ridge_value = ridge * float(latent_metric.diagonal().mean().clamp_min(1e-30))
        latent_metric += ridge_value * torch.eye(latent_dim, device=device)
        group_arange = torch.arange(n_groups, device=device)
        z_star_device = device if z_star_storage_device is None else torch.device(z_star_storage_device)
        z_star = torch.empty((vocab, latent_dim), dtype=torch.float32, device=z_star_device)
        quantization_error = torch.empty(vocab, dtype=torch.float32, device=device)
        codes_changed = 0
        for outer_start in range(0, vocab, chunk):
            outer_stop = min(vocab, outer_start + chunk)
            for start in range(outer_start, outer_stop, working_row_chunk):
                stop = min(outer_stop, start + working_row_chunk)
                x = packed_device[start:stop].to(torch.float32).view(stop - start, layers, d)
                linear = torch.einsum("bld,lde->be", x - bias_rows, metric_weight)
                z_star_chunk = torch.linalg.solve(latent_metric, linear.T).T
                z_star[start:stop].copy_(z_star_chunk.to(z_star_device))
                codes_chunk = codes[start:stop]
                active_high = high_rate_mask[start:stop]
                z_cur = books[group_arange[None, :], codes_chunk].reshape(stop - start, latent_dim)
                h = z_cur @ latent_metric - linear
                for _ in range(icd_sweeps):
                    for group in range(n_groups):
                        active = torch.ones(stop - start, dtype=torch.bool, device=device) if group < base_groups else active_high
                        if not bool(active.any().item()):
                            continue
                        cols = slice(group * group_dim, (group + 1) * group_dim)
                        block = latent_metric[cols, cols]
                        entries = books[group]
                        current = z_cur[active, cols]
                        linear_group = h[active, cols] - current @ block
                        quadratic = ((entries @ block) * entries).sum(dim=1)
                        cost = quadratic[None, :] + 2.0 * (linear_group @ entries.T)
                        new_codes = cost.argmin(dim=1)
                        delta = entries.index_select(0, new_codes) - current
                        h[active] += delta @ latent_metric[cols.start : cols.stop, :]
                        z_cur[active, cols] = entries.index_select(0, new_codes)
                        previous = codes_chunk[active, group]
                        codes_changed += int((new_codes != previous).sum().item())
                        codes_chunk[active, group] = new_codes
                        if group >= base_groups:
                            codes_chunk[~active_high, group] = 0
                diff = z_cur - z_star_chunk
                quantization_error[start:stop] = ((diff @ latent_metric) * diff).sum(dim=1)
        effective_slots = vocab * base_groups + int(high_rate_mask.sum().item()) * (n_groups - base_groups)
        stats = {
            "codes_changed": codes_changed,
            "codes_changed_fraction": codes_changed / float(effective_slots * icd_sweeps),
            "effective_reassigned_code_slots_per_sweep": effective_slots,
            "low_rate_extra_codes_forced_zero": True,
            "mean_output_metric_quantization_error": float(quantization_error.mean().item()),
            "max_output_metric_quantization_error": float(quantization_error.max().item()),
        }
    require(bool((codes[~high_rate_mask, base_groups:] == 0).all().item()), "low-rate ICD code drift")
    return z_star, quantization_error, stats


def frequency_adaptive_reinitialize_dead_codes(
    torch: Any, optimizer: Any, codes: Any, codebooks: Any, z_star: Any,
    quantization_error: Any, n_groups: int, group_dim: int, codebook_size: int,
    min_usage: int, *, high_rate_mask: Any, base_groups: int = BASE_GROUPS,
) -> int:
    total = 0
    with torch.no_grad():
        all_order = torch.argsort(quantization_error, descending=True)
        high_order = all_order[high_rate_mask.index_select(0, all_order)]
        state = optimizer.state.get(codebooks, {})
        all_cursor = 0
        high_cursor = 0
        for group in range(n_groups):
            usage = torch.bincount(codes[:, group], minlength=codebook_size)
            dead = (usage < min_usage).nonzero(as_tuple=False).squeeze(1)
            count = int(dead.numel())
            if count == 0:
                continue
            if group < base_groups:
                donors = all_order[all_cursor : all_cursor + count]
                all_cursor += count
            else:
                donors = high_order[high_cursor : high_cursor + count]
                high_cursor += count
                require(int(donors.numel()) == count, "not enough high-rate dead-code donors")
            cols = slice(group * group_dim, (group + 1) * group_dim)
            z_star_donors = donors if z_star.device == donors.device else donors.cpu()
            codebooks.data[group, dead] = z_star[z_star_donors][:, cols].to(codebooks.device)
            codes[donors, group] = dead
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][group, dead] = 0.0
            total += count
        codes[~high_rate_mask, base_groups:] = 0
    require(bool((codes[~high_rate_mask, base_groups:] == 0).all().item()), "dead-code repair changed low-rate codes")
    return total


def _load_broad_module() -> Any:
    spec = importlib.util.spec_from_file_location("frequency_adaptive_broad_inputs", BROAD_RUNNER)
    require(spec is not None and spec.loader is not None, "cannot load broad host-streaming runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_core_builder() -> Any:
    spec = importlib.util.spec_from_file_location("frequency_adaptive_core_builder", CORE_BUILDER)
    require(spec is not None and spec.loader is not None, "cannot load the latent-VQ builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_recipe(recipe: Mapping[str, Any]) -> None:
    require(recipe.get("status") == "fixed", "fit recipe status differs")
    allocation = recipe["rate_allocation"]
    require(
        (int(allocation["base_group_count"]), int(allocation["maximum_group_count"]), int(allocation["high_rate_ordinary_token_count"]))
        == (BASE_GROUPS, MAX_GROUPS, HIGH_RATE_COUNT),
        "fit recipe rate allocation differs",
    )
    schema = recipe["strict_100x_online_schema"]
    require(int(schema["persistent_resident_bytes"]) == 56_371_428, "online resident bytes differ")
    require(int(schema["strict_100x_slack_bytes"]) == 17, "strict-100x slack differs")


def run_fit(output_directory: Path, device_name: str) -> dict[str, Any]:
    import torch

    recipe = load_json(RECIPE)
    _validate_recipe(recipe)
    output = output_directory.resolve()
    require(not output.exists(), f"refusing to reuse fit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frequency_record = recipe["fit_inputs"]["fit_only_frequency"]
    frequency_path = resolve_path(frequency_record["path"])
    require(sha256_file(frequency_path) == frequency_record["sha256"], "fit frequency SHA differs")
    special_ids = tuple(int(value) for value in recipe["strict_100x_online_schema"]["special_token_ids"])
    frequency = np.fromfile(frequency_path, dtype="<u8")
    high_rate_ids = select_high_rate_token_ids(frequency, special_ids)
    high_rate_ids_sha = hashlib.sha256(high_rate_ids.astype("<u4", copy=False).tobytes()).hexdigest()
    high_rate_mask = torch.zeros(VOCAB, dtype=torch.bool, device=torch.device(device_name))
    high_rate_mask[torch.from_numpy(high_rate_ids.astype(np.int64)).to(device_name)] = True

    broad = _load_broad_module()
    builder = _load_core_builder()
    original_kmeans = builder.kmeans_per_group
    builder.kmeans_per_group = lambda torch_module, latents, n_groups, group_dim, codebook_size, iterations, generator, chunk: frequency_adaptive_kmeans(
        torch_module, latents, n_groups, group_dim, codebook_size, iterations, generator, chunk,
        high_rate_mask=high_rate_mask, original_kmeans=original_kmeans,
    )
    builder.assignment_step = lambda *args: frequency_adaptive_assignment_step(*args, high_rate_mask=high_rate_mask)
    builder.reinitialize_dead_codes = lambda *args: frequency_adaptive_reinitialize_dead_codes(*args, high_rate_mask=high_rate_mask)

    fit_inputs = recipe["fit_inputs"]
    fit_recipe = recipe["fit_recipe"]
    model = recipe["model"]
    natural_path = resolve_path(fit_inputs["natural_token_file"]["path"])
    complete_path = resolve_path(fit_inputs["complete_tokenizer_rows"]["path"])

    def load_training_inputs(torch_module: Any, _safe_open: Any, gate_cache: str, gate_manifest: str, vocab: int, layers: int, d: int):
        require(resolve_path(gate_cache) == resolve_path(fit_inputs["gate_file"]["path"]), "gate cache differs")
        require(resolve_path(gate_manifest) == resolve_path(fit_inputs["gate_manifest"]["path"]), "gate manifest differs")
        natural = np.memmap(natural_path, dtype="<u4", mode="r", shape=(int(fit_inputs["natural_token_file"]["positions"]),))
        complete = np.memmap(complete_path, dtype="<u4", mode="r", shape=(VOCAB,))
        repeated = np.repeat(np.asarray(special_ids, dtype=np.int64), int(fit_recipe["special_token_repetitions_each"]))
        tokens = torch_module.from_numpy(np.concatenate([np.asarray(natural, dtype=np.int64), np.asarray(complete, dtype=np.int64), repeated])).long().contiguous()
        require(int(tokens.numel()) == int(fit_recipe["total_training_positions"]), "training position count differs")
        gates = [
            broad.LazyLayerGate(
                torch_module, resolve_path(fit_inputs["gate_file"]["path"]), layer,
                int(fit_inputs["natural_token_file"]["positions"]), LAYERS, D_PLE,
                int(tokens.numel()), device_name,
            )
            for layer in range(LAYERS)
        ]
        return tokens, gates, float(fit_recipe["runtime_token_scale"]), {
            "training_gate_cache_file": str(resolve_path(fit_inputs["gate_file"]["path"])),
            "training_gate_cache_sha256": fit_inputs["gate_file"]["sha256"],
            "training_gate_cache_config_file": str(resolve_path(fit_inputs["gate_manifest"]["path"])),
            "training_gate_cache_config_sha256": fit_inputs["gate_manifest"]["sha256"],
            "training_gate_cache_positions": int(tokens.numel()),
            "training_natural_context_positions": int(fit_inputs["natural_token_file"]["positions"]),
            "training_model_only_row_positions": VOCAB,
            "training_special_token_positions": int(fit_recipe["special_token_positions"]),
            "training_input_mode": fit_recipe["input_mode"],
            "training_forward_path_runtime_token_scale": float(fit_recipe["runtime_token_scale"]),
            "fit_config": str(RECIPE.resolve()),
            "fit_config_sha256": sha256_file(RECIPE),
            "frequency_adaptive_variable_rate": True,
            "high_rate_token_count": HIGH_RATE_COUNT,
            "high_rate_token_ids_sha256": high_rate_ids_sha,
            "low_rate_extra_code_constraint": "zero",
        }

    builder.build_exact_gate_cache_forward_path_training_inputs = load_training_inputs
    output.mkdir()
    sys.argv = [
        str(CORE_BUILDER), "--it-weights", model["weights"], "--num-layers", str(LAYERS),
        "--d-ple", str(D_PLE), "--latent-mode", "discrete", "--latent-dim", str(LATENT_DIM),
        "--n-groups", str(MAX_GROUPS), "--codebook-size", str(CODEBOOK_SIZE),
        "--group-dim-allocation", "contiguous", "--alternations", str(fit_recipe["alternations"]),
        "--steps-per-alternation", str(fit_recipe["steps_per_alternation"]),
        "--assignment-icd-sweeps", str(fit_recipe["assignment_icd_sweeps"]),
        "--assignment-ridge", str(fit_recipe["assignment_ridge"]),
        "--assignment-chunk-size", str(fit_recipe["assignment_chunk_size"]),
        "--dead-code-min-usage", str(fit_recipe["dead_code_min_usage"]),
        "--kmeans-iterations", str(fit_recipe["kmeans_iterations"]),
        "--metric-eigenvalue-floor", str(fit_recipe["metric_eigenvalue_floor"]),
        "--lr", str(fit_recipe["learning_rate"]), "--batch-size", str(fit_recipe["batch_size"]),
        "--training-gate-cache", fit_inputs["gate_file"]["path"],
        "--training-gate-cache-config", fit_inputs["gate_manifest"]["path"],
        "--heldout-eval-size", str(fit_recipe["heldout_eval_size"]),
        "--heldout-eval-seed", str(fit_recipe["heldout_eval_seed"]),
        "--heldout-eval-interval-steps", str(fit_recipe["heldout_eval_interval_steps"]),
        "--resident-roundtrip-atol", str(fit_recipe["resident_roundtrip_atol"]),
        "--device", device_name, "--seed", str(fit_recipe["seed"]),
        "--save-recon", str(output / "dense_ple_reconstruction.safetensors"),
        "--save-compressed-ple-npz", str(output / "full_code_fit_artifact.npz"),
        "--output", str(output / "fit_metrics.json"),
    ]
    builder.main()
    artifact = output / "full_code_fit_artifact.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        packed_codes = np.ascontiguousarray(archive["token_codes_packed_uint8"])
        metadata = json.loads(str(archive["metadata"].item()))
    codes = packed_codes.reshape(VOCAB, MAX_GROUPS)
    high_lookup = np.zeros(VOCAB, dtype=bool)
    high_lookup[high_rate_ids.astype(np.int64)] = True
    require(bool((codes[~high_lookup, BASE_GROUPS:] == 0).all()), "saved low-rate extra codes are not zero")
    metrics = load_json(output / "fit_metrics.json")
    require(int(metrics["storage_bytes"]) == RAW_FP16_RESIDENT_BYTES, "raw fit byte geometry differs")
    require(metadata.get("fit_config_sha256") == sha256_file(RECIPE), "raw fit recipe binding differs")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "fit_complete",
        "method_name": METHOD_NAME,
        "fit_recipe_sha256": sha256_file(RECIPE),
        "fit_runner_sha256": sha256_file(Path(__file__).resolve()),
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "metrics_path": str(output / "fit_metrics.json"),
        "metrics_sha256": sha256_file(output / "fit_metrics.json"),
        "dense_reconstruction_path": str(output / "dense_ple_reconstruction.safetensors"),
        "dense_reconstruction_storage": "ephemeral_dev_shm_only",
        "high_rate_token_ids_sha256": high_rate_ids_sha,
        "high_rate_token_count": HIGH_RATE_COUNT,
        "base_group_count": BASE_GROUPS,
        "maximum_group_count": MAX_GROUPS,
        "low_rate_extra_codes_all_zero": True,
        "raw_full_code_fit_resident_bytes": RAW_FP16_RESIDENT_BYTES,
        "benchmark_inputs_used": False,
        "l1_or_l2_inputs_used": False,
        "quality_evaluation_run": False,
    }
    receipt_path = output / "fit_receipt.json"
    require(not receipt_path.exists(), "refusing to overwrite fit receipt")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run_fit(args.output_dir, args.device), indent=2))

if __name__ == "__main__":
    main()
