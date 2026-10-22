#!/usr/bin/env python3
"""OPTIC: shared token codes with layer-specific affine decoders for the full PLE table.

Representation: every token t owns a latent z_t of dimension Z built from
n_groups concatenated sub-vectors, each taken from that group's codebook of
codebook_size entries (product quantization of the latent). Layer l's PLE row is
decoded as `E_hat[l, t] = decoder_weight[l] @ z_t + decoder_bias[l]`.
Resident state = packed codes + codebooks + per-layer linear decoder.

With `--latent-mode continuous`, z_t is stored directly as an fp16 table with
the same per-layer linear decoder and the same training objective, without
vector quantization.

Training objective: gate-cache forward-path PLE MSE against the original
checkpoint rows through the per-layer-input gates and per-layer projection
weights (`mean || W_d_l((g_l * (E_hat - E)) * runtime_token_scale) ||^2`).

Optimization (discrete mode) alternates:
  (a) exact per-token code re-assignment by iterated conditional decoding (ICD)
      under the output-weighted latent metric
      `A_z = sum_l decoder_weight[l]^T G_l decoder_weight[l]` with
      `G_l = (W_d_l^T W_d_l) (elementwise*) E_positions[g_l g_l^T]`
      (position-averaged pullback of the forward path onto the latent);
  (b) Adam updates of codebooks + decoder (+ centers) with codes fixed
      (gradients reach codebook entries directly through the gather; no
      straight-through estimator is involved).
An optional straight-through-estimator warm-start phase with EMA codebook
updates is available behind `--ste-warmup-steps`.

Initialization: output-weighted whitened PCA of all token vectors (each token's
concatenated layer rows, whitened by blockdiag G_l^{1/2}; PCA; decoder =
G^{-1/2} V_Z), eigenvalue-balanced allocation of latent dimensions to groups,
then per-group k-means on the init latent (Euclidean there equals the output
metric because A_z = I at init). All randomness derives from --seed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

BASELINE_BUILDERS = Path(__file__).resolve().parents[1] / "baselines"
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_BUILDERS))
sys.path.insert(0, str(THIS_DIR))

from baseline_common import RECON_KEY, load_ple_tensor  # noqa: E402
from fitting_common import (  # noqa: E402
    benchmark_id_for_metrics_output,
    build_exact_gate_cache_forward_path_training_inputs,
    load_projection_weights,
    set_learning_rate,
)

METHOD_ID = "optic"
FORMAT_ID = "optic_npz"
LATENT_MODES = ("discrete", "continuous")
GROUP_DIM_ALLOCATION_MODES = ("eigenvalue_balanced", "contiguous")
OPTIMIZER_MODES = ("adamw",)
LR_SCHEDULE_MODES = ("constant", "cosine")
TRAINING_OBJECTIVE_MODE = "exact_gate_cache_forward_path_ple_mse"
TRAINING_LOSS_TRACE_INTERVAL_STEPS = 100


# --------------------------------------------------------------------------- #
# Byte accounting
# --------------------------------------------------------------------------- #
def packed_code_bytes(vocab: int, n_groups: int, code_bits: int) -> int:
    return (vocab * n_groups * code_bits + 7) // 8


def storage_components(
    vocab: int,
    layers: int,
    d: int,
    latent_mode: str,
    latent_dim: int,
    n_groups: int,
    codebook_size: int,
) -> dict[str, int]:
    components: dict[str, int] = {}
    if latent_mode == "discrete":
        code_bits = int(math.log2(codebook_size))
        group_dim = latent_dim // n_groups
        components["token_code_bytes"] = packed_code_bytes(vocab, n_groups, code_bits)
        components["codebook_bytes"] = n_groups * codebook_size * group_dim * 2
    elif latent_mode == "continuous":
        components["latent_table_bytes"] = vocab * latent_dim * 2
    else:
        raise ValueError(f"unknown latent mode: {latent_mode}")
    components["decoder_weight_bytes"] = layers * d * latent_dim * 2
    components["decoder_bias_bytes"] = layers * d * 2
    components["storage_bytes"] = sum(components.values())
    components["raw_bytes"] = vocab * layers * d * 2
    return components


# --------------------------------------------------------------------------- #
# Packed code encoding (deterministic, allow_pickle-free)
# --------------------------------------------------------------------------- #
def pack_codes(np: Any, codes, code_bits: int):
    codes = np.ascontiguousarray(codes, dtype=np.int64)
    if codes.min() < 0 or codes.max() >= (1 << code_bits):
        raise SystemExit("codes out of range for the declared code bit width")
    shifts = np.arange(code_bits - 1, -1, -1, dtype=np.int64)
    bits = ((codes.reshape(-1, 1) >> shifts) & 1).astype(np.uint8).reshape(-1)
    packed = np.packbits(bits)
    expected = packed_code_bytes(int(codes.shape[0]), int(codes.shape[1]), code_bits)
    if int(packed.nbytes) != expected:
        raise SystemExit("packed code byte count does not match closed-form accounting")
    return packed


def unpack_codes(np: Any, packed, vocab: int, n_groups: int, code_bits: int):
    total_bits = vocab * n_groups * code_bits
    bits = np.unpackbits(packed, count=total_bits).reshape(vocab, n_groups, code_bits)
    weights = (1 << np.arange(code_bits - 1, -1, -1, dtype=np.int64))
    return (bits.astype(np.int64) * weights).sum(axis=2)


# --------------------------------------------------------------------------- #
# Output-weighted metric and whitened-PCA initialization
# --------------------------------------------------------------------------- #
def compute_gate_second_moments(torch: Any, gate_by_layer_device, pool_sample_ids, layers: int, d: int, chunk: int):
    """G2[l] = mean over pool positions of g_l g_l^T (fp32, [layers, d, d])."""
    device = gate_by_layer_device[0].device
    second_moment = torch.zeros((layers, d, d), dtype=torch.float32, device=device)
    n_positions = int(pool_sample_ids.numel())
    for layer in range(layers):
        for start in range(0, n_positions, chunk):
            stop = min(n_positions, start + chunk)
            gate = gate_by_layer_device[layer].index_select(0, pool_sample_ids[start:stop]).to(torch.float32)
            second_moment[layer] += gate.T @ gate
            del gate
    second_moment /= float(n_positions)
    return second_moment


def build_output_metric(torch: Any, projection_weights, gate_second_moment, layers: int):
    """G[l] = (W_d_l^T W_d_l) * E[g g^T] elementwise; exact position-averaged pullback."""
    metric = torch.empty_like(gate_second_moment)
    for layer in range(layers):
        proj_gram = projection_weights[layer].T @ projection_weights[layer]
        metric[layer] = proj_gram * gate_second_moment[layer]
        metric[layer] = 0.5 * (metric[layer] + metric[layer].T)
    return metric


def metric_sqrt_and_invsqrt(torch: Any, metric, eigenvalue_floor: float):
    layers = int(metric.shape[0])
    sqrt_metric = torch.empty_like(metric)
    invsqrt_metric = torch.empty_like(metric)
    for layer in range(layers):
        evals, evecs = torch.linalg.eigh(metric[layer])
        floor = float(evals.max().clamp_min(1e-30)) * eigenvalue_floor
        evals = evals.clamp_min(floor)
        sqrt_metric[layer] = (evecs * evals.sqrt()) @ evecs.T
        invsqrt_metric[layer] = (evecs * evals.rsqrt()) @ evecs.T
    return sqrt_metric, invsqrt_metric


def token_vector_mean_and_covariance(torch: Any, packed_device, chunk: int):
    vocab = int(packed_device.shape[0])
    token_vector_dim = int(packed_device.shape[1])
    device = packed_device.device
    mean_accum = torch.zeros(token_vector_dim, dtype=torch.float64, device=device)
    gram = torch.zeros((token_vector_dim, token_vector_dim), dtype=torch.float32, device=device)
    for start in range(0, vocab, chunk):
        stop = min(vocab, start + chunk)
        x = packed_device[start:stop].to(torch.float32)
        mean_accum += x.sum(dim=0).double()
        gram += x.T @ x
        del x
    mean = (mean_accum / vocab).float()
    covariance = gram / vocab - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    del gram
    return mean, covariance


def output_weighted_pca_basis(torch: Any, covariance, sqrt_metric, layers: int, d: int, latent_dim: int):
    """Top-latent_dim eigenpairs of blockdiag(G^1/2) C blockdiag(G^1/2)."""
    whitened = torch.empty_like(covariance)
    for layer in range(layers):
        rows = slice(layer * d, (layer + 1) * d)
        whitened[rows, :] = sqrt_metric[layer] @ covariance[rows, :]
    for layer in range(layers):
        cols = slice(layer * d, (layer + 1) * d)
        whitened[:, cols] = whitened[:, cols] @ sqrt_metric[layer]
    whitened = 0.5 * (whitened + whitened.T)
    evals, evecs = torch.linalg.eigh(whitened)
    del whitened
    evals = evals.flip(0).clamp_min(0.0)
    evecs = evecs.flip(1)
    return evals[:latent_dim].contiguous(), evecs[:, :latent_dim].contiguous()


def build_initial_latents(torch: Any, packed_device, mean, sqrt_metric, basis, layers: int, d: int, chunk: int):
    vocab = int(packed_device.shape[0])
    latent_dim = int(basis.shape[1])
    device = packed_device.device
    latents = torch.empty((vocab, latent_dim), dtype=torch.float32, device=device)
    mean_rows = mean.view(layers, d)
    for start in range(0, vocab, chunk):
        stop = min(vocab, start + chunk)
        x = packed_device[start:stop].to(torch.float32).view(stop - start, layers, d)
        whitened = torch.einsum("bld,lde->ble", x - mean_rows, sqrt_metric)
        latents[start:stop] = whitened.reshape(stop - start, layers * d) @ basis
        del x, whitened
    return latents


def eigenvalue_balanced_allocation(eigenvalues, n_groups: int, group_dim: int) -> list[int]:
    """OPQ-style allocation: balance per-group sums of log-eigenvalue.

    Returns a permutation of latent dimensions such that group m owns the
    contiguous slice [m*group_dim, (m+1)*group_dim) after permutation. The
    permutation is absorbed into the decoder columns, so it changes neither the
    representation family nor the byte accounting.
    """
    group_log_sums = [0.0] * n_groups
    group_slots: list[list[int]] = [[] for _ in range(n_groups)]
    for dim in range(len(eigenvalues)):
        open_groups = [g for g in range(n_groups) if len(group_slots[g]) < group_dim]
        target = min(open_groups, key=lambda g: (group_log_sums[g], g))
        group_slots[target].append(dim)
        group_log_sums[target] += math.log(max(float(eigenvalues[dim]), 1e-12))
    return [dim for slots in group_slots for dim in slots]


def kmeans_per_group(torch: Any, latents, n_groups: int, group_dim: int, codebook_size: int, iterations: int, generator, chunk: int):
    vocab = int(latents.shape[0])
    device = latents.device
    codebooks = torch.empty((n_groups, codebook_size, group_dim), dtype=torch.float32, device=device)
    codes = torch.empty((vocab, n_groups), dtype=torch.int64, device=device)
    for group in range(n_groups):
        data = latents[:, group * group_dim : (group + 1) * group_dim].contiguous()
        init_rows = torch.randperm(vocab, generator=generator)[:codebook_size].to(device)
        centroids = data.index_select(0, init_rows).clone()
        assign = torch.empty(vocab, dtype=torch.int64, device=device)
        for _ in range(iterations):
            min_dist = torch.empty(vocab, dtype=torch.float32, device=device)
            cent_sq = centroids.square().sum(dim=1)
            for start in range(0, vocab, chunk):
                stop = min(vocab, start + chunk)
                dist = cent_sq[None, :] - 2.0 * (data[start:stop] @ centroids.T)
                best = dist.argmin(dim=1)
                assign[start:stop] = best
                min_dist[start:stop] = dist.gather(1, best[:, None]).squeeze(1) + data[start:stop].square().sum(dim=1)
                del dist, best
            counts = torch.bincount(assign, minlength=codebook_size).to(torch.float32)
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, assign, data)
            nonempty = counts > 0
            centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
            empty = (~nonempty).nonzero(as_tuple=False).squeeze(1)
            if int(empty.numel()) > 0:
                far_rows = torch.topk(min_dist, int(empty.numel())).indices
                centroids[empty] = data.index_select(0, far_rows)
            del min_dist, counts, sums
        cent_sq = centroids.square().sum(dim=1)
        for start in range(0, vocab, chunk):
            stop = min(vocab, start + chunk)
            dist = cent_sq[None, :] - 2.0 * (data[start:stop] @ centroids.T)
            assign[start:stop] = dist.argmin(dim=1)
            del dist
        codebooks[group] = centroids
        codes[:, group] = assign
        del data, centroids, assign
    return codebooks, codes


# --------------------------------------------------------------------------- #
# Exact code re-assignment (ICD under the output-weighted latent metric)
# --------------------------------------------------------------------------- #
def assignment_step(
    torch: Any,
    packed_device,
    codes,
    codebooks,
    decoder_weight,
    decoder_bias,
    output_metric,
    layers: int,
    d: int,
    n_groups: int,
    group_dim: int,
    icd_sweeps: int,
    ridge: float,
    chunk: int,
):
    """Re-assign every token's codes to minimize (z - z*)^T A_z (z - z*).

    A_z = sum_l W_l^T G_l W_l pulls the position-averaged forward-path output
    metric back onto the latent. Per-token code choice is independent of token
    frequency because each token's codes only affect its own rows, so the
    uniform full-vocabulary sweep is exact for the population metric.
    Returns (z_star, quantization_error, stats).
    """
    vocab = int(packed_device.shape[0])
    latent_dim = n_groups * group_dim
    device = packed_device.device
    with torch.no_grad():
        weight = decoder_weight.detach()
        bias_rows = decoder_bias.detach()
        codebooks_det = codebooks.detach()
        latent_metric = torch.zeros((latent_dim, latent_dim), dtype=torch.float32, device=device)
        metric_weight = torch.empty((layers, d, latent_dim), dtype=torch.float32, device=device)
        for layer in range(layers):
            metric_weight[layer] = output_metric[layer] @ weight[layer]
            latent_metric += weight[layer].T @ metric_weight[layer]
        latent_metric = 0.5 * (latent_metric + latent_metric.T)
        ridge_value = ridge * float(latent_metric.diagonal().mean().clamp_min(1e-30))
        latent_metric += ridge_value * torch.eye(latent_dim, device=device)
        group_arange = torch.arange(n_groups, device=device)
        z_star = torch.empty((vocab, latent_dim), dtype=torch.float32, device=device)
        quantization_error = torch.empty(vocab, dtype=torch.float32, device=device)
        codes_changed = 0
        for start in range(0, vocab, chunk):
            stop = min(vocab, start + chunk)
            x = packed_device[start:stop].to(torch.float32).view(stop - start, layers, d)
            linear = torch.einsum("bld,lde->be", x - bias_rows, metric_weight)
            z_star[start:stop] = torch.linalg.solve(latent_metric, linear.T).T
            codes_chunk = codes[start:stop]
            z_cur = codebooks_det[group_arange[None, :], codes_chunk].reshape(stop - start, latent_dim)
            h = z_cur @ latent_metric - linear
            for _ in range(icd_sweeps):
                for group in range(n_groups):
                    cols = slice(group * group_dim, (group + 1) * group_dim)
                    block = latent_metric[cols, cols]
                    entries = codebooks_det[group]
                    quad = ((entries @ block) * entries).sum(dim=1)
                    lin = h[:, cols] - z_cur[:, cols] @ block
                    cost = quad[None, :] + 2.0 * (lin @ entries.T)
                    new_codes = cost.argmin(dim=1)
                    delta = entries.index_select(0, new_codes) - z_cur[:, cols]
                    h += delta @ latent_metric[cols.start : cols.stop, :]
                    z_cur[:, cols] = entries.index_select(0, new_codes)
                    codes_changed += int((new_codes != codes_chunk[:, group]).sum().item())
                    codes_chunk[:, group] = new_codes
                    del quad, lin, cost, new_codes, delta
            diff = z_cur - z_star[start:stop]
            quantization_error[start:stop] = ((diff @ latent_metric) * diff).sum(dim=1)
            del x, linear, codes_chunk, z_cur, h, diff
        stats = {
            "codes_changed": codes_changed,
            "codes_changed_fraction": codes_changed / float(vocab * n_groups * icd_sweeps),
            "mean_output_metric_quantization_error": float(quantization_error.mean().item()),
            "max_output_metric_quantization_error": float(quantization_error.max().item()),
        }
    return z_star, quantization_error, stats


def reinitialize_dead_codes(
    torch: Any,
    optimizer,
    codes,
    codebooks,
    z_star,
    quantization_error,
    n_groups: int,
    group_dim: int,
    codebook_size: int,
    min_usage: int,
):
    """Move unused codebook entries onto the output-metric-optimal latents of the
    worst-quantized tokens and zero their Adam moments."""
    total_reinitialized = 0
    with torch.no_grad():
        error_order = torch.argsort(quantization_error, descending=True)
        state = optimizer.state.get(codebooks, {})
        cursor = 0
        for group in range(n_groups):
            usage = torch.bincount(codes[:, group], minlength=codebook_size)
            dead = (usage < min_usage).nonzero(as_tuple=False).squeeze(1)
            n_dead = int(dead.numel())
            if n_dead == 0:
                continue
            donor_tokens = error_order[cursor : cursor + n_dead]
            cursor += n_dead
            cols = slice(group * group_dim, (group + 1) * group_dim)
            codebooks.data[group, dead] = z_star[donor_tokens][:, cols]
            codes[donor_tokens, group] = dead
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][group, dead] = 0.0
            total_reinitialized += n_dead
    return total_reinitialized


# --------------------------------------------------------------------------- #
# Resident payload save / reload / decode
# --------------------------------------------------------------------------- #
def load_npz_metadata_object(np: Any, npz) -> dict[str, Any]:
    if "metadata" not in npz.files:
        raise SystemExit("compressed PLE representation file missing scalar JSON metadata")
    raw = npz["metadata"]
    if raw.shape != ():
        raise SystemExit("compressed PLE representation metadata must be a scalar JSON string")
    metadata = json.loads(str(raw.item()))
    if not isinstance(metadata, dict):
        raise SystemExit("compressed PLE representation metadata JSON must be an object")
    return metadata


def require_npz_array(np: Any, npz, name: str, dtype, shape: tuple[int, ...]):
    if name not in npz.files:
        raise SystemExit(f"compressed PLE representation file missing required array {name!r}")
    value = npz[name]
    if value.dtype != np.dtype(dtype):
        raise SystemExit(f"resident array {name!r} dtype {value.dtype} != {np.dtype(dtype)}")
    if tuple(value.shape) != shape:
        raise SystemExit(f"resident array {name!r} shape {tuple(value.shape)} != {shape}")
    return value.copy()


def load_resident_payload(
    np: Any,
    compressed_path: Path,
    vocab: int,
    layers: int,
    d: int,
    latent_mode: str,
    latent_dim: int,
    n_groups: int,
    codebook_size: int,
    expected_storage_bytes: int,
) -> dict[str, Any]:
    expected_files = {"metadata", "decoder_weight_fp16", "decoder_bias_fp16"}
    if latent_mode == "discrete":
        expected_files.update({"token_codes_packed_uint8", "codebooks_fp16"})
    else:
        expected_files.add("latent_table_fp16")
    with np.load(str(compressed_path), allow_pickle=False) as npz:
        if set(npz.files) != expected_files:
            raise SystemExit(f"resident file entries {sorted(npz.files)} != {sorted(expected_files)}")
        metadata = load_npz_metadata_object(np, npz)
        if metadata.get("compressed_ple_format_id") != FORMAT_ID:
            raise SystemExit("resident metadata compressed_ple_format_id does not match this builder")
        if metadata.get("method") != METHOD_ID or metadata.get("kind") != METHOD_ID:
            raise SystemExit("resident metadata method/kind does not match this builder")
        if metadata.get("ple_shape") != [vocab, layers * d]:
            raise SystemExit("resident metadata ple_shape does not match expected full PLE shape")
        if metadata.get("latent_mode") != latent_mode or int(metadata.get("latent_dim", -1)) != latent_dim:
            raise SystemExit("resident metadata latent mode/dim does not match launch arguments")
        payload: dict[str, Any] = {"metadata": metadata}
        payload["decoder_weight_fp16"] = require_npz_array(
            np, npz, "decoder_weight_fp16", np.float16, (layers, d, latent_dim)
        )
        payload["decoder_bias_fp16"] = require_npz_array(np, npz, "decoder_bias_fp16", np.float16, (layers, d))
        if latent_mode == "discrete":
            code_bits = int(math.log2(codebook_size))
            group_dim = latent_dim // n_groups
            payload["token_codes_packed_uint8"] = require_npz_array(
                np, npz, "token_codes_packed_uint8", np.uint8, (packed_code_bytes(vocab, n_groups, code_bits),)
            )
            payload["codebooks_fp16"] = require_npz_array(
                np, npz, "codebooks_fp16", np.float16, (n_groups, codebook_size, group_dim)
            )
        else:
            payload["latent_table_fp16"] = require_npz_array(
                np, npz, "latent_table_fp16", np.float16, (vocab, latent_dim)
            )
    resident_bytes = sum(int(value.nbytes) for key, value in payload.items() if key != "metadata")
    if resident_bytes != expected_storage_bytes:
        raise SystemExit(
            f"resident tensor bytes {resident_bytes} != declared storage bytes {expected_storage_bytes}"
        )
    return payload


def latents_from_payload(torch: Any, np: Any, payload: dict[str, Any], vocab: int, latent_mode: str, latent_dim: int, n_groups: int, codebook_size: int, device):
    if latent_mode == "discrete":
        code_bits = int(math.log2(codebook_size))
        codes_np = unpack_codes(np, payload["token_codes_packed_uint8"], vocab, n_groups, code_bits)
        codes = torch.from_numpy(codes_np).to(device)
        codebooks = torch.from_numpy(payload["codebooks_fp16"]).to(device=device, dtype=torch.float32)
        group_arange = torch.arange(n_groups, device=device)
        return lambda start, stop: codebooks[group_arange[None, :], codes[start:stop]].reshape(
            stop - start, latent_dim
        )
    latent_table = torch.from_numpy(payload["latent_table_fp16"]).to(device=device, dtype=torch.float32)
    return lambda start, stop: latent_table[start:stop]


def decode_payload_chunk(torch: Any, latent_chunk_fn, decoder_weight, decoder_bias, layer: int, start: int, stop: int):
    z = latent_chunk_fn(start, stop)
    return z @ decoder_weight[layer].T + decoder_bias[layer]


def materialize_reconstruction(
    torch: Any,
    np: Any,
    payload: dict[str, Any],
    recon_path: Path,
    vocab: int,
    layers: int,
    d: int,
    latent_mode: str,
    latent_dim: int,
    n_groups: int,
    codebook_size: int,
    device,
    chunk: int,
) -> None:
    from safetensors.torch import save_file

    latent_chunk_fn = latents_from_payload(
        torch, np, payload, vocab, latent_mode, latent_dim, n_groups, codebook_size, device
    )
    decoder_weight = torch.from_numpy(payload["decoder_weight_fp16"]).to(device=device, dtype=torch.float32)
    decoder_bias = torch.from_numpy(payload["decoder_bias_fp16"]).to(device=device, dtype=torch.float32)
    recon = torch.empty((vocab, layers * d), dtype=torch.bfloat16)
    for layer in range(layers):
        for start in range(0, vocab, chunk):
            stop = min(vocab, start + chunk)
            decoded = decode_payload_chunk(torch, latent_chunk_fn, decoder_weight, decoder_bias, layer, start, stop)
            recon[start:stop, layer * d : (layer + 1) * d] = decoded.to(torch.bfloat16).cpu()
            del decoded
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({RECON_KEY: recon.contiguous()}, str(recon_path))


def verify_resident_roundtrip(
    torch: Any,
    np: Any,
    safe_open: Any,
    compressed_path: Path,
    recon_path: Path,
    vocab: int,
    layers: int,
    d: int,
    latent_mode: str,
    latent_dim: int,
    n_groups: int,
    codebook_size: int,
    expected_storage_bytes: int,
    device,
    chunk: int,
    atol: float,
) -> float:
    payload = load_resident_payload(
        np, compressed_path, vocab, layers, d, latent_mode, latent_dim, n_groups, codebook_size, expected_storage_bytes
    )
    latent_chunk_fn = latents_from_payload(
        torch, np, payload, vocab, latent_mode, latent_dim, n_groups, codebook_size, device
    )
    decoder_weight = torch.from_numpy(payload["decoder_weight_fp16"]).to(device=device, dtype=torch.float32)
    decoder_bias = torch.from_numpy(payload["decoder_bias_fp16"]).to(device=device, dtype=torch.float32)
    max_abs_diff = 0.0
    with safe_open(str(recon_path), framework="pt", device="cpu") as recon_file:
        if RECON_KEY not in set(recon_file.keys()):
            raise SystemExit(f"missing dense reconstruction tensor key in {recon_path}: {RECON_KEY}")
        recon_slice = recon_file.get_slice(RECON_KEY)
        if list(recon_slice.get_shape()) != [vocab, layers * d]:
            raise SystemExit("resident roundtrip recon shape does not match expected full PLE shape")
        for layer in range(layers):
            for start in range(0, vocab, chunk):
                stop = min(vocab, start + chunk)
                decoded = decode_payload_chunk(
                    torch, latent_chunk_fn, decoder_weight, decoder_bias, layer, start, stop
                ).to(torch.bfloat16).to(torch.float32)
                saved = recon_slice[start:stop, layer * d : (layer + 1) * d].to(device).to(torch.float32)
                if not bool(torch.isfinite(decoded).all().item()):
                    raise SystemExit("resident file reload/decode produced non-finite values")
                if not bool(torch.isfinite(saved).all().item()):
                    raise SystemExit("saved BF16 dense PLE reconstruction contains non-finite values")
                diff = float((decoded - saved).abs().max().item())
                max_abs_diff = max(max_abs_diff, diff)
                if max_abs_diff > atol:
                    raise SystemExit(
                        f"resident file reload/decode max_abs_diff {max_abs_diff} exceeds tolerance {atol}"
                    )
                del decoded, saved
    return max_abs_diff


def plain_nmse_recon_file_vs_device_original(
    torch: Any,
    safe_open: Any,
    recon_path: Path,
    packed_original_device,
    layers: int,
    d: int,
    chunk: int,
) -> float:
    """Row-space NMSE of the saved recon file against the device-resident
    original checkpoint PLE tensor (bf16, loaded unchanged from --it-weights).
    GPU-resident equivalent of the reference builders' CPU NMSE pass."""
    device = packed_original_device.device
    vocab = int(packed_original_device.shape[0])
    err = torch.zeros((), dtype=torch.float64, device=device)
    norm = torch.zeros((), dtype=torch.float64, device=device)
    with safe_open(str(recon_path), framework="pt", device="cpu") as recon_file:
        recon_slice = recon_file.get_slice(RECON_KEY)
        for layer in range(layers):
            for start in range(0, vocab, chunk):
                stop = min(vocab, start + chunk)
                recon = recon_slice[start:stop, layer * d : (layer + 1) * d].to(device).to(torch.float32)
                original = packed_original_device[start:stop, layer * d : (layer + 1) * d].to(torch.float32)
                err += (recon - original).double().square().sum()
                norm += original.double().square().sum()
                del recon, original
    return float(err.item()) / max(float(norm.item()), 1e-30)


def gate_cache_forward_path_nmse(
    torch: Any,
    safe_open: Any,
    recon_path: Path,
    packed_original_device,
    token_ids_device,
    gate_by_layer_device,
    projection_weights,
    runtime_token_scale: float,
    layers: int,
    d: int,
    position_chunk: int,
) -> float:
    """NMSE of the exact forward path over every locked-cache position and layer."""
    device = packed_original_device.device
    n_positions = int(token_ids_device.numel())
    scale = float(runtime_token_scale)
    err = 0.0
    sig = 0.0
    with safe_open(str(recon_path), framework="pt", device="cpu") as recon_file:
        recon_slice = recon_file.get_slice(RECON_KEY)
        for layer in range(layers):
            recon_layer = recon_slice[:, layer * d : (layer + 1) * d].to(device).to(torch.float32)
            original_layer = packed_original_device[:, layer * d : (layer + 1) * d]
            for start in range(0, n_positions, position_chunk):
                stop = min(n_positions, start + position_chunk)
                tokens = token_ids_device[start:stop]
                gate = gate_by_layer_device[layer][start:stop].to(torch.float32)
                original_rows = original_layer.index_select(0, tokens).to(torch.float32)
                diff_rows = recon_layer.index_select(0, tokens) - original_rows
                err += float(
                    (((gate * diff_rows) * scale) @ projection_weights[layer].T).square().sum().item()
                )
                sig += float(
                    (((gate * original_rows) * scale) @ projection_weights[layer].T).square().sum().item()
                )
                del tokens, gate, original_rows, diff_rows
            del recon_layer
    return err / max(sig, 1e-30)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--it-weights", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--d-ple", type=int, default=256)
    parser.add_argument("--latent-mode", choices=LATENT_MODES, default="discrete")
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--n-groups", type=int, default=0)
    parser.add_argument("--codebook-size", type=int, default=0)
    parser.add_argument("--group-dim-allocation", choices=GROUP_DIM_ALLOCATION_MODES, default="eigenvalue_balanced")
    parser.add_argument("--alternations", type=int, required=True)
    parser.add_argument("--steps-per-alternation", type=int, required=True)
    parser.add_argument("--assignment-icd-sweeps", type=int, default=2)
    parser.add_argument("--assignment-ridge", type=float, default=1e-8)
    parser.add_argument("--assignment-chunk-size", type=int, default=65536)
    parser.add_argument("--dead-code-min-usage", type=int, default=1)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--ste-warmup-steps", type=int, default=0)
    parser.add_argument("--ste-commitment-beta", type=float, default=0.25)
    parser.add_argument("--ste-ema-gamma", type=float, default=0.99)
    parser.add_argument("--ste-ema-eps", type=float, default=1e-5)
    parser.add_argument("--metric-eigenvalue-floor", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=OPTIMIZER_MODES, default="adamw")
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay applied to the decoder weight/bias only; codebook entries and the continuous latent table always train without weight decay",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULE_MODES, default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--training-gate-cache", required=True)
    parser.add_argument("--training-gate-cache-config", required=True)
    parser.add_argument("--heldout-eval-size", type=int, default=0)
    parser.add_argument("--heldout-eval-seed", type=int, default=1)
    parser.add_argument("--heldout-eval-interval-steps", type=int, default=0)
    parser.add_argument("--eval-chunk-size", type=int, default=8192)
    parser.add_argument("--init-covariance-chunk-size", type=int, default=4096)
    parser.add_argument("--gate-moment-chunk-size", type=int, default=16384)
    parser.add_argument("--forward-path-eval-position-chunk-size", type=int, default=16384)
    parser.add_argument("--resident-roundtrip-atol", type=float, default=0.0)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-recon", required=True)
    parser.add_argument("--save-compressed-ple-npz", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_layers <= 0 or args.d_ple != 256:
        raise SystemExit("this builder supports positive Gemma PLE layer counts with d=256")
    if args.latent_dim <= 0:
        raise SystemExit("--latent-dim must be positive")
    # Non-positive chunk sizes would leave the materialize and verify loops
    # empty and save an uninitialized reconstruction.
    for flag_name in (
        "assignment_chunk_size",
        "eval_chunk_size",
        "init_covariance_chunk_size",
        "gate_moment_chunk_size",
        "forward_path_eval_position_chunk_size",
    ):
        if getattr(args, flag_name) <= 0:
            raise SystemExit(f"--{flag_name.replace('_', '-')} must be positive")
    if not math.isfinite(args.resident_roundtrip_atol) or args.resident_roundtrip_atol < 0.0:
        raise SystemExit("--resident-roundtrip-atol must be finite and nonnegative")
    if args.latent_mode == "discrete":
        if args.n_groups <= 0:
            raise SystemExit("discrete latent mode requires positive --n-groups")
        if args.latent_dim % args.n_groups != 0:
            raise SystemExit("--latent-dim must be divisible by --n-groups")
        if (
            args.codebook_size < 2
            or args.codebook_size > 65536
            or (args.codebook_size & (args.codebook_size - 1)) != 0
        ):
            raise SystemExit("--codebook-size must be a power of two in [2, 65536]")
        if args.kmeans_iterations <= 0:
            raise SystemExit("--kmeans-iterations must be positive")
        if args.assignment_icd_sweeps <= 0:
            raise SystemExit("--assignment-icd-sweeps must be positive")
        if args.dead_code_min_usage < 0:
            raise SystemExit("--dead-code-min-usage must be nonnegative")
    else:
        if args.n_groups != 0 or args.codebook_size != 0:
            raise SystemExit("continuous latent mode must not receive --n-groups or --codebook-size")
        if args.ste_warmup_steps != 0:
            raise SystemExit("continuous latent mode has no vector-quantization warm start")
    if args.alternations <= 0 or args.steps_per_alternation <= 0 or args.batch_size <= 0:
        raise SystemExit("--alternations, --steps-per-alternation, and --batch-size must be positive")
    if args.ste_warmup_steps < 0:
        raise SystemExit("--ste-warmup-steps must be nonnegative")
    if not (0.0 < args.ste_ema_gamma < 1.0):
        raise SystemExit("--ste-ema-gamma must be in (0, 1)")
    if args.ste_commitment_beta < 0 or not math.isfinite(args.ste_commitment_beta):
        raise SystemExit("--ste-commitment-beta must be finite and nonnegative")
    if args.lr <= 0 or not math.isfinite(args.lr):
        raise SystemExit("--lr must be positive and finite")
    if not (0.0 <= args.adamw_beta1 < 1.0) or not (0.0 <= args.adamw_beta2 < 1.0):
        raise SystemExit("--adamw-beta1 and --adamw-beta2 must be in [0, 1)")
    if args.adamw_eps <= 0 or not math.isfinite(args.adamw_eps):
        raise SystemExit("--adamw-eps must be positive and finite")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise SystemExit("--weight-decay must be finite and nonnegative")
    if args.grad_clip_norm <= 0 or not math.isfinite(args.grad_clip_norm):
        raise SystemExit("--grad-clip-norm must be positive and finite")
    total_steps = args.ste_warmup_steps + args.alternations * args.steps_per_alternation
    if args.warmup_steps < 0 or args.warmup_steps > total_steps:
        raise SystemExit("--warmup-steps must be between 0 and the total Adam step count")
    if args.heldout_eval_size < 0 or args.heldout_eval_interval_steps < 0:
        raise SystemExit("--heldout-eval-size and --heldout-eval-interval-steps must be nonnegative")
    if args.metric_eigenvalue_floor <= 0 or args.metric_eigenvalue_floor >= 1:
        raise SystemExit("--metric-eigenvalue-floor must be in (0, 1)")
    if args.assignment_ridge < 0 or not math.isfinite(args.assignment_ridge):
        raise SystemExit("--assignment-ridge must be finite and nonnegative")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:  # noqa: PLR0915
    args = parse_args()
    validate_args(args)

    import numpy as np
    import torch
    from safetensors import safe_open

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    layers = args.num_layers
    d = args.d_ple
    latent_dim = args.latent_dim
    n_groups = args.n_groups
    codebook_size = args.codebook_size
    group_dim = latent_dim // n_groups if args.latent_mode == "discrete" else 0
    code_bits = int(math.log2(codebook_size)) if args.latent_mode == "discrete" else 0
    total_adam_steps = args.ste_warmup_steps + args.alternations * args.steps_per_alternation

    components = storage_components(
        262144, layers, d, args.latent_mode, latent_dim, n_groups, codebook_size
    )

    # ----- load original PLE and the locked training inputs ----- #
    original_packed_cpu, ple_key, ple_shape = load_ple_tensor(
        safe_open, args.it_weights, layers, d, allow_recon_key=False
    )
    vocab = int(original_packed_cpu.shape[0])
    packed = original_packed_cpu.to(device=device, dtype=torch.bfloat16)
    del original_packed_cpu

    (
        token_ids_cpu,
        gate_by_layer_cpu,
        runtime_token_scale,
        training_loss_metadata,
    ) = build_exact_gate_cache_forward_path_training_inputs(
        torch, safe_open, args.training_gate_cache, args.training_gate_cache_config, vocab, layers, d
    )
    training_loss_metadata = {
        **training_loss_metadata,
        "training_objective_mode": TRAINING_OBJECTIVE_MODE,
        "training_forward_path_formula": (
            "mean || W_d_l((g_l * (decoded_row - original_row)) * runtime_token_scale) ||^2"
        ),
        "training_gate_cache_materialization": (
            "device-resident per-layer float16 gate tensors loaded from the fixed D_out gate-cache file"
        ),
    }
    projection_weights, projection_metadata = load_projection_weights(
        torch, safe_open, args.it_weights, layers, d, device
    )
    token_ids_device = token_ids_cpu.to(device)
    gate_by_layer = [gate.to(device=device) for gate in gate_by_layer_cpu]
    del gate_by_layer_cpu
    n_positions = int(token_ids_device.numel())
    scale = float(runtime_token_scale)

    # ----- heldout split over gate-cache sample positions ----- #
    heldout_eval_enabled = args.heldout_eval_size > 0
    heldout_sample_ids_cpu = None
    heldout_layer_ids_cpu = None
    train_sample_pool_cpu = torch.arange(n_positions, dtype=torch.int64)
    heldout_split_scope = "disabled"
    if heldout_eval_enabled:
        if args.heldout_eval_size >= n_positions:
            raise SystemExit("--heldout-eval-size must be smaller than the gate-cache position count")
        heldout_rng = torch.Generator(device="cpu").manual_seed(args.heldout_eval_seed)
        permutation = torch.randperm(n_positions, generator=heldout_rng)
        heldout_sample_ids_cpu = permutation[: args.heldout_eval_size].contiguous()
        train_sample_pool_cpu = permutation[args.heldout_eval_size :].contiguous()
        heldout_layer_ids_cpu = torch.randint(0, layers, (args.heldout_eval_size,), generator=heldout_rng)
        heldout_split_scope = "finite_training_sample_id_holdout_excluded_from_training_sampling_and_gate_second_moments"
    pool_sample_ids_device = train_sample_pool_cpu.to(device)

    print(
        f"[optic] V={vocab} L={layers} d={d} latent_mode={args.latent_mode} "
        f"Z={latent_dim} n_groups={n_groups} codebook_size={codebook_size} code_bits={code_bits} "
        f"alternations={args.alternations} steps_per_alternation={args.steps_per_alternation} "
        f"ste_warmup_steps={args.ste_warmup_steps} storage={components['storage_bytes']} "
        f"ratio={components['raw_bytes'] / components['storage_bytes']:.1f}x "
        f"[{time.time() - t0:.0f}s]",
        flush=True,
    )

    # ----- output-weighted metric and whitened-PCA initialization ----- #
    gate_second_moment = compute_gate_second_moments(
        torch, gate_by_layer, pool_sample_ids_device, layers, d, args.gate_moment_chunk_size
    )
    output_metric = build_output_metric(torch, projection_weights, gate_second_moment, layers)
    del gate_second_moment
    sqrt_metric, invsqrt_metric = metric_sqrt_and_invsqrt(torch, output_metric, args.metric_eigenvalue_floor)
    mean, covariance = token_vector_mean_and_covariance(torch, packed, args.init_covariance_chunk_size)
    init_eigenvalues, basis = output_weighted_pca_basis(torch, covariance, sqrt_metric, layers, d, latent_dim)
    del covariance
    print(
        f"[optic] init whitened-PCA done. top5 output-weighted eigenvalues="
        f"{[round(float(v), 6) for v in init_eigenvalues[:5].tolist()]} [{time.time() - t0:.0f}s]",
        flush=True,
    )
    decoder_weight_init = torch.empty((layers, d, latent_dim), dtype=torch.float32, device=device)
    for layer in range(layers):
        decoder_weight_init[layer] = invsqrt_metric[layer] @ basis[layer * d : (layer + 1) * d, :]
    del invsqrt_metric
    latents_init = build_initial_latents(torch, packed, mean, sqrt_metric, basis, layers, d, args.eval_chunk_size)
    del sqrt_metric, basis

    allocation_permutation = list(range(latent_dim))
    if args.latent_mode == "discrete" and args.group_dim_allocation == "eigenvalue_balanced":
        allocation_permutation = eigenvalue_balanced_allocation(
            init_eigenvalues.tolist(), n_groups, group_dim
        )
        perm = torch.tensor(allocation_permutation, dtype=torch.int64, device=device)
        latents_init = latents_init[:, perm].contiguous()
        decoder_weight_init = decoder_weight_init[:, :, perm].contiguous()

    decoder_weight = torch.nn.Parameter(decoder_weight_init)
    decoder_bias = torch.nn.Parameter(mean.view(layers, d).clone())
    del decoder_weight_init

    rng_init = torch.Generator(device="cpu").manual_seed(args.seed)
    codes = None
    codebooks = None
    latent_table = None
    group_arange = None
    if args.latent_mode == "discrete":
        codebooks_init, codes = kmeans_per_group(
            torch,
            latents_init,
            n_groups,
            group_dim,
            codebook_size,
            args.kmeans_iterations,
            rng_init,
            args.assignment_chunk_size,
        )
        codebooks = torch.nn.Parameter(codebooks_init)
        group_arange = torch.arange(n_groups, device=device)
        latent_parameter = codebooks
        print(f"[optic] per-group k-means init done [{time.time() - t0:.0f}s]", flush=True)
    else:
        latent_table = torch.nn.Parameter(latents_init.clone())
        latent_parameter = latent_table
    ste_latent_table = None
    if args.ste_warmup_steps > 0:
        ste_latent_table = torch.nn.Parameter(latents_init.clone())
    del latents_init

    # ----- optimizer ----- #
    if args.optimizer != "adamw":
        raise SystemExit(f"unsupported optimizer: {args.optimizer}")
    parameter_groups = [
        {"params": [decoder_weight, decoder_bias], "weight_decay": args.weight_decay},
        {"params": [latent_parameter], "weight_decay": 0.0},
    ]
    if ste_latent_table is not None:
        parameter_groups.append({"params": [ste_latent_table], "weight_decay": 0.0})
    opt = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        betas=(args.adamw_beta1, args.adamw_beta2),
        eps=args.adamw_eps,
    )
    clip_parameters = [decoder_weight, decoder_bias, latent_parameter]
    if ste_latent_table is not None:
        clip_parameters.append(ste_latent_table)
    rng_train = torch.Generator(device="cpu").manual_seed(args.seed)

    # ----- shared forward-path loss ----- #
    def latent_rows_quantized(tokens):
        idx = codes.index_select(0, tokens)
        return codebooks[group_arange[None, :], idx].reshape(int(tokens.numel()), latent_dim)

    def latent_rows_continuous(tokens):
        return latent_table.index_select(0, tokens)

    def forward_path_loss(sample_ids_cpu, layer_ids_cpu, latent_rows_fn):
        sample_ids = sample_ids_cpu.to(device)
        layer_ids = layer_ids_cpu.to(device)
        token_ids = token_ids_device.index_select(0, sample_ids)
        order = torch.argsort(layer_ids)
        segment_bounds = [0, *torch.bincount(layer_ids, minlength=layers).cumsum(0).tolist()]
        tokens_sorted = token_ids.index_select(0, order)
        samples_sorted = sample_ids.index_select(0, order)
        loss_sum = decoder_bias.new_zeros(())
        for layer in range(layers):
            start = segment_bounds[layer]
            stop = segment_bounds[layer + 1]
            if stop <= start:
                continue
            tokens = tokens_sorted[start:stop]
            z = latent_rows_fn(tokens)
            pred = z @ decoder_weight[layer].T + decoder_bias[layer]
            target = packed[:, layer * d : (layer + 1) * d].index_select(0, tokens).to(torch.float32)
            gate = gate_by_layer[layer].index_select(0, samples_sorted[start:stop]).to(torch.float32)
            projected_error = ((gate * (pred - target)) * scale) @ projection_weights[layer].T
            loss_sum = loss_sum + projected_error.square().sum()
            del tokens, z, pred, target, gate, projected_error
        return loss_sum / float(sample_ids_cpu.numel())

    train_latent_rows = latent_rows_quantized if args.latent_mode == "discrete" else latent_rows_continuous

    def heldout_loss_value():
        with torch.no_grad():
            return float(
                forward_path_loss(heldout_sample_ids_cpu, heldout_layer_ids_cpu, train_latent_rows).item()
            )

    training_loss_trace: list[dict[str, Any]] = []
    heldout_loss_trace: list[dict[str, Any]] = []
    assignment_trace: list[dict[str, Any]] = []
    if heldout_eval_enabled:
        heldout_loss_trace.append({"step": 0, "loss": heldout_loss_value()})

    global_step = 0

    def sample_training_batch():
        offsets = torch.randint(0, int(train_sample_pool_cpu.numel()), (args.batch_size,), generator=rng_train)
        sample_ids_cpu = train_sample_pool_cpu[offsets]
        layer_ids_cpu = torch.randint(0, layers, (args.batch_size,), generator=rng_train)
        return sample_ids_cpu, layer_ids_cpu

    def run_adam_step(loss_fn, phase: str):
        nonlocal global_step
        current_lr = set_learning_rate(opt, args.lr, args.lr_schedule, global_step, total_adam_steps, args.warmup_steps)
        sample_ids_cpu, layer_ids_cpu = sample_training_batch()
        loss = loss_fn(sample_ids_cpu, layer_ids_cpu)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_parameters, args.grad_clip_norm)
        opt.step()
        global_step += 1
        should_log = (
            global_step == 1
            or global_step % TRAINING_LOSS_TRACE_INTERVAL_STEPS == 0
            or global_step == total_adam_steps
        )
        if should_log:
            loss_value = float(loss.item())
            training_loss_trace.append(
                {"step": global_step, "loss": loss_value, "lr": float(current_lr), "phase": phase}
            )
            print(
                f"[optic] step={global_step}/{total_adam_steps} phase={phase} "
                f"objective_loss={loss_value:.6e} lr={current_lr:.6e} [{time.time() - t0:.0f}s]",
                flush=True,
            )
        if (
            heldout_eval_enabled
            and args.heldout_eval_interval_steps > 0
            and global_step % args.heldout_eval_interval_steps == 0
        ):
            heldout_loss_trace.append({"step": global_step, "loss": heldout_loss_value()})
        del sample_ids_cpu, layer_ids_cpu, loss

    # ----- optional straight-through warm start with EMA codebooks ----- #
    if args.ste_warmup_steps > 0:
        ema_counts = torch.zeros((n_groups, codebook_size), dtype=torch.float32, device=device)
        ema_sums = torch.zeros((n_groups, codebook_size, group_dim), dtype=torch.float32, device=device)
        with torch.no_grad():
            for group in range(n_groups):
                usage = torch.bincount(codes[:, group], minlength=codebook_size).to(torch.float32)
                ema_counts[group] = usage
                ema_sums[group].index_add_(
                    0, codes[:, group], ste_latent_table.data[:, group * group_dim : (group + 1) * group_dim]
                )

        def ste_loss(sample_ids_cpu, layer_ids_cpu):
            batch_z_e_pairs: list[tuple[Any, Any, Any]] = []

            def latent_rows_ste(tokens):
                z = ste_latent_table.index_select(0, tokens)
                z_groups = z.view(int(tokens.numel()), n_groups, group_dim)
                with torch.no_grad():
                    cb = codebooks.detach()
                    dists = (
                        cb.square().sum(dim=2)[None, :, :]
                        - 2.0 * torch.einsum("bmg,mkg->bmk", z_groups, cb)
                    )
                    idx = dists.argmin(dim=2)
                    entries = cb[group_arange[None, :], idx]
                batch_z_e_pairs.append((z_groups, entries, idx))
                quantized = z_groups + (entries - z_groups).detach()
                return quantized.reshape(int(tokens.numel()), latent_dim)

            loss = forward_path_loss(sample_ids_cpu, layer_ids_cpu, latent_rows_ste)
            commitment = decoder_bias.new_zeros(())
            n_rows = 0
            for z_groups, entries, _ in batch_z_e_pairs:
                commitment = commitment + (z_groups - entries.detach()).square().sum()
                n_rows += int(z_groups.shape[0])
            loss = loss + args.ste_commitment_beta * commitment / float(max(n_rows * latent_dim, 1))
            with torch.no_grad():
                for z_groups, _, idx in batch_z_e_pairs:
                    for group in range(n_groups):
                        counts = torch.bincount(idx[:, group], minlength=codebook_size).to(torch.float32)
                        sums = torch.zeros((codebook_size, group_dim), dtype=torch.float32, device=device)
                        sums.index_add_(0, idx[:, group], z_groups[:, group, :].detach())
                        ema_counts[group].mul_(args.ste_ema_gamma).add_(counts, alpha=1.0 - args.ste_ema_gamma)
                        ema_sums[group].mul_(args.ste_ema_gamma).add_(sums, alpha=1.0 - args.ste_ema_gamma)
                        denom = (ema_counts[group] + args.ste_ema_eps)[:, None]
                        codebooks.data[group] = ema_sums[group] / denom
            return loss

        for _ in range(args.ste_warmup_steps):
            run_adam_step(ste_loss, "ste_warmup")
        del ema_counts, ema_sums, ste_latent_table

    # ----- alternation loop ----- #
    def train_loss(sample_ids_cpu, layer_ids_cpu):
        return forward_path_loss(sample_ids_cpu, layer_ids_cpu, train_latent_rows)

    for cycle in range(1, args.alternations + 1):
        run_assignment = args.latent_mode == "discrete" and (cycle > 1 or args.ste_warmup_steps > 0)
        if run_assignment:
            assignment_start = time.time()
            z_star, quantization_error, stats = assignment_step(
                torch,
                packed,
                codes,
                codebooks,
                decoder_weight,
                decoder_bias,
                output_metric,
                layers,
                d,
                n_groups,
                group_dim,
                args.assignment_icd_sweeps,
                args.assignment_ridge,
                args.assignment_chunk_size,
            )
            dead_reinitialized = reinitialize_dead_codes(
                torch,
                opt,
                codes,
                codebooks,
                z_star,
                quantization_error,
                n_groups,
                group_dim,
                codebook_size,
                args.dead_code_min_usage,
            )
            del z_star, quantization_error
            stats.update(
                {
                    "alternation": cycle,
                    "step": global_step,
                    "dead_codes_reinitialized": dead_reinitialized,
                    "assignment_seconds": time.time() - assignment_start,
                }
            )
            assignment_trace.append(stats)
            print(
                f"[optic] assignment cycle={cycle} changed_fraction={stats['codes_changed_fraction']:.4f} "
                f"mean_qerr={stats['mean_output_metric_quantization_error']:.6e} "
                f"dead_reinit={dead_reinitialized} ({stats['assignment_seconds']:.1f}s) [{time.time() - t0:.0f}s]",
                flush=True,
            )
        for _ in range(args.steps_per_alternation):
            run_adam_step(train_loss, "alternation")

    if heldout_eval_enabled and (not heldout_loss_trace or heldout_loss_trace[-1]["step"] != global_step):
        heldout_loss_trace.append({"step": global_step, "loss": heldout_loss_value()})

    with torch.no_grad():
        for name, parameter in (("decoder_weight", decoder_weight), ("decoder_bias", decoder_bias)):
            if not bool(torch.isfinite(parameter).all().item()):
                raise SystemExit(f"trained parameter {name} contains non-finite values")
        if not bool(torch.isfinite(latent_parameter).all().item()):
            raise SystemExit("trained latent parameter contains non-finite values")

    # ----- save resident payload ----- #
    metadata = {
        "compressed_ple_format_id": FORMAT_ID,
        "kind": METHOD_ID,
        "method": METHOD_ID,
        "full_ple_coverage": True,
        "hidden_raw_copy_allowed": False,
        "latent_mode": args.latent_mode,
        "latent_dim": latent_dim,
        "n_groups": n_groups,
        "codebook_size": codebook_size,
        "code_bits": code_bits,
        "group_dim": group_dim,
        "group_dim_allocation": args.group_dim_allocation if args.latent_mode == "discrete" else None,
        "decoder_family": "per_layer_linear_head_plus_per_layer_center",
        "decode_formula": "E_hat[l, t] = decoder_weight[l] @ z_t + decoder_bias[l]",
        "latent_construction": (
            "concatenation of per-group codebook entries selected by packed token codes"
            if args.latent_mode == "discrete"
            else "cached fp16 per-token latent table"
        ),
        "init_recipe": (
            "output-weighted whitened PCA of all token vectors "
            "(G_l = per-layer projection gram elementwise-weighted by the training-pool gate second moment), "
            "eigenvalue-balanced group allocation, per-group k-means on the init latent"
        ),
        "init_top_output_weighted_eigenvalues": [float(v) for v in init_eigenvalues.tolist()],
        "group_dim_allocation_permutation": allocation_permutation if args.latent_mode == "discrete" else None,
        "alternations": args.alternations,
        "steps_per_alternation": args.steps_per_alternation,
        "assignment_icd_sweeps": args.assignment_icd_sweeps,
        "assignment_ridge": args.assignment_ridge,
        "dead_code_min_usage": args.dead_code_min_usage,
        "kmeans_iterations": args.kmeans_iterations,
        "ste_warmup_steps": args.ste_warmup_steps,
        "ste_commitment_beta": args.ste_commitment_beta,
        "ste_ema_gamma": args.ste_ema_gamma,
        "metric_eigenvalue_floor": args.metric_eigenvalue_floor,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "adamw_beta1": args.adamw_beta1,
        "adamw_beta2": args.adamw_beta2,
        "adamw_eps": args.adamw_eps,
        "weight_decay_decoder_only": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "total_adam_steps": total_adam_steps,
        "seed": args.seed,
        "heldout_eval_enabled": heldout_eval_enabled,
        "heldout_eval_size": args.heldout_eval_size,
        "heldout_eval_seed": args.heldout_eval_seed,
        "heldout_split_scope": heldout_split_scope,
        **training_loss_metadata,
        **projection_metadata,
        "ple_key": ple_key,
        "ple_shape": list(ple_shape),
        "vocab_size": vocab,
        "num_layers": layers,
        "d_ple": d,
        "resident_tensors": (
            ["token_codes_packed_uint8", "codebooks_fp16", "decoder_weight_fp16", "decoder_bias_fp16"]
            if args.latent_mode == "discrete"
            else ["latent_table_fp16", "decoder_weight_fp16", "decoder_bias_fp16"]
        ),
        **{key: value for key, value in components.items()},
    }
    payload: dict[str, Any] = {
        "metadata": np.array(json.dumps(metadata)),
        "decoder_weight_fp16": decoder_weight.detach().cpu().numpy().astype("float16"),
        "decoder_bias_fp16": decoder_bias.detach().cpu().numpy().astype("float16"),
    }
    if args.latent_mode == "discrete":
        payload["token_codes_packed_uint8"] = pack_codes(np, codes.cpu().numpy(), code_bits)
        payload["codebooks_fp16"] = codebooks.detach().cpu().numpy().astype("float16")
    else:
        payload["latent_table_fp16"] = latent_table.detach().cpu().numpy().astype("float16")
    compressed_path = Path(args.save_compressed_ple_npz)
    compressed_path.parent.mkdir(parents=True, exist_ok=True)
    with compressed_path.open("wb") as handle:
        np.savez(handle, **payload)

    # ----- reload, materialize, verify, evaluate ----- #
    resident_payload = load_resident_payload(
        np,
        compressed_path,
        vocab,
        layers,
        d,
        args.latent_mode,
        latent_dim,
        n_groups,
        codebook_size,
        components["storage_bytes"],
    )
    materialize_reconstruction(
        torch,
        np,
        resident_payload,
        Path(args.save_recon),
        vocab,
        layers,
        d,
        args.latent_mode,
        latent_dim,
        n_groups,
        codebook_size,
        device,
        args.eval_chunk_size,
    )
    resident_file_roundtrip_max_abs_diff = verify_resident_roundtrip(
        torch,
        np,
        safe_open,
        compressed_path,
        Path(args.save_recon),
        vocab,
        layers,
        d,
        args.latent_mode,
        latent_dim,
        n_groups,
        codebook_size,
        components["storage_bytes"],
        device,
        args.eval_chunk_size,
        args.resident_roundtrip_atol,
    )
    decoded_original_checkpoint_ple_nmse = plain_nmse_recon_file_vs_device_original(
        torch, safe_open, Path(args.save_recon), packed, layers, d, args.eval_chunk_size
    )
    decoded_gate_cache_forward_path_nmse = gate_cache_forward_path_nmse(
        torch,
        safe_open,
        Path(args.save_recon),
        packed,
        token_ids_device,
        gate_by_layer,
        projection_weights,
        scale,
        layers,
        d,
        args.forward_path_eval_position_chunk_size,
    )

    ratio = components["raw_bytes"] / components["storage_bytes"]
    out = {
        "benchmark": benchmark_id_for_metrics_output(Path(args.output)),
        "method": METHOD_ID,
        "compressed_ple_format_id": FORMAT_ID,
        "latent_mode": args.latent_mode,
        "latent_dim": latent_dim,
        "n_groups": n_groups,
        "codebook_size": codebook_size,
        "code_bits": code_bits,
        "group_dim": group_dim,
        "group_dim_allocation": args.group_dim_allocation if args.latent_mode == "discrete" else None,
        "alternations": args.alternations,
        "steps_per_alternation": args.steps_per_alternation,
        "assignment_icd_sweeps": args.assignment_icd_sweeps,
        "assignment_ridge": args.assignment_ridge,
        "assignment_chunk_size": args.assignment_chunk_size,
        "dead_code_min_usage": args.dead_code_min_usage,
        "kmeans_iterations": args.kmeans_iterations,
        "ste_warmup_steps": args.ste_warmup_steps,
        "ste_commitment_beta": args.ste_commitment_beta,
        "ste_ema_gamma": args.ste_ema_gamma,
        "ste_ema_eps": args.ste_ema_eps,
        "metric_eigenvalue_floor": args.metric_eigenvalue_floor,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "adamw_beta1": args.adamw_beta1,
        "adamw_beta2": args.adamw_beta2,
        "adamw_eps": args.adamw_eps,
        "weight_decay_decoder_only": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "total_adam_steps": total_adam_steps,
        "eval_chunk_size": args.eval_chunk_size,
        "seed": args.seed,
        "heldout_eval_enabled": heldout_eval_enabled,
        "heldout_eval_size": args.heldout_eval_size,
        "heldout_eval_seed": args.heldout_eval_seed,
        "heldout_eval_interval_steps": args.heldout_eval_interval_steps,
        "heldout_split_scope": heldout_split_scope,
        "heldout_split_enforced": heldout_eval_enabled,
        "training_loss_trace": training_loss_trace,
        "training_loss_trace_interval_steps": TRAINING_LOSS_TRACE_INTERVAL_STEPS,
        "training_loss_name": "objective_loss",
        "training_loss_initial_logged": training_loss_trace[0]["loss"] if training_loss_trace else None,
        "training_loss_final_logged": training_loss_trace[-1]["loss"] if training_loss_trace else None,
        "training_loss_min_logged": min((entry["loss"] for entry in training_loss_trace), default=None),
        "heldout_loss_trace": heldout_loss_trace,
        "heldout_loss_name": "objective_loss",
        "heldout_loss_initial_logged": heldout_loss_trace[0]["loss"] if heldout_loss_trace else None,
        "heldout_loss_final_logged": heldout_loss_trace[-1]["loss"] if heldout_loss_trace else None,
        "heldout_loss_min_logged": min((entry["loss"] for entry in heldout_loss_trace), default=None),
        "assignment_trace": assignment_trace,
        "init_top_output_weighted_eigenvalues": [float(v) for v in init_eigenvalues.tolist()],
        "group_dim_allocation_permutation": allocation_permutation if args.latent_mode == "discrete" else None,
        **training_loss_metadata,
        **projection_metadata,
        "ple_key": ple_key,
        "ple_shape": list(ple_shape),
        "validated_full_ple_shape": list(ple_shape) == [vocab, layers * d],
        "decoded_original_checkpoint_ple_nmse": decoded_original_checkpoint_ple_nmse,
        "decoded_gate_cache_forward_path_nmse": decoded_gate_cache_forward_path_nmse,
        "decoded_gate_cache_forward_path_relative_error": math.sqrt(decoded_gate_cache_forward_path_nmse),
        "decoded_gate_cache_forward_path_nmse_scope": (
            "exact forward-path NMSE over every position and layer of the fixed D_out gate cache, "
            "including any heldout positions"
        ),
        "resident_roundtrip_atol": args.resident_roundtrip_atol,
        "resident_file_roundtrip_max_abs_diff": resident_file_roundtrip_max_abs_diff,
        "resident_roundtrip_max_abs_diff": resident_file_roundtrip_max_abs_diff,
        "resident_file_roundtrip_passed": resident_file_roundtrip_max_abs_diff <= args.resident_roundtrip_atol,
        "resident_roundtrip_passed": resident_file_roundtrip_max_abs_diff <= args.resident_roundtrip_atol,
        "resident_tensor_byte_check_passed": True,
        "resident_byte_accounting_exact": True,
        **components,
        "storage_gib": components["storage_bytes"] / 1024**3,
        "compression_ratio": ratio,
        "elapsed_seconds": time.time() - t0,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(
        f"[optic] decoded-original-checkpoint-PLE NMSE={decoded_original_checkpoint_ple_nmse:.6f} "
        f"gate-cache-forward-path NMSE={decoded_gate_cache_forward_path_nmse:.6f}",
        flush=True,
    )
    print(
        f"[optic] storage={components['storage_bytes']} bytes ratio={ratio:.1f}x "
        f"roundtrip_max_abs_diff={resident_file_roundtrip_max_abs_diff:.6g} "
        f"elapsed={time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
