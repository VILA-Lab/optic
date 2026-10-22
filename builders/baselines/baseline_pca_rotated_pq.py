#!/usr/bin/env python3
"""Generic baseline: global PCA-rotated product quantization for full PLE.

It learns one global mean and one global orthogonal PCA rotation over all
(layer, token) PLE rows, then applies flat PQ in the rotated coordinate system.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from baseline_common import RECON_KEY, load_ple_tensor, require_uint16_code_count
from baseline_residual_vq import benchmark_id_for_metrics_output, sha256_file, unpack_nbit

DEFAULT_IT_WEIGHTS = (
    "<CACHE_DIR>/huggingface/hub/"
    "models--google--gemma-4-E2B-it/snapshots/"
    "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf/model.safetensors"
)


def pack_nbit(values, bits):
    import numpy as np

    values = values.astype(np.uint16, copy=False)
    shifts = np.arange(bits, dtype=np.uint16)
    bit_matrix = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little")



def verify_resident_roundtrip(
    torch,
    np,
    safe_open,
    compressed_path: Path,
    recon_path: Path,
    *,
    vocab: int,
    layers: int,
    d: int,
    subspaces: int,
    clusters: int,
    bits: int,
    eval_batch_size: int,
    atol: float,
    device,
) -> float:
    rows = vocab * layers
    d_sub = d // subspaces
    with np.load(compressed_path, allow_pickle=False) as npz:
        if "metadata" not in npz.files:
            raise SystemExit(f"{compressed_path} missing metadata")
        metadata = json.loads(str(npz["metadata"].item()))
        if metadata.get("compressed_ple_format_id") != "global_pca_rotated_pq_npz":
            raise SystemExit(f"{compressed_path} compressed_ple_format_id mismatch")
        if (
            int(metadata.get("num_subspaces")) != subspaces
            or int(metadata.get("clusters")) != clusters
            or int(metadata.get("bits")) != bits
        ):
            raise SystemExit(f"{compressed_path} metadata num_subspaces/clusters/bits mismatch")
        if list(metadata.get("ple_shape", [])) != [vocab, layers * d]:
            raise SystemExit(f"{compressed_path} metadata ple_shape mismatch")
        codebook_array = npz["codebooks"]
        if tuple(codebook_array.shape) != (subspaces, clusters, d_sub):
            raise SystemExit(
                f"{compressed_path} codebooks shape {tuple(codebook_array.shape)} != {(subspaces, clusters, d_sub)}"
            )
        rotation_array = npz["rotation"]
        if tuple(rotation_array.shape) != (d, d):
            raise SystemExit(f"{compressed_path} rotation shape {tuple(rotation_array.shape)} != {(d, d)}")
        mean_array = npz["mean"]
        if tuple(mean_array.shape) != (d,):
            raise SystemExit(f"{compressed_path} mean shape {tuple(mean_array.shape)} != {(d,)}")
        resident_codebooks = torch.from_numpy(codebook_array.astype("float32", copy=False)).to(device)
        resident_rotation_t = torch.from_numpy(rotation_array.astype("float32", copy=False)).to(device).T.contiguous()
        resident_mean = torch.from_numpy(mean_array.astype("float32", copy=False)).to(device)
        resident_codes = []
        for subspace in range(subspaces):
            name = f"codes_m{subspace:02d}"
            if name not in npz.files:
                raise SystemExit(f"{compressed_path} missing {name}")
            resident_codes.append(torch.from_numpy(unpack_nbit(npz[name], bits, rows)).to(torch.long))

    max_abs_diff = 0.0
    with safe_open(str(recon_path), framework="pt", device="cpu") as recon_file:
        if RECON_KEY not in set(recon_file.keys()):
            raise SystemExit(f"missing dense reconstruction tensor key in {recon_path}: {RECON_KEY}")
        recon_slice = recon_file.get_slice(RECON_KEY)
        expected_shape = [vocab, layers * d]
        if list(recon_slice.get_shape()) != expected_shape:
            raise SystemExit(f"dense reconstruction shape {list(recon_slice.get_shape())} != {expected_shape}")
        for layer in range(layers):
            row_offset = layer * vocab
            for start in range(0, vocab, eval_batch_size):
                stop = min(vocab, start + eval_batch_size)
                row_slice = slice(row_offset + start, row_offset + stop)
                decoded_rot = torch.empty((stop - start, d), dtype=torch.float32, device=device)
                for subspace in range(subspaces):
                    start_dim = subspace * d_sub
                    stop_dim = (subspace + 1) * d_sub
                    codes = resident_codes[subspace][row_slice].to(device)
                    decoded_rot[:, start_dim:stop_dim] = resident_codebooks[subspace, codes]
                decoded = decoded_rot @ resident_rotation_t + resident_mean
                decoded_bf16 = decoded.to(torch.bfloat16).cpu()
                saved = recon_slice[start:stop, layer * d : (layer + 1) * d]
                decoded_f32 = decoded_bf16.to(torch.float32)
                saved_f32 = saved.to(torch.float32)
                if not bool(torch.isfinite(decoded_f32).all().item()):
                    raise SystemExit("resident file reload/decode produced non-finite values")
                if not bool(torch.isfinite(saved_f32).all().item()):
                    raise SystemExit("saved BF16 dense PLE reconstruction contains non-finite values")
                diff_tensor = (decoded_f32 - saved_f32).abs()
                diff = float(diff_tensor.max().item()) if diff_tensor.numel() else 0.0
                max_abs_diff = max(max_abs_diff, diff)
                if max_abs_diff > atol:
                    raise SystemExit(f"resident file reload/decode max_abs_diff {max_abs_diff} exceeds tolerance {atol}")
                del decoded_rot, decoded, decoded_bf16, saved, decoded_f32, saved_f32, diff_tensor
    return max_abs_diff


def kmeans(torch, x_cpu, k, iters, batch_size, device, seed):
    n = x_cpu.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(n, generator=rng)[:k]
    centers = x_cpu[init_idx].to(device).clone()
    for it in range(iters):
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, dtype=torch.long, device=device)
        center_norm = centers.square().sum(dim=1).unsqueeze(0)
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            chunk = x_cpu[start:stop].to(device)
            dist = chunk.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * (chunk @ centers.T)
            assign = dist.argmin(dim=1)
            new_centers.index_add_(0, assign, chunk)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.long))
            del chunk, dist, assign
        nonempty = counts > 0
        new_centers[nonempty] /= counts[nonempty].unsqueeze(1).float()
        if bool((~nonempty).any().item()):
            empty = (~nonempty).nonzero(as_tuple=False).flatten()
            repl = torch.randperm(n, generator=rng)[: empty.numel()]
            new_centers[empty] = x_cpu[repl].to(device)
        centers = new_centers
        print(f"[pca-pq] iter {it + 1}/{iters} nonempty={int(nonempty.sum().item())}/{k}", flush=True)

    assignments = torch.empty(n, dtype=torch.int64)
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        chunk = x_cpu[start:stop].to(device)
        dist = chunk.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * (chunk @ centers.T)
        assignments[start:stop] = dist.argmin(dim=1).cpu()
        del chunk, dist
    return centers.cpu(), assignments


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--it-weights", default=DEFAULT_IT_WEIGHTS)
    p.add_argument(
        "--source-ple",
        default="",
        help="Optional source PLE tensor to compress instead of --it-weights. Supports dense PLE reconstruction safetensors.",
    )
    p.add_argument(
        "--reference-it-weights",
        default=DEFAULT_IT_WEIGHTS,
        help="Original checkpoint used only to report reconstruction NMSE versus checkpoint PLE when --source-ple is set.",
    )
    p.add_argument("--num-layers", type=int, default=35)
    p.add_argument("--d-ple", type=int, default=256)
    p.add_argument("--num-subspaces", type=int, default=8)
    p.add_argument("--clusters", type=int, default=4096)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--resident-roundtrip-atol", type=float, default=0.002)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-recon", required=True)
    p.add_argument("--save-compressed-ple-npz", default="")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if args.resident_roundtrip_atol < 0.0 or not math.isfinite(args.resident_roundtrip_atol):
        raise SystemExit("--resident-roundtrip-atol must be finite and nonnegative")

    source_path = args.source_ple or args.it_weights
    print(f"[pca-pq] loading PLE source={source_path}", flush=True)
    packed, ple_key, ple_shape = load_ple_tensor(
        safe_open, source_path, args.num_layers, args.d_ple, allow_recon_key=bool(args.source_ple)
    )
    packed = packed.to(torch.float32)

    v = packed.shape[0]
    l_count = args.num_layers
    d = args.d_ple
    m_count = args.num_subspaces
    k = args.clusters
    require_uint16_code_count("--clusters", k)
    n = v * l_count
    if d % m_count != 0:
        raise ValueError("d_ple must be divisible by num_subspaces")
    d_sub = d // m_count
    bits = int(math.ceil(math.log2(k)))
    device = torch.device(args.device)
    print(f"[pca-pq] V={v} L={l_count} d={d} rows={n} M={m_count} K={k}", flush=True)
    t0 = time.time()

    x_flat = packed.reshape(v, l_count, d).permute(1, 0, 2).reshape(n, d).contiguous()
    del packed
    mean = x_flat.mean(dim=0)
    cov = torch.zeros((d, d), dtype=torch.float64)
    for start in range(0, n, args.batch_size):
        stop = min(n, start + args.batch_size)
        chunk = (x_flat[start:stop] - mean).to(torch.float64)
        cov += chunk.T @ chunk
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    rotation = eigvecs[:, order].to(torch.float32).contiguous()
    print("[pca-pq] global PCA rotation ready", flush=True)

    x_rot = torch.empty_like(x_flat)
    for start in range(0, n, args.batch_size):
        stop = min(n, start + args.batch_size)
        x_rot[start:stop] = (x_flat[start:stop] - mean) @ rotation

    codebooks = np.empty((m_count, k, d_sub), dtype=np.float16)
    assignments = np.empty((m_count, n), dtype=np.uint16)
    recon_rot = torch.empty_like(x_rot)
    for m in range(m_count):
        print(f"[pca-pq] subspace {m + 1}/{m_count}", flush=True)
        sub = x_rot[:, m * d_sub:(m + 1) * d_sub].contiguous()
        centers, codes = kmeans(torch, sub, k, args.iters, args.batch_size, device, args.seed + 1009 * m)
        centers_resident = centers.to(torch.float16).to(torch.float32)
        recon_rot[:, m * d_sub:(m + 1) * d_sub] = centers_resident[codes.long()].to(torch.float32)
        codebooks[m] = centers_resident.to(torch.float16).numpy()
        assignments[m] = codes.to(torch.uint16).numpy()
        del sub, centers, centers_resident, codes

    recon_packed = torch.empty((v, l_count * d), dtype=torch.bfloat16)
    err = 0.0
    norm = 0.0
    rotation_resident = rotation.to(torch.float16).to(torch.float32)
    mean_resident = mean.to(torch.float16).to(torch.float32)
    rotation_t = rotation_resident.T.contiguous()
    for layer in range(l_count):
        rows = slice(layer * v, (layer + 1) * v)
        recon_layer = recon_rot[rows] @ rotation_t + mean_resident
        recon_bf16 = recon_layer.to(torch.bfloat16)
        x_ref = x_flat[rows]
        err += (x_ref - recon_bf16.to(torch.float32)).pow(2).sum().item()
        norm += x_ref.pow(2).sum().item()
        recon_packed[:, layer * d:(layer + 1) * d] = recon_bf16
        del recon_layer, recon_bf16, x_ref
    decoded_source_ple_nmse = err / max(norm, 1e-30)

    decoded_original_checkpoint_ple_nmse = None
    if args.source_ple:
        ref, _, _ = load_ple_tensor(safe_open, args.reference_it_weights, args.num_layers, args.d_ple)
        ref = ref.to(torch.float32)
        if tuple(ref.shape) != tuple(recon_packed.shape):
            raise SystemExit(f"reference shape mismatch: ref={tuple(ref.shape)} recon={tuple(recon_packed.shape)}")
        diff = recon_packed.to(torch.float32) - ref
        decoded_original_checkpoint_ple_nmse = diff.double().square().sum().item() / max(ref.double().square().sum().item(), 1e-30)
        del ref, diff

    code_bytes = math.ceil(n * m_count * bits / 8)
    codebook_bytes = m_count * k * d_sub * 2
    rotation_bytes = d * d * 2
    mean_bytes = d * 2
    storage_bytes = code_bytes + codebook_bytes + rotation_bytes + mean_bytes
    raw_bytes = n * d * 2
    ratio = raw_bytes / storage_bytes
    print(f"[pca-pq] decoded-source-PLE NMSE={decoded_source_ple_nmse:.6f}", flush=True)
    if decoded_original_checkpoint_ple_nmse is not None:
        print(f"[pca-pq] decoded-original-checkpoint-PLE NMSE={decoded_original_checkpoint_ple_nmse:.6f}", flush=True)
    print(f"[pca-pq] storage={storage_bytes} bytes ratio={ratio:.4f}x", flush=True)

    Path(args.save_recon).parent.mkdir(parents=True, exist_ok=True)
    save_file({"embed_tokens_per_layer.weight": recon_packed}, args.save_recon)
    compressed_path = Path(args.save_compressed_ple_npz) if args.save_compressed_ple_npz else None
    if compressed_path is not None:
        compressed_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "codebooks": codebooks,
            "rotation": rotation_resident.to(torch.float16).numpy(),
            "mean": mean_resident.to(torch.float16).numpy(),
            "metadata": np.array(json.dumps({
                "compressed_ple_format_id": "global_pca_rotated_pq_npz",
                "num_subspaces": m_count,
                "clusters": k,
                "bits": bits,
                "ple_key": ple_key,
                "ple_shape": ple_shape,
            })),
        }
        for m in range(m_count):
            payload[f"codes_m{m:02d}"] = pack_nbit(assignments[m], bits)
        np.savez(compressed_path, **payload)

    recon_path = Path(args.save_recon)
    output_path = Path(args.output)
    resident_roundtrip_max_abs_diff = None
    if compressed_path is not None:
        resident_roundtrip_max_abs_diff = verify_resident_roundtrip(
            torch,
            np,
            safe_open,
            compressed_path,
            recon_path,
            vocab=v,
            layers=l_count,
            d=d,
            subspaces=m_count,
            clusters=k,
            bits=bits,
            eval_batch_size=args.eval_batch_size,
            atol=args.resident_roundtrip_atol,
            device=device,
        )

    out = {
        "benchmark": benchmark_id_for_metrics_output(output_path),
        "method": "global_pca_rotated_pq",
        "compressed_ple_format_id": "global_pca_rotated_pq_npz",
        "num_subspaces": m_count,
        "clusters": k,
        "bits_per_code": bits,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "resident_roundtrip_atol": args.resident_roundtrip_atol,
        "seed": args.seed,
        "source_ple": args.source_ple or "",
        "ple_key": ple_key,
        "ple_shape": list(ple_shape),
        "validated_full_ple_shape": True,
        "decoded_source_ple_nmse": decoded_source_ple_nmse,
        "decoded_original_checkpoint_ple_nmse": decoded_original_checkpoint_ple_nmse,
        "target_nmse": decoded_source_ple_nmse,
        "raw_nmse": decoded_source_ple_nmse,
        "resident_file_roundtrip_max_abs_diff": resident_roundtrip_max_abs_diff,
        "resident_file_roundtrip_passed": (
            None if resident_roundtrip_max_abs_diff is None else resident_roundtrip_max_abs_diff <= args.resident_roundtrip_atol
        ),
        "code_bytes": code_bytes,
        "codebook_bytes": codebook_bytes,
        "rotation_bytes": rotation_bytes,
        "mean_bytes": mean_bytes,
        "raw_bytes": raw_bytes,
        "storage_bytes": storage_bytes,
        "storage_gib": storage_bytes / 1024**3,
        "compression_ratio": ratio,
        "input_sha256": sha256_file(source_path),
        "dense_recon_sha256": sha256_file(recon_path),
        "compressed_ple_sha256": sha256_file(compressed_path) if compressed_path is not None else None,
        "compressed_ple_npz_total_bytes": compressed_path.stat().st_size if compressed_path is not None else None,
        "script_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": time.time() - t0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
