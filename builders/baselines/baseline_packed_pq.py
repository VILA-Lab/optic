#!/usr/bin/env python3
"""Fit product quantization to the layer-token embedding rows.

Each fixed coordinate subspace has its own k-means codebook."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from baseline_common import RECON_KEY, load_ple_tensor, require_uint16_code_count
from baseline_residual_vq import (
    DEFAULT_IT_WEIGHTS,
    benchmark_id_for_metrics_output,
    kmeans,
    pack_nbit,
    sha256_file,
    unpack_nbit,
)


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
        if metadata.get("compressed_ple_format_id") != "packed_pq_npz":
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
        resident_codebooks = torch.from_numpy(codebook_array.astype("float32", copy=False)).to(device)
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
                decoded = torch.empty((stop - start, d), dtype=torch.float32, device=device)
                for subspace in range(subspaces):
                    start_dim = subspace * d_sub
                    stop_dim = (subspace + 1) * d_sub
                    codes = resident_codes[subspace][row_slice].to(device)
                    decoded[:, start_dim:stop_dim] = resident_codebooks[subspace, codes]
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
                del decoded, decoded_bf16, saved, decoded_f32, saved_f32, diff_tensor
    return max_abs_diff


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
    p.add_argument("--num-subspaces", type=int, default=16)
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
    print(f"[packed-pq] loading PLE source={source_path}", flush=True)
    packed, ple_key, ple_shape = load_ple_tensor(
        safe_open, source_path, args.num_layers, args.d_ple, allow_recon_key=bool(args.source_ple)
    )
    packed = packed.to(torch.float32)

    vocab = packed.shape[0]
    layers = args.num_layers
    d = args.d_ple
    m_count = args.num_subspaces
    clusters = args.clusters
    rows = vocab * layers
    if d % m_count != 0:
        raise ValueError("d_ple must be divisible by num_subspaces")
    d_sub = d // m_count
    require_uint16_code_count("--clusters", clusters)
    bits = int(math.ceil(math.log2(clusters)))

    device = torch.device(args.device)
    print(
        f"[packed-pq] V={vocab} L={layers} d={d} rows={rows} M={m_count} "
        f"K={clusters} bits={bits}",
        flush=True,
    )
    t0 = time.time()

    x_flat = packed.reshape(vocab, layers, d).permute(1, 0, 2).reshape(rows, d).contiguous()
    del packed

    codebooks = np.empty((m_count, clusters, d_sub), dtype=np.float16)
    assignments: list[np.ndarray] = []
    recon_flat = torch.empty_like(x_flat)

    for m in range(m_count):
        print(f"[packed-pq] subspace {m + 1}/{m_count}", flush=True)
        start_dim = m * d_sub
        stop_dim = (m + 1) * d_sub
        sub = x_flat[:, start_dim:stop_dim].contiguous()
        centers, codes = kmeans(torch, sub, clusters, args.iters, args.batch_size, device, args.seed + 1009 * m)
        centers_resident = centers.to(torch.float16).to(torch.float32)
        recon_flat[:, start_dim:stop_dim] = centers_resident[codes.long()].to(torch.float32)
        codebooks[m] = centers_resident.to(torch.float16).numpy()
        assignments.append(codes.to(torch.uint16).numpy())
        del sub, centers, centers_resident, codes

    err = 0.0
    norm = 0.0
    recon_packed = torch.empty((vocab, layers * d), dtype=torch.bfloat16)
    for layer in range(layers):
        row_slice = slice(layer * vocab, (layer + 1) * vocab)
        ref = x_flat[row_slice]
        recon_layer = recon_flat[row_slice]
        recon_bf16 = recon_layer.to(torch.bfloat16)
        recon_packed[:, layer * d:(layer + 1) * d] = recon_bf16
        diff = recon_bf16.to(torch.float32) - ref
        err += diff.double().square().sum().item()
        norm += ref.double().square().sum().item()
        del ref, recon_layer, recon_bf16, diff
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

    code_bytes = math.ceil(rows * m_count * bits / 8)
    codebook_bytes = m_count * clusters * d_sub * 2
    storage_bytes = code_bytes + codebook_bytes
    raw_bytes = rows * d * 2
    ratio = raw_bytes / storage_bytes
    print(f"[packed-pq] decoded-source-PLE NMSE={decoded_source_ple_nmse:.6f}", flush=True)
    if decoded_original_checkpoint_ple_nmse is not None:
        print(f"[packed-pq] decoded-original-checkpoint-PLE NMSE={decoded_original_checkpoint_ple_nmse:.6f}", flush=True)
    print(f"[packed-pq] storage={storage_bytes} bytes ratio={ratio:.4f}x", flush=True)

    Path(args.save_recon).parent.mkdir(parents=True, exist_ok=True)
    save_file({"embed_tokens_per_layer.weight": recon_packed}, args.save_recon)
    compressed_path = Path(args.save_compressed_ple_npz) if args.save_compressed_ple_npz else None
    if compressed_path is not None:
        compressed_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "codebooks": codebooks,
            "metadata": np.array(json.dumps({
                "compressed_ple_format_id": "packed_pq_npz",
                "num_subspaces": m_count,
                "clusters": clusters,
                "bits": bits,
                "ple_key": ple_key,
                "ple_shape": ple_shape,
            })),
        }
        for m, codes in enumerate(assignments):
            payload[f"codes_m{m:02d}"] = pack_nbit(codes, bits)
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
            vocab=vocab,
            layers=layers,
            d=d,
            subspaces=m_count,
            clusters=clusters,
            bits=bits,
            eval_batch_size=args.eval_batch_size,
            atol=args.resident_roundtrip_atol,
            device=device,
        )

    out = {
        "benchmark": benchmark_id_for_metrics_output(output_path),
        "method": "packed_pq",
        "compressed_ple_format_id": "packed_pq_npz",
        "num_subspaces": m_count,
        "clusters": clusters,
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
