#!/usr/bin/env python3
"""Fit randomized low-rank SVD to the full per-layer embedding matrix.

The approximation X = F @ G stores INT8 factors and FP16 scales."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baseline_common import load_ple_tensor

DEFAULT_IT_WEIGHTS = (
    "<CACHE_DIR>/huggingface/hub/"
    "models--google--gemma-4-E2B-it/snapshots/"
    "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf/model.safetensors"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--it-weights", default=DEFAULT_IT_WEIGHTS)
    p.add_argument("--num-layers", type=int, default=35)
    p.add_argument("--d-ple", type=int, default=256)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--power-iters", type=int, default=4)
    p.add_argument("--oversample", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--recon-chunk", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-recon", required=True)
    p.add_argument("--save-compressed-ple-npz", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    print("[qlr] loading PLE", flush=True)
    x_cpu, ple_key, ple_shape = load_ple_tensor(safe_open, args.it_weights, args.num_layers, args.d_ple)
    x_cpu = x_cpu.to(torch.float32)

    v = x_cpu.shape[0]
    l_count = args.num_layers
    d = args.d_ple
    n = l_count * d
    r = args.rank
    q = r + args.oversample
    device = torch.device(args.device)
    print(f"[qlr] V={v} L={l_count} d={d} rank={r} q={q}", flush=True)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    omega = torch.randn(n, q, device=device, dtype=torch.float32)
    y = torch.zeros(v, q, device=device, dtype=torch.float32)
    for s in range(0, v, args.chunk):
        e = min(v, s + args.chunk)
        y[s:e] = x_cpu[s:e].to(device) @ omega

    for _ in range(args.power_iters):
        b = torch.zeros(n, q, device=device, dtype=torch.float32)
        for s in range(0, v, args.chunk):
            e = min(v, s + args.chunk)
            b += x_cpu[s:e].to(device).T @ y[s:e]
        b, _ = torch.linalg.qr(b)
        y = torch.zeros(v, q, device=device, dtype=torch.float32)
        for s in range(0, v, args.chunk):
            e = min(v, s + args.chunk)
            y[s:e] = x_cpu[s:e].to(device) @ b

    q_basis, _ = torch.linalg.qr(y)
    del y, omega

    b_small = torch.zeros(q, n, device=device, dtype=torch.float32)
    for s in range(0, v, args.chunk):
        e = min(v, s + args.chunk)
        b_small += q_basis[s:e].T @ x_cpu[s:e].to(device)

    u_b, sigma, vt = torch.linalg.svd(b_small, full_matrices=False)
    u_b = u_b[:, :r]
    sigma = sigma[:r]
    vt = vt[:r, :]
    f = q_basis @ u_b
    f = f * sigma.unsqueeze(0)
    g = vt.reshape(r, l_count, d).contiguous()
    print(f"[qlr] rSVD done in {time.time() - t0:.1f}s", flush=True)

    f_scale = (f.abs().amax(dim=0) / 127.0).clamp_min(1e-12).to(torch.float16).to(torch.float32)
    q_f = torch.round(f / f_scale.unsqueeze(0)).clamp(-127, 127).to(torch.int8)

    g_scale = (g.abs().amax(dim=2) / 127.0).clamp_min(1e-12).to(torch.float16).to(torch.float32)
    q_g = torch.round(g / g_scale.unsqueeze(2)).clamp(-127, 127).to(torch.int8)

    factor_f_bytes = v * r
    factor_g_bytes = r * l_count * d
    scale_f_bytes = 2 * r
    scale_g_bytes = 2 * r * l_count
    storage_bytes = factor_f_bytes + factor_g_bytes + scale_f_bytes + scale_g_bytes
    raw_bytes = v * l_count * d * 2
    ratio = raw_bytes / storage_bytes

    print(f"[qlr] storage={storage_bytes} bytes ratio={ratio:.4f}x", flush=True)

    metadata = {
        "compressed_ple_format_id": "quant_lowrank_factor_npz",
        "ple_key": ple_key,
        "ple_shape": list(ple_shape),
        "rank": r,
        "factor_dtype": "int8",
        "scale_dtype": "float16",
    }
    Path(args.save_compressed_ple_npz).parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez(
        args.save_compressed_ple_npz,
        q_f_int8=q_f.cpu().numpy(),
        f_scale_fp16=f_scale.to(torch.float16).cpu().numpy(),
        q_g_int8=q_g.cpu().numpy(),
        g_scale_fp16=g_scale.to(torch.float16).cpu().numpy(),
        metadata=np.array(json.dumps(metadata)),
    )
    print(f"[qlr] saved compressed PLE NPZ {args.save_compressed_ple_npz}", flush=True)

    # Reload the saved compact artifact through the same compact codec used for
    # requested-row inference before producing the dense reconstruction.  Running
    # this path on CPU makes a later round trip independent of the reduction
    # order of a CUDA GEMM.
    del f, g, q_basis, u_b, sigma, vt, b_small, q_f, q_g, f_scale, g_scale
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from ple_eval.compact_adapter import load_compact_ple

    canonical = load_compact_ple(args.save_compressed_ple_npz, device="cpu")
    err = 0.0
    norm = 0.0
    recon_cpu = torch.empty((v, n), dtype=torch.bfloat16)
    with torch.inference_mode():
        for s in range(0, v, args.recon_chunk):
            e = min(v, s + args.recon_chunk)
            token_ids = torch.arange(s, e, dtype=torch.long)
            x_hat_bf16 = canonical(token_ids)
            x_ref = x_cpu[s:e]
            err += (x_ref - x_hat_bf16.to(torch.float32)).pow(2).sum().item()
            norm += x_ref.pow(2).sum().item()
            recon_cpu[s:e] = x_hat_bf16
    nmse = err / max(norm, 1e-30)
    print(f"[qlr] raw NMSE={nmse:.6f}", flush=True)

    # Record the difference between the CUDA and CPU decoders for reference.
    # Both decoders read the saved compact artifact.
    cross_backend_telemetry = None
    if device.type == "cuda":
        probe_ids = torch.tensor([0, 1, 2, 1024, 131072, 258884], dtype=torch.long)
        with torch.inference_mode():
            cpu_probe = canonical(probe_ids)
            cuda_codec = load_compact_ple(args.save_compressed_ple_npz, device=device)
            cuda_probe = cuda_codec(probe_ids.to(device)).cpu()
        code_distance = (
            cpu_probe.view(torch.int16).to(torch.int32)
            - cuda_probe.view(torch.int16).to(torch.int32)
        ).abs()
        cross_backend_telemetry = {
            "devices": ["cpu", str(device)],
            "requested_token_ids": probe_ids.tolist(),
            "different_elements": int((cpu_probe != cuda_probe).sum().item()),
            "max_abs_diff_after_bfloat16_materialization": float(
                (cpu_probe.float() - cuda_probe.float()).abs().max().item()
            ),
            "max_bfloat16_code_distance": int(code_distance.max().item()),
            "gate_role": "telemetry_only",
        }
        del cuda_codec, cuda_probe

    Path(args.save_recon).parent.mkdir(parents=True, exist_ok=True)
    save_file({"embed_tokens_per_layer.weight": recon_cpu}, args.save_recon)
    print(f"[qlr] saved {args.save_recon}", flush=True)

    out = {
        "method": "quant_lowrank_factor",
        "ple_key": ple_key,
        "ple_shape": list(ple_shape),
        "validated_full_ple_shape": list(ple_shape) == [262144, l_count * d],
        "source_ple": args.it_weights,
        "rank": r,
        "factor_dtype": "int8",
        "scale_dtype": "fp16_accounted",
        "power_iters": args.power_iters,
        "oversample": args.oversample,
        "seed": args.seed,
        "decoded_source_ple_nmse": nmse,
        "capability_materialization": {
            "source": "persisted_compact_artifact_reload",
            "codec": "ple_eval.compact_adapter.CompactQuantizedLowRank",
            "device": "cpu",
            "chunk_rows": args.recon_chunk,
        },
        "cross_backend_requested_row_telemetry": cross_backend_telemetry,
        "factor_f_bytes": factor_f_bytes,
        "factor_g_bytes": factor_g_bytes,
        "scale_f_bytes": scale_f_bytes,
        "scale_g_bytes": scale_g_bytes,
        "storage_bytes": storage_bytes,
        "storage_gib": storage_bytes / 1024**3,
        "compression_ratio": ratio,
        "elapsed_seconds": time.time() - t0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
