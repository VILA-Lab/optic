#!/usr/bin/env python3
"""Fit residual vector quantization to the layer-token embedding rows.

Each residual codebook models the error left by the preceding codebooks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from baseline_common import RECON_KEY, load_ple_tensor, require_uint16_code_count

DEFAULT_IT_WEIGHTS = (
    "<CACHE_DIR>/huggingface/hub/"
    "models--google--gemma-4-E2B-it/snapshots/"
    "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf/model.safetensors"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_id_for_metrics_output(output_path: Path) -> str | None:
    for parent in (output_path.parent, *output_path.parents):
        record_path = parent / "experiment.json"
        if not record_path.is_file():
            continue
        try:
            experiment = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"{record_path} is not valid JSON: {exc}") from exc
        benchmark_id = experiment.get("benchmark_id")
        if not isinstance(benchmark_id, str) or not benchmark_id:
            raise SystemExit(f"{record_path} benchmark_id must be a nonempty string")
        return benchmark_id
    return None


def unpack_nbit(values, bits: int, count: int):
    import numpy as np

    bit_array = np.unpackbits(values.astype("uint8", copy=False), bitorder="little")[: count * bits]
    bit_array = bit_array.reshape(count, bits)
    shifts = (1 << np.arange(bits, dtype=np.uint32))[None, :]
    return (bit_array.astype(np.uint32) * shifts).sum(axis=1).astype(np.uint16, copy=False)


def pack_nbit(values, bits):
    import numpy as np

    values = values.astype(np.uint16, copy=False)
    shifts = np.arange(bits, dtype=np.uint16)
    bit_matrix = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little")


def assign_chunks(torch, x_cpu, centers, batch_size, device):
    n = x_cpu.shape[0]
    assignments = torch.empty(n, dtype=torch.int64)
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        chunk = x_cpu[start:stop].to(device)
        dist = chunk.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * (chunk @ centers.T)
        assignments[start:stop] = dist.argmin(dim=1).cpu()
        del chunk, dist
    return assignments


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
        print(f"[rq] iter {it + 1}/{iters} nonempty={int(nonempty.sum().item())}/{k}", flush=True)
    return centers.cpu(), assign_chunks(torch, x_cpu, centers, batch_size, device)


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
    codebooks: int,
    clusters: int,
    bits: int,
    eval_batch_size: int,
    atol: float,
    device,
) -> float:
    with np.load(compressed_path, allow_pickle=False) as npz:
        if "metadata" not in npz.files:
            raise SystemExit(f"{compressed_path} missing metadata")
        metadata = json.loads(str(npz["metadata"].item()))
        if metadata.get("compressed_ple_format_id") != "residual_vq_npz":
            raise SystemExit(f"{compressed_path} compressed_ple_format_id mismatch")
        if int(metadata.get("M")) != codebooks or int(metadata.get("K")) != clusters or int(metadata.get("bits")) != bits:
            raise SystemExit(f"{compressed_path} metadata M/K/bits mismatch")
        codebook_array = npz["codebooks"]
        if tuple(codebook_array.shape) != (codebooks, clusters, d):
            raise SystemExit(f"{compressed_path} codebooks shape {tuple(codebook_array.shape)} != {(codebooks, clusters, d)}")
        resident_codebooks = torch.from_numpy(codebook_array.astype("float32", copy=False)).to(device)
        row_count = vocab * layers
        resident_codes = []
        for cb in range(codebooks):
            name = f"codes_m{cb:02d}"
            if name not in npz.files:
                raise SystemExit(f"{compressed_path} missing {name}")
            resident_codes.append(torch.from_numpy(unpack_nbit(npz[name], bits, row_count)).to(torch.long))
        coord_scales = None
        if bool(metadata.get("uses_coordinate_weights")):
            if "coord_weights_fp16" not in npz.files:
                raise SystemExit(f"{compressed_path} metadata declares coordinate weights but coord_weights_fp16 is missing")
            coord_weights = torch.from_numpy(npz["coord_weights_fp16"].astype("float32", copy=False))
            if tuple(coord_weights.shape) != (layers, d):
                raise SystemExit(f"{compressed_path} coord_weights_fp16 shape {tuple(coord_weights.shape)} != {(layers, d)}")
            coord_scales = coord_weights.sqrt().to(device)

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
                decoded = torch.zeros((stop - start, d), dtype=torch.float32, device=device)
                for cb in range(codebooks):
                    decoded += resident_codebooks[cb, resident_codes[cb][row_slice].to(device)]
                if coord_scales is not None:
                    decoded = decoded / coord_scales[layer]
                decoded_bf16 = decoded.to(torch.bfloat16).cpu()
                saved = recon_slice[start:stop, layer * d : (layer + 1) * d]
                decoded_f32 = decoded_bf16.to(torch.float32)
                saved_f32 = saved.to(torch.float32)
                diff_tensor = (decoded_f32 - saved_f32).abs()
                if not bool(torch.isfinite(decoded_f32).all().item()):
                    raise SystemExit("resident file reload/decode produced non-finite values")
                if not bool(torch.isfinite(saved_f32).all().item()):
                    raise SystemExit("saved BF16 dense PLE reconstruction contains non-finite values")
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
    p.add_argument("--codebooks", type=int, default=11)
    p.add_argument("--clusters", type=int, default=512)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--resident-roundtrip-atol", type=float, default=0.002)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-recon", required=True)
    p.add_argument("--save-compressed-ple-npz", default="")
    p.add_argument(
        "--coord-weights",
        default="",
        help="Optional .npy array with shape (L,d). Train in x*sqrt(weight) space and decode back.",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_path = args.source_ple or args.it_weights
    print(f"[rq] loading PLE source={source_path}", flush=True)
    packed, ple_key, ple_shape = load_ple_tensor(
        safe_open, source_path, args.num_layers, args.d_ple, allow_recon_key=bool(args.source_ple)
    )
    packed = packed.to(torch.float32)

    v = packed.shape[0]
    l_count = args.num_layers
    d = args.d_ple
    n = v * l_count
    m_count = args.codebooks
    k = args.clusters
    require_uint16_code_count("--clusters", k)
    bits = int(math.ceil(math.log2(k)))
    device = torch.device(args.device)
    print(f"[rq] V={v} L={l_count} d={d} rows={n} M={m_count} K={k} bits={bits}", flush=True)

    t0 = time.time()
    x_flat = packed.reshape(v, l_count, d).permute(1, 0, 2).reshape(n, d).contiguous()
    del packed
    coord_weights = None
    coord_scales = None
    coord_weight_bytes = 0
    if args.coord_weights:
        coord_weights_np = np.load(args.coord_weights).astype("float32")
        if tuple(coord_weights_np.shape) != (l_count, d):
            raise SystemExit(f"--coord-weights shape {coord_weights_np.shape} != {(l_count, d)}")
        coord_weights_np = coord_weights_np / max(float(coord_weights_np.mean()), 1e-30)
        coord_weights = torch.from_numpy(coord_weights_np).to(torch.float16).to(torch.float32)
        coord_scales = coord_weights.sqrt().to(torch.float32)
        coord_weight_bytes = l_count * d * 2
        for layer in range(l_count):
            rows = slice(layer * v, (layer + 1) * v)
            x_flat[rows] *= coord_scales[layer]
        print(
            f"[rq] weighted training coord_weights={args.coord_weights} "
            f"min={float(coord_weights.min()):.4g} max={float(coord_weights.max()):.4g}",
            flush=True,
        )
    raw_x_for_nmse = None
    if coord_scales is not None:
        raw_x_for_nmse = x_flat.clone()
        for layer in range(l_count):
            rows = slice(layer * v, (layer + 1) * v)
            raw_x_for_nmse[rows] /= coord_scales[layer]
    residual = x_flat.clone()
    all_codebooks = np.empty((m_count, k, d), dtype=np.float16)
    all_codes = []

    for cb in range(m_count):
        print(f"[rq] codebook {cb + 1}/{m_count}", flush=True)
        centers, assignments = kmeans(
            torch,
            residual,
            k,
            args.iters,
            args.batch_size,
            device,
            args.seed + 1009 * cb,
        )
        centers_decoded = centers.to(torch.float16).to(torch.float32)
        for start in range(0, n, args.batch_size):
            stop = min(n, start + args.batch_size)
            residual[start:stop] -= centers_decoded[assignments[start:stop].long()]
        all_codebooks[cb] = centers_decoded.to(torch.float16).numpy()
        all_codes.append(assignments.to(torch.uint16).numpy())
        err_now = residual.double().square().sum().item()
        norm_now = x_flat.double().square().sum().item()
        print(f"[rq] after codebook {cb + 1}: NMSE={err_now / max(norm_now, 1e-30):.6f}", flush=True)
        del centers, centers_decoded, assignments

    err = residual.double().square().sum().item()
    norm = x_flat.double().square().sum().item()
    training_space_nmse = err / max(norm, 1e-30)

    resident_codebooks = torch.from_numpy(all_codebooks.astype("float32"))
    resident_codes = [torch.from_numpy(c.astype("int64", copy=False)) for c in all_codes]
    recon_packed = torch.empty((v, l_count * d), dtype=torch.bfloat16)
    raw_err = 0.0
    raw_norm = 0.0
    for layer in range(l_count):
        rows = slice(layer * v, (layer + 1) * v)
        recon_layer = torch.zeros((v, d), dtype=torch.float32)
        for cb in range(m_count):
            recon_layer += resident_codebooks[cb, resident_codes[cb][rows]]
        if coord_scales is not None:
            recon_layer = recon_layer / coord_scales[layer]
            raw_ref = raw_x_for_nmse[rows]
        else:
            raw_ref = x_flat[rows]
        recon_bf16 = recon_layer.to(torch.bfloat16)
        recon_packed[:, layer * d:(layer + 1) * d] = recon_bf16
        diff = recon_bf16.to(torch.float32) - raw_ref
        raw_err += diff.double().square().sum().item()
        raw_norm += raw_ref.double().square().sum().item()
    Path(args.save_recon).parent.mkdir(parents=True, exist_ok=True)
    save_file({"embed_tokens_per_layer.weight": recon_packed}, args.save_recon)

    decoded_source_ple_nmse = raw_err / max(raw_norm, 1e-30)

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
    codebook_bytes = m_count * k * d * 2
    storage_bytes = code_bytes + codebook_bytes + coord_weight_bytes
    raw_bytes = n * d * 2
    ratio = raw_bytes / storage_bytes
    print(f"[rq] training-space NMSE={training_space_nmse:.6f}", flush=True)
    print(f"[rq] decoded-source-PLE NMSE={decoded_source_ple_nmse:.6f}", flush=True)
    if decoded_original_checkpoint_ple_nmse is not None:
        print(f"[rq] decoded-original-checkpoint-PLE NMSE={decoded_original_checkpoint_ple_nmse:.6f}", flush=True)
    print(f"[rq] storage={storage_bytes} bytes ratio={ratio:.4f}x", flush=True)

    compressed_path = Path(args.save_compressed_ple_npz) if args.save_compressed_ple_npz else None
    if compressed_path is not None:
        compressed_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "codebooks": all_codebooks,
            "metadata": np.array(json.dumps({
                "compressed_ple_format_id": "residual_vq_npz",
                "M": m_count,
                "K": k,
                "bits": bits,
                "uses_coordinate_weights": bool(args.coord_weights),
                "coordinate_weight_file_path": args.coord_weights,
                "ple_key": ple_key,
                "ple_shape": ple_shape,
            })),
        }
        if coord_weights is not None:
            payload["coord_weights_fp16"] = coord_weights.to(torch.float16).numpy()
        for cb, codes in enumerate(all_codes):
            payload[f"codes_m{cb:02d}"] = pack_nbit(codes, bits)
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
            codebooks=m_count,
            clusters=k,
            bits=bits,
            eval_batch_size=args.eval_batch_size,
            atol=args.resident_roundtrip_atol,
            device=device,
        )

    out = {
        "benchmark": benchmark_id_for_metrics_output(output_path),
        "method": "residual_vq",
        "compressed_ple_format_id": "residual_vq_npz",
        "codebooks": m_count,
        "clusters": k,
        "bits_per_code": bits,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "resident_roundtrip_atol": args.resident_roundtrip_atol,
        "seed": args.seed,
        "source_ple": args.source_ple or "",
        "coordinate_weight_file_path": args.coord_weights,
        "ple_key": ple_key,
        "ple_shape": list(ple_shape),
        "validated_full_ple_shape": True,
        "training_space_nmse": training_space_nmse,
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
        "coord_weight_bytes": coord_weight_bytes,
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
    with open(output_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
