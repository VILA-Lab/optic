#!/usr/bin/env python3
"""Jointly refine the fixed 100x residual-VQ baseline by block coordinate descent.

This is the PLE adaptation of the public additive/local-search quantization
family: initialize from the greedy residual-VQ codec, then alternate
code assignment and codebook centroid updates while all other additive
codebooks remain active.  The procedure uses only the checkpoint PLE tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_BUILDERS = ROOT / "builders" / "baselines"
sys.path.insert(0, str(BASELINE_BUILDERS))

from baseline_common import PLE_KEY, RECON_KEY  # noqa: E402
from baseline_residual_vq import pack_nbit, unpack_nbit  # noqa: E402


FORMAT_ID = "joint_refined_additive_vq_npz"
METHOD_ID = "joint_refined_additive_quantization"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assign_and_accumulate(torch, x, recon, old_codes, old_centers, batch_size, device):
    rows, d = x.shape
    clusters = old_centers.shape[0]
    new_codes = torch.empty(rows, dtype=torch.int64)
    sums = torch.zeros((clusters, d), dtype=torch.float64)
    counts = torch.zeros(clusters, dtype=torch.long)
    centers_gpu = old_centers.to(device=device, dtype=torch.float32)
    center_norm = centers_gpu.square().sum(dim=1).unsqueeze(0)
    for start in range(0, rows, batch_size):
        stop = min(rows, start + batch_size)
        previous_ids = old_codes[start:stop].to(device)
        target = (
            x[start:stop].to(device)
            - recon[start:stop].to(device)
            + centers_gpu.index_select(0, previous_ids)
        )
        distances = target.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * (target @ centers_gpu.T)
        ids = distances.argmin(dim=1)
        new_codes[start:stop] = ids.cpu()
        target_cpu = target.to(device="cpu", dtype=torch.float64)
        ids_cpu = ids.cpu()
        sums.index_add_(0, ids_cpu, target_cpu)
        counts.index_add_(0, ids_cpu, torch.ones_like(ids_cpu, dtype=torch.long))
        del previous_ids, target, distances, ids, target_cpu, ids_cpu
    centers = old_centers.clone()
    nonempty = counts > 0
    centers[nonempty] = (sums[nonempty] / counts[nonempty, None]).to(torch.float32)
    # The deployed codebooks are FP16.  Keep the optimization in the exact
    # resident domain before updating the running reconstruction.
    centers = centers.to(torch.float16).to(torch.float32)
    return centers, new_codes, int(nonempty.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--it-weights", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--d-ple", type=int, required=True)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-recon", required=True)
    parser.add_argument("--save-compressed-ple-npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if args.sweeps != 2:
        raise SystemExit("the 100x baseline requires exactly two refinement sweeps")
    if args.seed != 0:
        raise SystemExit("the 100x baseline requires seed 0")
    started = time.time()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with safe_open(args.it_weights, framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(PLE_KEY).to(torch.float32)
    vocab = int(packed.shape[0])
    expected_shape = (vocab, args.num_layers * args.d_ple)
    if tuple(packed.shape) != expected_shape:
        raise SystemExit(f"checkpoint PLE shape {tuple(packed.shape)} != {expected_shape}")
    x = packed.reshape(vocab, args.num_layers, args.d_ple).permute(1, 0, 2).reshape(-1, args.d_ple).contiguous()
    del packed
    rows = int(x.shape[0])

    with np.load(args.source_artifact, allow_pickle=False) as archive:
        source_metadata = json.loads(str(archive["metadata"].item()))
        if source_metadata.get("compressed_ple_format_id") != "residual_vq_npz":
            raise SystemExit("--source-artifact is not the audited residual-VQ baseline")
        codebooks_np = archive["codebooks"].copy()
        m_count, clusters, d = codebooks_np.shape
        bits = int(source_metadata["bits"])
        if d != args.d_ple:
            raise SystemExit("source additive-codebook width does not match --d-ple")
        codes = [
            torch.from_numpy(unpack_nbit(archive[f"codes_m{stage:02d}"], bits, rows).astype(np.int64))
            for stage in range(m_count)
        ]
    codebooks = [torch.from_numpy(codebooks_np[stage].astype("float32")) for stage in range(m_count)]

    reconstruction = torch.zeros_like(x)
    for stage in range(m_count):
        for start in range(0, rows, args.batch_size):
            stop = min(rows, start + args.batch_size)
            reconstruction[start:stop] += codebooks[stage].index_select(0, codes[stage][start:stop])

    denominator = x.double().square().sum().item()
    curve = [
        {
            "sweep": 0,
            "stage": -1,
            "nmse": (x - reconstruction).double().square().sum().item() / max(denominator, 1e-30),
        }
    ]
    for sweep in range(args.sweeps):
        for stage in range(m_count):
            old_centers = codebooks[stage]
            old_codes = codes[stage]
            new_centers, new_codes, nonempty = _assign_and_accumulate(
                torch,
                x,
                reconstruction,
                old_codes,
                old_centers,
                args.batch_size,
                device,
            )
            for start in range(0, rows, args.batch_size):
                stop = min(rows, start + args.batch_size)
                reconstruction[start:stop] += new_centers.index_select(0, new_codes[start:stop])
                reconstruction[start:stop] -= old_centers.index_select(0, old_codes[start:stop])
            codebooks[stage] = new_centers
            codes[stage] = new_codes
            nmse = (x - reconstruction).double().square().sum().item() / max(denominator, 1e-30)
            curve.append({"sweep": sweep + 1, "stage": stage, "nmse": nmse, "nonempty": nonempty})
            print(f"[joint-aq] sweep={sweep + 1}/2 stage={stage + 1}/{m_count} nmse={nmse:.6f}", flush=True)

    recon_packed = torch.empty(expected_shape, dtype=torch.bfloat16)
    raw_error = 0.0
    for layer in range(args.num_layers):
        row_slice = slice(layer * vocab, (layer + 1) * vocab)
        layer_recon = reconstruction[row_slice].to(torch.bfloat16)
        recon_packed[:, layer * args.d_ple : (layer + 1) * args.d_ple] = layer_recon
        raw_error += (layer_recon.to(torch.float32) - x[row_slice]).double().square().sum().item()
    decoded_nmse = raw_error / max(denominator, 1e-30)
    recon_path = Path(args.save_recon)
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({RECON_KEY: recon_packed.contiguous()}, recon_path)

    codebooks_resident = torch.stack(codebooks).to(torch.float16).numpy()
    code_bytes = math.ceil(rows * m_count * bits / 8)
    codebook_bytes = int(codebooks_resident.nbytes)
    storage_bytes = code_bytes + codebook_bytes
    raw_bytes = rows * args.d_ple * 2
    metadata = {
        "compressed_ple_format_id": FORMAT_ID,
        "method": METHOD_ID,
        "kind": METHOD_ID,
        "schema_version": "1.0",
        "hidden_raw_copy_allowed": False,
        "full_ple_coverage": True,
        "M": m_count,
        "K": clusters,
        "bits": bits,
        "uses_coordinate_weights": False,
        "coordinate_weight_file_path": "",
        "ple_key": PLE_KEY,
        "ple_shape": list(expected_shape),
        "storage_bytes": storage_bytes,
        "seed": args.seed,
        "initialization": "audited_greedy_residual_vq_r100",
        "refinement": "two_sweep_block_coordinate_assignment_and_centroid_update",
        "calibration_corpus_used": False,
    }
    payload = {"metadata": np.array(json.dumps(metadata)), "codebooks": codebooks_resident}
    for stage in range(m_count):
        payload[f"codes_m{stage:02d}"] = pack_nbit(codes[stage].numpy().astype(np.uint16), bits)
    artifact_path = Path(args.save_compressed_ple_npz)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(artifact_path, **payload)

    result = {
        **metadata,
        "source_artifact": args.source_artifact,
        "source_artifact_sha256": sha256_file(args.source_artifact),
        "sweeps": args.sweeps,
        "optimization_curve": curve,
        "raw_nmse": decoded_nmse,
        "decoded_source_ple_nmse": decoded_nmse,
        "target_nmse": decoded_nmse,
        "code_bytes": code_bytes,
        "codebook_bytes": codebook_bytes,
        "raw_bytes": raw_bytes,
        "compression_ratio": raw_bytes / storage_bytes,
        "input_sha256": sha256_file(args.it_weights),
        "dense_recon_sha256": sha256_file(recon_path),
        "compressed_ple_sha256": sha256_file(artifact_path),
        "compressed_ple_npz_total_bytes": artifact_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"[joint-aq] decoded NMSE={decoded_nmse:.6f} storage={storage_bytes} "
        f"ratio={raw_bytes / storage_bytes:.4f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
