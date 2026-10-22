#!/usr/bin/env python3
"""Encode a fitted Additive Quantization model in parallel.

The encode command writes per-layer shards. The fan-in command combines them
into a compressed artifact, a dense reconstruction, and an evaluation configuration."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import os
import tempfile

# Directory for large intermediate files.
SCRATCH_DIR = os.path.realpath(os.environ.get("PLE_SCRATCH_DIR") or tempfile.gettempdir())

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_aq.py"
ONLINE_ADAPTER_PATH = ROOT / "scripts" / "aq_codec.py"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from aq_codec import (  # noqa: E402
    FORMAT_ID,
    PLE_KEY,
    RECON_KEY,
    artifact_sha256,
    audit_requested_rows,
    compact_resident_bytes,
    expected_resident_bytes,
    load_full_aq_ple,
    pack_nbit,
)
from ple_eval.gpu_binding import (  # noqa: E402
    bind_cuda_visible_device,
    validate_torch_cuda_binding,
)


AUTHOR_REPOSITORY = "https://github.com/arbabenko/Quantizations"
AUTHOR_COMMIT = "149e8d5226b96c36b55ef2423bcdebc94aec4099"
AUTHOR_FILE = "aqCoding.py"
PRIMARY_PAPER = "Babenko and Lempitsky, Additive Quantization for Extreme Vector Compression, CVPR 2014"
SHARD_DIRECTORY = "_parallel_final_encoding_v1"
SHARD_SCHEMA = "full-additive-quantization-ple-final-layer-shard-v1"
FAN_IN_SCHEMA = "full-additive-quantization-ple-parallel-finalization-v1"
ROW_CHUNK_TIMING_RATIOS = (20, 40)
ROW_CHUNK_TIMING_CANDIDATES = (128, 512, 1024)
ROW_CHUNK_TIMING_MEMORY_LIMIT_BYTES = 16 * 1024**3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    import torch

    tensor = value.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return array_sha256(tensor.view(torch.uint16).numpy())
    return array_sha256(tensor.numpy())


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_preexisting_training_trajectory(
    path: Path, expected_trajectory: list[dict[str, Any]],
) -> bool:
    """Accept only a pre-existing trajectory equal to the terminal checkpoint."""
    if not path.exists():
        return False
    try:
        parsed = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"pre-existing training trajectory is unreadable: {path}") from error
    require(
        parsed == expected_trajectory,
        "pre-existing training trajectory differs from the terminal fit checkpoint",
    )
    return True


def require_dev_shm(path: Path) -> None:
    resolved_parent = path.parent.resolve()
    require(
        str(resolved_parent) == SCRATCH_DIR or str(resolved_parent).startswith(SCRATCH_DIR + "/"),
        "capability reconstruction must remain under PLE_SCRATCH_DIR",
    )


def row_chunk_timing_schema(ratio: int) -> str:
    require(ratio in ROW_CHUNK_TIMING_RATIOS, "row-chunk timing ratio must be r20 or r40")
    return f"full-additive-quantization-r{ratio}-row-chunk-timing-v1"


def row_chunk_timing_receipt_path(output_root: Path, ratio: int) -> Path:
    row_chunk_timing_schema(ratio)
    return (output_root / SHARD_DIRECTORY / f"r{ratio}_row_chunk_timing_receipt.json").resolve()


def method_identity(config: dict[str, Any]) -> dict[str, str]:
    configured = config.get("method_identity", {})
    require(isinstance(configured, dict), "method_identity must be an object")
    return {
        "method_id_prefix": str(configured.get("method_id_prefix", "full_additive_quantization_ple")),
        "method_config_template": str(
            configured.get(
                "method_config_template",
                "configs/methods/full_additive_quantization_ple_{model}_{ratio_key}.json",
            )
        ),
        "publication_name": str(
            configured.get(
                "publication_name",
                "Full Additive Quantization (AQ) adapted to per-layer embeddings",
            )
        ),
    }


def method_config_path(contract: dict[str, Any]) -> Path:
    template = method_identity(contract["config"])["method_config_template"]
    rendered = template.format(model=contract["model_key"], ratio_key=contract["ratio_key"])
    path = Path(rendered)
    return path if path.is_absolute() else ROOT / path


def validate_contract(config_path: Path, model_key: str, ratio: int) -> dict[str, Any]:
    config = read_json(config_path)
    require(config.get("schema_version") == "full-additive-quantization-ple-w4-matrix-v1", "schema differs")
    require(config.get("status") == "fixed", "build matrix is not fixed")
    author = config.get("author_algorithm", {})
    require(author.get("repository") == AUTHOR_REPOSITORY, "AQ author repository differs")
    require(author.get("commit") == AUTHOR_COMMIT, "AQ author commit differs")
    require(author.get("source_file") == AUTHOR_FILE, "AQ author source file differs")
    require(model_key in config.get("models", {}), f"unknown model selector: {model_key}")
    model = dict(config["models"][model_key])
    ratio_key = f"r{ratio}"
    require(ratio_key in config.get("rates", {}), f"unsupported ratio selector: {ratio}")
    rate = dict(config["rates"][ratio_key])
    fit = config.get("fit", {})
    completion = fit.get("completion", {})
    require(int(fit.get("seed", -1)) == 0, "AQ seed must be zero")
    require(int(fit.get("fit_beam_size", 0)) == 16, "fit beam must remain 16")
    require(int(fit.get("final_beam_size", 0)) == 64, "final beam must remain 64")
    require(int(fit.get("row_chunk", 0)) > 0, "configured row chunk is invalid")
    require(completion.get("mode") == "fixed_outer_sweeps", "completion mode differs")
    require(int(completion.get("fixed_outer_sweeps", 0)) == 10, "fixed schedule must contain ten sweeps")
    require(completion.get("terminal_status") == "completed_fixed_schedule", "terminal status differs")
    require(completion.get("assignment_churn_role") == "diagnostic_only", "assignment churn became a gate")
    require(completion.get("relative_improvement_role") == "diagnostic_only", "objective change became a gate")
    vocab, layers, d_ple = int(model["vocab_size"]), int(model["num_layers"]), int(model["d_ple"])
    rows = vocab * layers
    dense_bytes = rows * d_ple * 2
    require(int(model["dense_ple_bytes"]) == dense_bytes, "dense PLE byte contract differs")
    weights = Path(model["snapshot"]) / str(model["weight_file"])
    require(weights.is_file(), f"source weights are missing: {weights}")
    m_count, clusters, bits = int(rate["M"]), int(rate["K"]), int(rate["bits"])
    resident_bytes = expected_resident_bytes(rows, d_ple, m_count, clusters, bits)
    require(
        int(rate["expected_resident_bytes_by_model"][model_key]) == resident_bytes,
        "resident-byte contract differs",
    )
    return {
        "config": config,
        "config_path": config_path,
        "model": model,
        "model_key": model_key,
        "ratio": ratio,
        "ratio_key": ratio_key,
        "rate": rate,
        "fit": fit,
        "weights": weights,
        "vocab": vocab,
        "layers": layers,
        "d_ple": d_ple,
        "rows": rows,
        "dense_bytes": dense_bytes,
        "resident_bytes": resident_bytes,
        "achieved_ratio": dense_bytes / resident_bytes,
    }


def deterministic_fit_row_ids(rows: int, requested: int, seed: int) -> Any:
    import torch

    count = min(rows, requested)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(rows, generator=generator)[:count].sort().values


def expected_checkpoint_binding(contract: dict[str, Any], row_ids: Any) -> dict[str, Any]:
    model, rate, fit = contract["model"], contract["rate"], contract["fit"]
    binding = {
        "schema_version": "full-additive-quantization-ple-fit-binding-v1",
        "builder_sha256": sha256_file(BUILDER_PATH),
        "build_config_path": str(contract["config_path"].resolve()),
        "build_config_sha256": sha256_file(contract["config_path"]),
        "model_selector": contract["model_key"],
        "ratio_bucket": contract["ratio_key"],
        "seed": int(fit["seed"]),
        "source_model_id": model["model_id"],
        "source_revision": model["revision"],
        "source_weights_path": str(contract["weights"].resolve()),
        "source_weights_sha256": sha256_file(contract["weights"]),
        "fit_row_count": int(row_ids.numel()),
        "fit_row_ids_sha256": array_sha256(row_ids.numpy()),
        "assignment_beams": {
            "fit": int(fit["fit_beam_size"]),
            "final_encoding": int(fit["final_beam_size"]),
        },
        "artifact_geometry": {
            "rows": int(contract["rows"]),
            "num_layers": int(contract["layers"]),
            "d_ple": int(contract["d_ple"]),
            "M": int(rate["M"]),
            "K": int(rate["K"]),
            "bits": int(rate["bits"]),
            "resident_bytes": int(contract["resident_bytes"]),
        },
    }
    return {**binding, "binding_sha256": canonical_json_sha256(binding)}


def load_terminal_checkpoint(contract: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    require(checkpoint_path.is_file(), f"fit checkpoint is missing: {checkpoint_path}")
    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            require(
                set(checkpoint.files) == {"fit_row_ids", "metadata", "codebooks_fp16", "fit_codes"},
                "fit checkpoint is partial or has unexpected arrays",
            )
            row_ids = np.asarray(checkpoint["fit_row_ids"])
            codebooks = np.asarray(checkpoint["codebooks_fp16"])
            fit_codes = np.asarray(checkpoint["fit_codes"])
            metadata = json.loads(str(checkpoint["metadata"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"fit checkpoint is corrupt: {checkpoint_path}") from error
    require(metadata.get("schema_version") == "full-additive-quantization-ple-resumable-checkpoint-v2", "checkpoint schema differs")
    require(metadata.get("final_artifact") is False, "fit checkpoint is incorrectly marked final")
    expected_rows = deterministic_fit_row_ids(
        int(contract["rows"]), int(contract["fit"]["fit_rows"]), int(contract["fit"]["seed"])
    )
    binding = expected_checkpoint_binding(contract, expected_rows)
    require(metadata.get("binding") == binding, "fit checkpoint binding differs")
    hashes = metadata.get("content_hashes", {})
    trajectory = metadata.get("trajectory")
    require(hashes.get("fit_row_ids") == array_sha256(row_ids), "fit row IDs are corrupt")
    require(hashes.get("codebooks_fp16") == array_sha256(codebooks), "fit codebooks are corrupt")
    require(hashes.get("fit_codes") == array_sha256(fit_codes), "fit codes are corrupt")
    require(isinstance(trajectory, list) and len(trajectory) == 11, "terminal ten-sweep trajectory differs")
    require(hashes.get("trajectory") == canonical_json_sha256(trajectory), "fit trajectory is corrupt")
    require(array_sha256(row_ids) == array_sha256(expected_rows.numpy()), "fit row partition differs")
    m_count, clusters = int(contract["rate"]["M"]), int(contract["rate"]["K"])
    require(codebooks.dtype == np.float16, "fit codebooks are not FP16 deployment values")
    require(tuple(codebooks.shape) == (m_count, clusters, int(contract["d_ple"])), "codebook geometry differs")
    require(tuple(fit_codes.shape) == (row_ids.shape[0], m_count), "fit-code geometry differs")
    require(bool(np.issubdtype(fit_codes.dtype, np.integer)), "fit codes are not integers")
    require(bool((fit_codes >= 0).all() and (fit_codes < clusters).all()), "fit codes are out of range")
    require(metadata.get("stage") == "fit_completed_fixed_schedule", "checkpoint is not at the terminal fit boundary")
    require(metadata.get("training_status") == "completed_fixed_schedule", "fit did not complete the fixed schedule")
    require(metadata.get("status_reason") == "author_fixed_outer_sweep_schedule_completed", "fit terminal reason differs")
    require(int(metadata.get("next_sweep", -1)) == 11, "terminal next-sweep boundary differs")
    require(int(trajectory[-1].get("sweep", -1)) == 10, "terminal trajectory sweep differs")
    elapsed = float(metadata.get("fit_elapsed_seconds", float("nan")))
    require(math.isfinite(elapsed) and elapsed >= 0, "fit elapsed time is invalid")
    return {
        "codebooks": torch.from_numpy(codebooks.copy()),
        "fit_row_ids_raw_sha256": hashlib.sha256(np.ascontiguousarray(row_ids).tobytes()).hexdigest(),
        "trajectory": trajectory,
        "training_status": metadata["training_status"],
        "status_reason": metadata["status_reason"],
        "fit_elapsed_seconds": elapsed,
        "binding": binding,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def validate_method_contract(contract: dict[str, Any], output_root: Path, reconstruction_path: Path) -> dict[str, Any]:
    method_path = method_config_path(contract)
    method = read_json(method_path)
    expected_id = (
        f"{method_identity(contract['config'])['method_id_prefix']}_{contract['model_key']}_"
        f"r{contract['ratio']}_seed{int(contract['fit']['seed'])}"
    )
    require(method.get("status") in {"build_required", "ready_for_evaluation"}, "method status cannot be finalized")
    require(method.get("method_id") == expected_id, "method identity differs")
    require(method.get("method_family") == "full_additive_quantization", "method family differs")
    require(method.get("ratio_bucket") == contract["ratio_key"], "method ratio bucket differs")
    require(int(method.get("seed", -1)) == int(contract["fit"]["seed"]), "method seed differs")
    require(int(method.get("fit_assignment_beam_size", 0)) == 16, "method fit beam differs")
    require(int(method.get("final_encoding_beam_size", 0)) == 64, "method final beam differs")
    require(int(method.get("ple_resident_bytes", -1)) == int(contract["resident_bytes"]), "method resident bytes differ")
    require(float(method.get("ple_compression_ratio", -1.0)) == float(contract["achieved_ratio"]), "method compression ratio differs")
    require(Path(method["build_matrix_config"]).resolve() == contract["config_path"].resolve(), "method build config differs")
    require(Path(method["compressed_artifact_path"]).resolve() == (output_root / "compressed_ple.npz").resolve(), "output root differs from method contract")
    require(Path(method["training_metrics_path"]).resolve() == (output_root / "summary.json").resolve(), "summary path differs")
    require(Path(method["training_trajectory_path"]).resolve() == (output_root / "training_trajectory.jsonl").resolve(), "trajectory path differs")
    require(Path(method["roundtrip_receipt_path"]).resolve() == (output_root / "requested_row_roundtrip_receipt.json").resolve(), "roundtrip path differs")
    require(Path(method["reconstruction_path"]).resolve() == reconstruction_path.resolve(), "reconstruction path differs from method contract")
    require(Path(method["runtime_receipt_path"]).resolve() == (output_root / "requested_row_runtime_receipt.json").resolve(), "runtime path differs")
    require(Path(method["online_adapter_path"]).resolve() == ONLINE_ADAPTER_PATH.resolve(), "online adapter differs")
    return method


def assert_no_competing_builder(output_root: Path) -> None:
    """Refuse a race with the original builder without signaling any process."""
    current = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == current:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            decoded = [item.decode("utf-8", errors="replace") for item in argv if item]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(Path(value).name == BUILDER_PATH.name for value in decoded):
            continue
        for index, value in enumerate(decoded[:-1]):
            if value == "--output-root" and Path(decoded[index + 1]).resolve() == output_root.resolve():
                raise RuntimeError(
                    f"original builder PID {entry.name} still owns this output root; wait for its natural exit"
                )


def _candidate_unique_topk(torch: Any, candidate_codes: Any, candidate_loss: Any, beam_size: int) -> Any:
    batch, candidates, _ = candidate_codes.shape
    selected_codes = torch.full(
        (batch, beam_size, candidate_codes.shape[-1]),
        -2,
        device=candidate_codes.device,
        dtype=candidate_codes.dtype,
    )
    selected_indices = torch.zeros((batch, beam_size), device=candidate_codes.device, dtype=torch.long)
    selected_valid = torch.zeros((batch, beam_size), device=candidate_codes.device, dtype=torch.bool)
    counts = torch.zeros(batch, device=candidate_codes.device, dtype=torch.long)
    rows = torch.arange(batch, device=candidate_codes.device)
    for rank in range(candidates):
        current = candidate_codes[:, rank]
        duplicate = ((selected_codes == current[:, None, :]).all(dim=-1) & selected_valid).any(dim=1)
        take = (~duplicate) & (counts < beam_size) & torch.isfinite(candidate_loss[:, rank])
        if bool(take.any()):
            take_rows = rows[take]
            slots = counts[take]
            selected_codes[take_rows, slots] = current[take]
            selected_indices[take_rows, slots] = rank
            selected_valid[take_rows, slots] = True
            counts[take] += 1
        if bool((counts == beam_size).all()):
            break
    require(bool((counts == beam_size).all()), "beam expansion lacked enough unique assignments")
    return selected_indices


def _beam_encode_chunk(torch: Any, x: Any, codebooks: Any, beam_size: int) -> tuple[Any, Any]:
    batch, d_ple = x.shape
    m_count, clusters, codebook_d = codebooks.shape
    require(d_ple == codebook_d, "beam input/codebook width differs")
    beam = min(beam_size, m_count * clusters)
    codebook_norm = codebooks.square().sum(dim=-1)
    singleton = x.square().sum(dim=-1)[:, None, None] + codebook_norm[None, :, :]
    singleton = singleton - 2.0 * torch.einsum("bd,mkd->bmk", x, codebooks)
    losses, flat = singleton.reshape(batch, -1).topk(beam, dim=1, largest=False, sorted=True)
    books = torch.div(flat, clusters, rounding_mode="floor")
    words = flat.remainder(clusters)
    codes = torch.full((batch, beam, m_count), -1, device=x.device, dtype=torch.int16)
    codes.scatter_(2, books[..., None], words.to(torch.int16)[..., None])
    reconstruction = codebooks[books, words]
    for depth in range(2, m_count + 1):
        target_beam = min(beam_size, math.comb(m_count, depth) * (clusters**depth))
        residual = x[:, None, :] - reconstruction
        expansion = losses[:, :, None, None] + codebook_norm[None, None, :, :]
        expansion = expansion - 2.0 * torch.einsum("bhd,mkd->bhmk", residual, codebooks)
        expansion.masked_fill_((codes >= 0)[:, :, :, None], float("inf"))
        raw_count = min(expansion.shape[1] * m_count * clusters, target_beam * depth)
        raw_loss, raw_flat = expansion.reshape(batch, -1).topk(raw_count, dim=1, largest=False, sorted=True)
        parent = torch.div(raw_flat, m_count * clusters, rounding_mode="floor")
        remainder = raw_flat.remainder(m_count * clusters)
        new_books = torch.div(remainder, clusters, rounding_mode="floor")
        new_words = remainder.remainder(clusters)
        candidate_codes = codes.gather(1, parent[..., None].expand(-1, -1, m_count)).clone()
        candidate_codes.scatter_(2, new_books[..., None], new_words.to(torch.int16)[..., None])
        chosen = _candidate_unique_topk(torch, candidate_codes, raw_loss, target_beam)
        codes = candidate_codes.gather(1, chosen[..., None].expand(-1, -1, m_count))
        losses = raw_loss.gather(1, chosen)
        chosen_parent = parent.gather(1, chosen)
        chosen_books = new_books.gather(1, chosen)
        chosen_words = new_words.gather(1, chosen)
        parent_reconstruction = reconstruction.gather(
            1, chosen_parent[..., None].expand(-1, -1, d_ple)
        )
        reconstruction = parent_reconstruction + codebooks[chosen_books, chosen_words]
        beam = target_beam
    exact_loss = (x - reconstruction[:, 0]).square().sum(dim=-1)
    return codes[:, 0].to(torch.int64), exact_loss


def beam_encode_rows(x: Any, codebooks: Any, *, beam_size: int, row_chunk: int, device: str) -> tuple[Any, Any]:
    import torch

    require(x.ndim == 2 and codebooks.ndim == 3, "beam geometry differs")
    require(row_chunk > 0, "row chunk must be positive")
    target = torch.device(device)
    resident = codebooks.to(device=target, dtype=torch.float32)
    codes = torch.empty((x.shape[0], codebooks.shape[0]), dtype=torch.int64)
    losses = torch.empty(x.shape[0], dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, x.shape[0], row_chunk):
            stop = min(x.shape[0], start + row_chunk)
            chunk_codes, chunk_losses = _beam_encode_chunk(
                torch, x[start:stop].to(device=target, dtype=torch.float32), resident, beam_size
            )
            codes[start:stop] = chunk_codes.cpu()
            losses[start:stop] = chunk_losses.cpu()
    return codes, losses


def decode_codes(codes: Any, codebooks: Any, *, row_chunk: int, device: str) -> Any:
    import torch

    target = torch.device(device)
    resident = codebooks.to(device=target, dtype=torch.float32)
    output = torch.empty((codes.shape[0], codebooks.shape[2]), dtype=torch.bfloat16)
    with torch.inference_mode():
        for start in range(0, codes.shape[0], row_chunk):
            stop = min(codes.shape[0], start + row_chunk)
            chosen = codes[start:stop].to(target)
            decoded = torch.zeros((stop - start, codebooks.shape[2]), device=target, dtype=torch.float32)
            for stage in range(codebooks.shape[0]):
                decoded.add_(resident[stage].index_select(0, chosen[:, stage]))
            output[start:stop] = decoded.to(torch.bfloat16).cpu()
    return output


def prove_row_chunk_equivalence(
    device: str, candidate_chunk: int, reference_chunk: int = 128,
) -> dict[str, Any]:
    """Direct exact proof that batching leaves each synthetic row unchanged."""
    import torch

    require(0 < candidate_chunk <= 8192, "row-chunk proof supports candidate chunks from 1 through 8192")
    require(reference_chunk > 0, "row-chunk proof reference must be positive")
    proof_rows = max(candidate_chunk, reference_chunk) + 3
    generator = torch.Generator(device="cpu").manual_seed(20260903)
    x = torch.randn((proof_rows, 7), generator=generator, dtype=torch.float32)
    codebooks = torch.randn((3, 5, 7), generator=generator, dtype=torch.float32).to(torch.float16)
    reference_codes, reference_losses = beam_encode_rows(
        x, codebooks, beam_size=16, row_chunk=reference_chunk, device=device
    )
    candidate_codes, candidate_losses = beam_encode_rows(
        x, codebooks, beam_size=16, row_chunk=candidate_chunk, device=device
    )
    reference_reconstruction = decode_codes(
        reference_codes, codebooks, row_chunk=reference_chunk, device=device
    )
    candidate_reconstruction = decode_codes(
        candidate_codes, codebooks, row_chunk=candidate_chunk, device=device
    )
    require(bool(torch.equal(reference_codes, candidate_codes)), "row-chunk proof changed a code tuple")
    require(bool(torch.equal(reference_losses, candidate_losses)), "row-chunk proof changed a per-row error")
    require(bool(torch.equal(reference_reconstruction, candidate_reconstruction)), "row-chunk proof changed reconstruction")
    return {
        "schema_version": "full-aq-row-chunk-equivalence-proof-v1",
        "status": "passed",
        "device": str(device),
        "reference_row_chunk": int(reference_chunk),
        "candidate_row_chunk": int(candidate_chunk),
        "synthetic_rows": int(x.shape[0]),
        "codes_exact": True,
        "per_row_error_exact": True,
        "bfloat16_reconstruction_exact": True,
        "reason": "beam assignment has no cross-row state; chunks partition only the leading row dimension",
    }


def select_row_chunk(measurements: Iterable[dict[str, Any]]) -> int:
    """Choose the execution row chunk from the 128, 512 and 1024 timings."""
    by_chunk = {int(value["row_chunk"]): value for value in measurements}
    require(set(by_chunk) == set(ROW_CHUNK_TIMING_CANDIDATES), "row-chunk timing candidate set differs")

    def eligible(chunk: int) -> bool:
        value = by_chunk[chunk]
        return bool(
            value.get("status") == "passed"
            and value.get("codes_exact") is True
            and value.get("per_row_loss_exact") is True
            and value.get("bfloat16_reconstruction_exact") is True
            and int(value.get("peak_reserved_bytes", ROW_CHUNK_TIMING_MEMORY_LIMIT_BYTES + 1))
            <= ROW_CHUNK_TIMING_MEMORY_LIMIT_BYTES
        )

    require(eligible(128), "row-chunk 128 reference is not eligible")
    if not eligible(512):
        return 128
    elapsed_128 = float(by_chunk[128]["elapsed_seconds"])
    elapsed_512 = float(by_chunk[512]["elapsed_seconds"])
    require(math.isfinite(elapsed_128) and elapsed_128 > 0, "row-chunk 128 timing is invalid")
    require(math.isfinite(elapsed_512) and elapsed_512 > 0, "row-chunk 512 timing is invalid")
    if not eligible(1024):
        return 512 if elapsed_512 < elapsed_128 else 128
    elapsed_1024 = float(by_chunk[1024]["elapsed_seconds"])
    require(math.isfinite(elapsed_1024) and elapsed_1024 > 0, "row-chunk 1024 timing is invalid")
    if elapsed_1024 <= 0.90 * elapsed_512 and elapsed_1024 < elapsed_128:
        return 1024
    return 512 if elapsed_512 < elapsed_128 else 128


def row_chunk_timing(args: argparse.Namespace) -> None:
    """Time one real layer without publishing it, then fix the execution row chunk."""
    require(args.device == "cuda:0", "Scheduler-owned row-chunk timing must address its sole visible GPU as cuda:0")
    device_binding = bind_cuda_visible_device(
        os.environ,
        logical_gpu=0,
        expected_physical_gpu_uuid=args.expected_physical_gpu_uuid,
    )
    import torch
    from safetensors import safe_open

    gpu_binding_attestation = validate_torch_cuda_binding(torch, device_binding)

    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    reconstruction_path = Path(args.save_recon).resolve()
    receipt_path = Path(args.receipt).resolve()
    contract = validate_contract(config_path, args.model, args.ratio)
    timing_schema = row_chunk_timing_schema(contract["ratio"])
    ratio_bucket = contract["ratio_key"]
    require_dev_shm(reconstruction_path)
    method = validate_method_contract(contract, output_root, reconstruction_path)
    require(method.get("status") == "build_required", "method config was already written")
    require(
        receipt_path == row_chunk_timing_receipt_path(output_root, contract["ratio"]),
        "row-chunk timing receipt path differs from the ratio-scoped contract",
    )
    require(
        isinstance(args.expected_physical_gpu_uuid, str)
        and args.expected_physical_gpu_uuid.startswith("GPU-"),
        "expected physical GPU UUID is invalid",
    )
    assert_no_competing_builder(output_root)
    require(not (output_root / "compressed_ple.npz").exists(), "compact artifact already exists")
    checkpoint = load_terminal_checkpoint(contract, output_root / "fit_checkpoint.npz")
    finalizer_sha = sha256_file(Path(__file__))
    if receipt_path.exists():
        existing = read_json(receipt_path)
        require(existing.get("schema_version") == timing_schema, "row-chunk timing schema differs")
        require(existing.get("status") == "passed", "row-chunk timing is not complete")
        require(existing.get("model_selector") == contract["model_key"], "row-chunk timing model differs")
        require(existing.get("ratio_bucket") == ratio_bucket, "row-chunk timing ratio differs")
        require(existing.get("fit_checkpoint_sha256") == checkpoint["checkpoint_sha256"], "row-chunk timing checkpoint differs")
        require(existing.get("binding_sha256") == checkpoint["binding"]["binding_sha256"], "row-chunk timing binding differs")
        require(existing.get("finalizer_sha256") == finalizer_sha, "row-chunk timing finalizer differs")
        require(
            existing.get("expected_physical_gpu_uuid") == args.expected_physical_gpu_uuid,
            "row-chunk timing physical GPU UUID differs",
        )
        require(int(existing.get("selected_row_chunk", 0)) in ROW_CHUNK_TIMING_CANDIDATES, "selected row chunk differs")
        print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
        return

    device = torch.device(args.device)
    require(device.type == "cuda", "row-chunk timing requires a CUDA device")
    with safe_open(str(contract["weights"]), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(PLE_KEY)
        require(
            tuple(tensor_slice.get_shape()) == (contract["vocab"], contract["layers"] * contract["d_ple"]),
            "checkpoint PLE shape differs",
        )
        source = tensor_slice[:, : contract["d_ple"]].float().contiguous()
    codebooks = checkpoint["codebooks"]
    measurements: list[dict[str, Any]] = []
    reference_codes = reference_losses = reference_reconstruction = None
    for row_chunk in ROW_CHUNK_TIMING_CANDIDATES:
        proof = prove_row_chunk_equivalence(args.device, row_chunk)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            codes, losses = beam_encode_rows(
                source, codebooks, beam_size=64, row_chunk=row_chunk, device=args.device
            )
            reconstruction = decode_codes(codes, codebooks, row_chunk=row_chunk, device=args.device)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            if row_chunk == 128:
                reference_codes = codes
                reference_losses = losses
                reference_reconstruction = reconstruction
            require(reference_codes is not None, "row-chunk reference was not measured first")
            measurement = {
                "row_chunk": row_chunk,
                "status": "passed",
                "elapsed_seconds": elapsed,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "within_memory_limit": peak_reserved <= ROW_CHUNK_TIMING_MEMORY_LIMIT_BYTES,
                "codes_exact": bool(torch.equal(codes, reference_codes)),
                "per_row_loss_exact": bool(torch.equal(losses, reference_losses)),
                "bfloat16_reconstruction_exact": bool(
                    torch.equal(reconstruction, reference_reconstruction)
                ),
                "codes_sha256": tensor_sha256(codes),
                "per_row_loss_sha256": tensor_sha256(losses),
                "reconstruction_sha256": tensor_sha256(reconstruction),
                "synthetic_equivalence_proof": proof,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            measurement = {
                "row_chunk": row_chunk,
                "status": "cuda_out_of_memory",
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "within_memory_limit": False,
                "codes_exact": False,
                "per_row_loss_exact": False,
                "bfloat16_reconstruction_exact": False,
                "synthetic_equivalence_proof": proof,
            }
        measurements.append(measurement)
    selected = select_row_chunk(measurements)
    receipt = {
        "schema_version": timing_schema,
        "status": "passed",
        "final_artifact": False,
        "publishable": False,
        "model_selector": contract["model_key"],
        "ratio_bucket": ratio_bucket,
        "layer_index": 0,
        "candidate_row_chunks": list(ROW_CHUNK_TIMING_CANDIDATES),
        "memory_limit_bytes": ROW_CHUNK_TIMING_MEMORY_LIMIT_BYTES,
        "selection_rule": (
            "require exact codes, per-row loss, and BF16 reconstruction versus row_chunk=128 and "
            "peak reserved CUDA memory <=16GiB; select 1024 only when it is >=10% faster than "
            "eligible 512, otherwise select faster eligible 512 versus 128, else 128"
        ),
        "selected_row_chunk": selected,
        "measurements": measurements,
        "fit_checkpoint_path": str(output_root / "fit_checkpoint.npz"),
        "fit_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "binding_sha256": checkpoint["binding"]["binding_sha256"],
        "build_config_sha256": checkpoint["binding"]["build_config_sha256"],
        "original_builder_sha256": checkpoint["binding"]["builder_sha256"],
        "finalizer_sha256": finalizer_sha,
        "source_weights_sha256": checkpoint["binding"]["source_weights_sha256"],
        "source_layer_sha256": tensor_sha256(source),
        "expected_physical_gpu_uuid": args.expected_physical_gpu_uuid,
        "gpu_binding_attestation": gpu_binding_attestation,
        "requested_device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "benchmark_inputs_used": False,
        "quality_metrics_used": False,
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def shard_paths(output_root: Path, layer: int) -> tuple[Path, Path, Path]:
    directory = output_root / SHARD_DIRECTORY / "layers"
    stem = f"layer_{layer:03d}"
    return directory / f"{stem}.safetensors", directory / f"{stem}.json", directory / f"{stem}.lock"


def validate_layer_indices(receipts: Iterable[dict[str, Any]], layers: int) -> list[dict[str, Any]]:
    values = list(receipts)
    indices = [int(value.get("layer_index", -1)) for value in values]
    require(len(indices) == layers, f"need exactly {layers} layer receipts, found {len(indices)}")
    require(len(set(indices)) == len(indices), "layer receipts contain a duplicate layer")
    require(set(indices) == set(range(layers)), "layer receipt set is incomplete or out of range")
    return sorted(values, key=lambda value: int(value["layer_index"]))


def validate_shard(
    contract: dict[str, Any], checkpoint: dict[str, Any], output_root: Path, layer: int,
    *, expected_finalizer_sha256: str,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    shard_path, receipt_path, _ = shard_paths(output_root, layer)
    require(shard_path.is_file() and receipt_path.is_file(), f"layer {layer} shard is incomplete")
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == SHARD_SCHEMA, f"layer {layer} receipt schema differs")
    require(receipt.get("status") == "completed_layer_shard", f"layer {layer} status differs")
    require(receipt.get("final_artifact") is False, f"layer {layer} is incorrectly marked final")
    require(receipt.get("publishable") is False, f"layer {layer} is incorrectly marked publishable")
    require(int(receipt.get("layer_index", -1)) == layer, f"layer {layer} receipt index differs")
    require(receipt.get("binding_sha256") == checkpoint["binding"]["binding_sha256"], f"layer {layer} binding differs")
    require(receipt.get("fit_checkpoint_sha256") == checkpoint["checkpoint_sha256"], f"layer {layer} checkpoint differs")
    require(receipt.get("original_builder_sha256") == checkpoint["binding"]["builder_sha256"], f"layer {layer} builder differs")
    require(receipt.get("finalizer_sha256") == expected_finalizer_sha256, f"layer {layer} finalizer differs")
    require(int(receipt.get("final_encoding_beam_size", 0)) == 64, f"layer {layer} final beam differs")
    require(receipt.get("shard_file_sha256") == sha256_file(shard_path), f"layer {layer} shard file is corrupt")
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        require(set(handle.keys()) == {"codes", "reconstruction"}, f"layer {layer} shard tensors differ")
        codes = handle.get_tensor("codes")
        reconstruction = handle.get_tensor("reconstruction")
    require(codes.dtype == torch.uint8, f"layer {layer} codes are not uint8")
    require(tuple(codes.shape) == (contract["vocab"], int(contract["rate"]["M"])), f"layer {layer} code shape differs")
    require(reconstruction.dtype == torch.bfloat16, f"layer {layer} reconstruction is not BF16")
    require(tuple(reconstruction.shape) == (contract["vocab"], contract["d_ple"]), f"layer {layer} reconstruction shape differs")
    require(receipt.get("codes_sha256") == tensor_sha256(codes), f"layer {layer} codes are corrupt")
    require(receipt.get("reconstruction_sha256") == tensor_sha256(reconstruction), f"layer {layer} reconstruction is corrupt")
    for field in ("assignment_objective_sse", "reconstruction_sse", "source_squared_norm", "elapsed_seconds"):
        value = float(receipt.get(field, float("nan")))
        require(math.isfinite(value) and value >= 0, f"layer {layer} {field} is invalid")
    return receipt


def shallow_validate_existing_shard(
    checkpoint: dict[str, Any], output_root: Path, layer: int, finalizer_sha: str,
) -> None:
    shard_path, receipt_path, _ = shard_paths(output_root, layer)
    require(shard_path.is_file() and receipt_path.is_file(), f"layer {layer} shard is incomplete")
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == SHARD_SCHEMA, f"layer {layer} receipt schema differs")
    require(receipt.get("status") == "completed_layer_shard", f"layer {layer} status differs")
    require(int(receipt.get("layer_index", -1)) == layer, f"layer {layer} receipt index differs")
    require(receipt.get("binding_sha256") == checkpoint["binding"]["binding_sha256"], f"layer {layer} binding differs")
    require(receipt.get("finalizer_sha256") == finalizer_sha, f"layer {layer} finalizer differs")


def encode_layers(args: argparse.Namespace) -> None:
    import fcntl
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    reconstruction_path = Path(args.save_recon).resolve()
    contract = validate_contract(config_path, args.model, args.ratio)
    require_dev_shm(reconstruction_path)
    method = validate_method_contract(contract, output_root, reconstruction_path)
    require(method.get("status") == "build_required", "method config was already written")
    assert_no_competing_builder(output_root)
    require(not (output_root / "compressed_ple.npz").exists(), "compact artifact already exists")
    checkpoint_path = output_root / "fit_checkpoint.npz"
    checkpoint = load_terminal_checkpoint(contract, checkpoint_path)
    finalizer_sha = sha256_file(Path(__file__))
    layers = parse_layers(args.layers, contract["layers"])
    proof = prove_row_chunk_equivalence(args.device, int(args.row_chunk))
    codebooks = checkpoint["codebooks"]
    shard_directory = output_root / SHARD_DIRECTORY / "layers"
    shard_directory.mkdir(parents=True, exist_ok=True)
    with safe_open(str(contract["weights"]), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(PLE_KEY)
        require(
            tuple(tensor_slice.get_shape()) == (contract["vocab"], contract["layers"] * contract["d_ple"]),
            "checkpoint PLE shape differs",
        )
        for layer in layers:
            shard_path, receipt_path, lock_path = shard_paths(output_root, layer)
            with lock_path.open("a+b") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    if args.claim_next:
                        print(f"[full-aq-finalizer] layer {layer} claimed by another worker; skipped", flush=True)
                        continue
                    raise RuntimeError(f"layer {layer} is already owned by another worker") from error
                if shard_path.exists() and receipt_path.exists():
                    if args.claim_next:
                        shallow_validate_existing_shard(
                            checkpoint, output_root, layer, finalizer_sha
                        )
                    else:
                        validate_shard(
                            contract, checkpoint, output_root, layer,
                            expected_finalizer_sha256=finalizer_sha,
                        )
                    print(f"[full-aq-finalizer] layer {layer} already valid; skipped", flush=True)
                    continue
                # A process may stop after atomically replacing the tensor file
                # but before its receipt.  Such an unpaired intermediate file is
                # never published and is replaced by this retry.
                started = time.monotonic()
                start = layer * contract["d_ple"]
                stop = (layer + 1) * contract["d_ple"]
                source = tensor_slice[:, start:stop].float().contiguous()
                codes, per_row_loss = beam_encode_rows(
                    source,
                    codebooks,
                    beam_size=int(contract["fit"]["final_beam_size"]),
                    row_chunk=int(args.row_chunk),
                    device=args.device,
                )
                reconstruction = decode_codes(
                    codes, codebooks, row_chunk=int(args.row_chunk), device=args.device
                )
                source_sse = float((reconstruction.float() - source).double().square().sum().item())
                source_norm = float(source.double().square().sum().item())
                code_dtype = torch.uint8 if int(contract["rate"]["K"]) <= 256 else torch.uint16
                require(code_dtype == torch.uint8, "this finalizer supports the K=256 matrix")
                deployed_codes = codes.to(code_dtype)
                temporary = shard_path.with_name(shard_path.name + f".tmp.{os.getpid()}")
                save_file(
                    {"codes": deployed_codes.contiguous(), "reconstruction": reconstruction.contiguous()},
                    str(temporary),
                )
                with temporary.open("rb") as temporary_handle:
                    os.fsync(temporary_handle.fileno())
                os.replace(temporary, shard_path)
                receipt = {
                    "schema_version": SHARD_SCHEMA,
                    "status": "completed_layer_shard",
                    "final_artifact": False,
                    "publishable": False,
                    "model_selector": contract["model_key"],
                    "ratio_bucket": contract["ratio_key"],
                    "layer_index": layer,
                    "vocab_size": contract["vocab"],
                    "d_ple": contract["d_ple"],
                    "M": int(contract["rate"]["M"]),
                    "K": int(contract["rate"]["K"]),
                    "fit_checkpoint_path": str(checkpoint_path),
                    "fit_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "binding_sha256": checkpoint["binding"]["binding_sha256"],
                    "build_config_sha256": checkpoint["binding"]["build_config_sha256"],
                    "source_weights_sha256": checkpoint["binding"]["source_weights_sha256"],
                    "original_builder_sha256": checkpoint["binding"]["builder_sha256"],
                    "finalizer_sha256": finalizer_sha,
                    "final_encoding_beam_size": int(contract["fit"]["final_beam_size"]),
                    "row_chunk": int(args.row_chunk),
                    "row_chunk_equivalence_proof": proof,
                    "worker_id": str(args.worker_id),
                    "requested_device": str(args.device),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "assignment_objective_sse": float(per_row_loss.double().sum().item()),
                    "reconstruction_sse": source_sse,
                    "source_squared_norm": source_norm,
                    "codes_sha256": tensor_sha256(deployed_codes),
                    "reconstruction_sha256": tensor_sha256(reconstruction),
                    "shard_file": str(shard_path),
                    "shard_file_sha256": sha256_file(shard_path),
                    "elapsed_seconds": time.monotonic() - started,
                }
                write_json_atomic(receipt_path, receipt)
                print(
                    f"[full-aq-finalizer] encoded layer {layer}/{contract['layers'] - 1} "
                    f"in {receipt['elapsed_seconds']:.2f}s",
                    flush=True,
                )


def parse_layers(raw: str, layers: int) -> list[int]:
    if raw == "all":
        return list(range(layers))
    result: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        require(item, "empty layer selector")
        if "-" in item:
            first, last = item.split("-", 1)
            result.extend(range(int(first), int(last) + 1))
        else:
            result.append(int(item))
    require(result and len(result) == len(set(result)), "layer selector is empty or contains duplicates")
    require(all(0 <= layer < layers for layer in result), "layer selector is out of range")
    return result


def pack_layer_code_streams(
    layer_codes: dict[int, Any], *, layers: int, vocab: int, m_count: int, bits: int,
) -> dict[str, Any]:
    """Pack layer-major shards exactly as the monolithic builder's all_codes."""
    import numpy as np

    validate_layer_indices(
        [{"layer_index": layer} for layer in layer_codes], layers
    )
    payload: dict[str, Any] = {}
    for stage in range(m_count):
        stream = np.empty(layers * vocab, dtype=np.uint8 if bits <= 8 else np.uint16)
        for layer in range(layers):
            codes = np.asarray(layer_codes[layer])
            require(tuple(codes.shape) == (vocab, m_count), f"layer {layer} code geometry differs")
            stream[layer * vocab : (layer + 1) * vocab] = codes[:, stage]
        payload[f"codes_m{stage:02d}"] = pack_nbit(stream, bits)
    return payload


def write_deterministic_npz(path: Path, payload: dict[str, Any]) -> None:
    """Write a standard ZIP_STORED NPZ with stable member bytes and metadata."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, value in payload.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safetensors_header(shape: list[int], nbytes: int) -> bytes:
    header = json.dumps(
        {RECON_KEY: {"dtype": "BF16", "shape": shape, "data_offsets": [0, nbytes]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header


def write_dense_reconstruction(
    path: Path, contract: dict[str, Any], shard_files: dict[int, Path],
) -> None:
    """Stream layer BF16 tensors into one standard safetensors file."""
    import numpy as np
    import torch
    from safetensors import safe_open

    vocab, layers, d_ple = contract["vocab"], contract["layers"], contract["d_ple"]
    width = layers * d_ple
    nbytes = vocab * width * 2
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _safetensors_header([vocab, width], nbytes)
    with path.open("w+b") as output:
        output.write(header)
        output.truncate(len(header) + nbytes)
        output.flush()
        os.fsync(output.fileno())
    target = np.memmap(path, mode="r+", dtype=np.uint16, offset=len(header), shape=(vocab, width))
    try:
        for layer in range(layers):
            with safe_open(str(shard_files[layer]), framework="pt", device="cpu") as handle:
                reconstruction = handle.get_tensor("reconstruction").contiguous()
            source = reconstruction.view(-1).view(torch.uint16).numpy().reshape(vocab, d_ple)
            target[:, layer * d_ple : (layer + 1) * d_ple] = source
            target.flush()
    finally:
        del target


def runtime_receipt(artifact_path: Path, method_id: str, resident_bytes: int) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    module = load_full_aq_ple(artifact_path, device="cpu")
    load_seconds = time.perf_counter() - started
    require(int(compact_resident_bytes(module)) == resident_bytes, "runtime compact bytes differ")
    requested_values = [value for value in (0, 1, 2, 1024, 131072, 258884) if value < module.vocab_size]
    requested = torch.tensor(requested_values, dtype=torch.long)
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = module(requested)
    decode_seconds = time.perf_counter() - started
    require(bool(torch.isfinite(decoded.float()).all().item()), "runtime requested rows are non-finite")
    return {
        "schema_version": "ple-compact-requested-row-runtime-v1",
        "status": "passed",
        "method": method_id,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256(artifact_path),
        "device": "cpu",
        "requested_token_ids": requested.tolist(),
        "requested_rows": len(requested),
        "output_shape": list(decoded.shape),
        "persistent_resident_bytes": resident_bytes,
        "finite": True,
        "full_table_materialized": False,
        "dense_ple_tensor_resident": False,
        "benchmark_inputs_used": False,
        "module_load_seconds": load_seconds,
        "requested_row_decode_seconds": decode_seconds,
    }


def publication_paths(output_root: Path, reconstruction_path: Path) -> tuple[dict[Path, Path], Path, Path]:
    staging = output_root / SHARD_DIRECTORY / "fan_in_staging"
    staged_reconstruction = reconstruction_path.with_name(
        reconstruction_path.name + ".parallel-finalizer-staging"
    )
    targets = {
        reconstruction_path: staged_reconstruction,
        output_root / "compressed_ple.npz": staging / "compressed_ple.npz",
        output_root / "training_trajectory.jsonl": staging / "training_trajectory.jsonl",
        output_root / "requested_row_roundtrip_receipt.json": staging / "requested_row_roundtrip_receipt.json",
        output_root / "requested_row_runtime_receipt.json": staging / "requested_row_runtime_receipt.json",
        output_root / "parallel_final_encoding_receipt.json": staging / "parallel_final_encoding_receipt.json",
        output_root / "summary.json": staging / "summary.json",
    }
    return targets, staging / "method_config.json", output_root / SHARD_DIRECTORY / "publication_manifest.json"


def recover_prepared_publication(
    contract: dict[str, Any], output_root: Path, reconstruction_path: Path,
) -> bool:
    """Complete a method-config-last commit after a process interruption."""
    targets, staged_method, manifest_path = publication_paths(output_root, reconstruction_path)
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    require(
        manifest.get("schema_version") == "full-aq-parallel-publication-commit-v1",
        "publication recovery manifest schema differs",
    )
    expected_targets = manifest.get("final_targets")
    require(isinstance(expected_targets, dict), "publication recovery target hashes are missing")
    require(set(expected_targets) == {str(path) for path in targets}, "publication recovery target set differs")
    for target, staged in targets.items():
        expected_hash = str(expected_targets[str(target)])
        if target.exists():
            require(sha256_file(target) == expected_hash, f"existing recovery target differs: {target}")
            continue
        require(staged.is_file(), f"both staged and final recovery targets are missing: {target}")
        require(sha256_file(staged) == expected_hash, f"staged recovery target differs: {staged}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, target)
        require(sha256_file(target) == expected_hash, f"committed recovery target differs: {target}")
    method_path = method_config_path(contract)
    expected_method_hash = str(manifest.get("method_config_sha256", ""))
    current_method = read_json(method_path)
    if current_method.get("status") == "ready_for_evaluation":
        require(sha256_file(method_path) == expected_method_hash, "recovered method config differs")
    else:
        require(current_method.get("status") == "build_required", "recovery method status differs")
        require(staged_method.is_file(), "staged recovery method config is missing")
        require(sha256_file(staged_method) == expected_method_hash, "staged recovery method differs")
        os.replace(staged_method, method_path)
        require(sha256_file(method_path) == expected_method_hash, "recovered method config differs")
    manifest["status"] = "method_config_written"
    write_json_atomic(manifest_path, manifest)
    validate_published(contract, output_root, reconstruction_path, read_json(method_path))
    print("[full-aq-finalizer] recovered prepared method-config-last publication", flush=True)
    return True


def fan_in(args: argparse.Namespace) -> None:
    import numpy as np
    from safetensors import safe_open

    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    reconstruction_path = Path(args.save_recon).resolve()
    contract = validate_contract(config_path, args.model, args.ratio)
    require_dev_shm(reconstruction_path)
    method = validate_method_contract(contract, output_root, reconstruction_path)
    if method.get("status") == "ready_for_evaluation":
        validate_published(contract, output_root, reconstruction_path, method)
        print("[full-aq-finalizer] publication already complete and valid", flush=True)
        return
    assert_no_competing_builder(output_root)
    checkpoint_path = output_root / "fit_checkpoint.npz"
    checkpoint = load_terminal_checkpoint(contract, checkpoint_path)
    finalizer_sha = sha256_file(Path(__file__))
    receipt_files = sorted((output_root / SHARD_DIRECTORY / "layers").glob("layer_*.json"))
    receipts = validate_layer_indices([read_json(path) for path in receipt_files], contract["layers"])
    validated = [
        validate_shard(
            contract, checkpoint, output_root, layer,
            expected_finalizer_sha256=finalizer_sha,
        )
        for layer in range(contract["layers"])
    ]
    require(
        [int(value["layer_index"]) for value in receipts]
        == [int(value["layer_index"]) for value in validated],
        "discovered layer receipts differ from validated layer set",
    )
    if recover_prepared_publication(contract, output_root, reconstruction_path):
        return
    targets, staged_method, manifest_path = publication_paths(output_root, reconstruction_path)
    trajectory_path = output_root / "training_trajectory.jsonl"
    preexisting_trajectory = validate_preexisting_training_trajectory(
        trajectory_path, checkpoint["trajectory"]
    )
    for target in targets:
        if target != trajectory_path:
            require(not target.exists(), f"refusing unbound pre-existing final target: {target}")
    staging = output_root / SHARD_DIRECTORY / "fan_in_staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged_artifact = targets[output_root / "compressed_ple.npz"]
    staged_reconstruction = targets[reconstruction_path]
    shard_files: dict[int, Path] = {}
    layer_codes: dict[int, Any] = {}
    for layer in range(contract["layers"]):
        shard_path, _, _ = shard_paths(output_root, layer)
        shard_files[layer] = shard_path
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            layer_codes[layer] = handle.get_tensor("codes").numpy()
    packed = pack_layer_code_streams(
        layer_codes,
        layers=contract["layers"],
        vocab=contract["vocab"],
        m_count=int(contract["rate"]["M"]),
        bits=int(contract["rate"]["bits"]),
    )
    identity = method_identity(contract["config"])
    method_id = f"{identity['method_id_prefix']}_{contract['model_key']}_r{contract['ratio']}_seed0"
    metadata = {
        "schema_version": "full-additive-quantization-ple-artifact-v1",
        "compressed_ple_format_id": FORMAT_ID,
        "method_id": method_id,
        "method_family": "full_additive_quantization",
        "publication_name": identity["publication_name"],
        "publication_role": "external_baseline",
        "primary_paper": PRIMARY_PAPER,
        "author_repository": AUTHOR_REPOSITORY,
        "author_commit": AUTHOR_COMMIT,
        "author_algorithm_core": ["encodePointsBeamSearch", "learnCodebooksAQ sparse LSMR update"],
        "training_status": checkpoint["training_status"],
        "training_status_reason": checkpoint["status_reason"],
        "initialization": "weights-only greedy residual k-means",
        "optimization": "alternating author-style beam assignment and a multi-RHS Moore-Penrose solution of the same unregularized global sparse least-squares codebook update",
        "fit_assignment_beam_size": 16,
        "final_encoding_beam_size": 64,
        "calibration_corpus_used": False,
        "hidden_raw_copy_allowed": False,
        "full_ple_coverage": True,
        "deployment_mode": "online_requested_token_row_decode",
        "capability_mode": "dense_reconstruction_injection_ephemeral_only",
        "source_model_id": contract["model"]["model_id"],
        "source_revision": contract["model"]["revision"],
        "ple_key": PLE_KEY,
        "ple_shape": [contract["vocab"], contract["layers"] * contract["d_ple"]],
        "num_layers": contract["layers"],
        "d_ple": contract["d_ple"],
        "M": int(contract["rate"]["M"]),
        "K": int(contract["rate"]["K"]),
        "bits": int(contract["rate"]["bits"]),
        "storage_bytes": contract["resident_bytes"],
        "dense_ple_bytes": contract["dense_bytes"],
        "compression_ratio": contract["achieved_ratio"],
        "seed": 0,
        "original_builder_sha256": checkpoint["binding"]["builder_sha256"],
        "parallel_finalizer_sha256": finalizer_sha,
        "final_encoding_execution": "post_fit_parallel_layer_shards",
    }
    payload: dict[str, Any] = {"codebooks_fp16": checkpoint["codebooks"].numpy()}
    payload.update(packed)
    actual_resident = sum(int(value.nbytes) for value in payload.values())
    require(actual_resident == contract["resident_bytes"], "serialized arrays violate resident-byte contract")
    payload["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    write_deterministic_npz(staged_artifact, payload)
    write_dense_reconstruction(staged_reconstruction, contract, shard_files)
    reconstruction_sse = sum(float(value["reconstruction_sse"]) for value in validated)
    source_norm = sum(float(value["source_squared_norm"]) for value in validated)
    reconstruction_nmse = reconstruction_sse / max(source_norm, 1e-30)
    audit_ids = sorted({0, 1, contract["vocab"] // 2, contract["vocab"] - 2, contract["vocab"] - 1})
    roundtrip = audit_requested_rows(
        staged_artifact, staged_reconstruction, token_ids=audit_ids, device="cpu"
    )
    # Receipts refer to final paths; only their content is staged here.
    roundtrip["artifact_sha256"] = artifact_sha256(staged_artifact)
    roundtrip["reconstruction_sha256"] = artifact_sha256(staged_reconstruction)
    runtime = runtime_receipt(staged_artifact, method_id, actual_resident)
    runtime["artifact_path"] = str(output_root / "compressed_ple.npz")
    staged_trajectory = staging / "training_trajectory.jsonl"
    write_jsonl_atomic(staged_trajectory, checkpoint["trajectory"])
    if preexisting_trajectory:
        require(
            sha256_file(trajectory_path) == sha256_file(staged_trajectory),
            "pre-existing training trajectory serialization differs from the staged checkpoint trajectory",
        )
    fan_in_receipt = {
        "schema_version": FAN_IN_SCHEMA,
        "status": "passed_complete_layer_fan_in",
        "final_artifact": True,
        "model_selector": contract["model_key"],
        "ratio_bucket": contract["ratio_key"],
        "expected_layers": contract["layers"],
        "received_layers": [int(value["layer_index"]) for value in validated],
        "duplicate_layers": [],
        "missing_layers": [],
        "fit_checkpoint_path": str(checkpoint_path),
        "fit_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "binding_sha256": checkpoint["binding"]["binding_sha256"],
        "original_builder_path": str(BUILDER_PATH),
        "original_builder_sha256": checkpoint["binding"]["builder_sha256"],
        "finalizer_path": str(Path(__file__).resolve()),
        "finalizer_sha256": finalizer_sha,
        "final_encoding_beam_size": 64,
        "layer_receipt_sha256": {
            f"layer_{int(value['layer_index']):03d}": sha256_file(shard_paths(output_root, int(value["layer_index"]))[1])
            for value in validated
        },
        "compact_payload_sha256": artifact_sha256(staged_artifact),
        "dense_reconstruction_sha256": artifact_sha256(staged_reconstruction),
        "reconstruction_sse": reconstruction_sse,
        "source_squared_norm": source_norm,
        "reconstruction_nmse": reconstruction_nmse,
    }
    common = {
        "schema_version": "full-additive-quantization-ple-build-summary-v1",
        "method_family": "full_additive_quantization",
        "method_id": method_id,
        "publication_name": identity["publication_name"],
        "publication_role": "external_baseline",
        "primary_paper": PRIMARY_PAPER,
        "author_repository": AUTHOR_REPOSITORY,
        "author_commit": AUTHOR_COMMIT,
        "model_selector": contract["model_key"],
        "ratio_bucket": contract["ratio_key"],
        "source_model_id": contract["model"]["model_id"],
        "source_revision": contract["model"]["revision"],
        "source_weights": str(contract["weights"]),
        "source_weights_sha256": checkpoint["binding"]["source_weights_sha256"],
        "fit_row_count": int(checkpoint["binding"]["fit_row_count"]),
        "fit_row_sampling": "seeded uniform sample without replacement from layer-major PLE rows",
        "fit_row_ids_sha256": checkpoint["fit_row_ids_raw_sha256"],
        "fit_assignment_beam_size": 16,
        "final_encoding_beam_size": 64,
        "training_status": checkpoint["training_status"],
        "training_status_reason": checkpoint["status_reason"],
        "completed_sweeps": len(checkpoint["trajectory"]) - 1,
        "fit_elapsed_seconds": checkpoint["fit_elapsed_seconds"],
        "fit_objective_sse": float(checkpoint["trajectory"][-1]["objective_sse"]),
        "fit_objective_nmse": float(checkpoint["trajectory"][-1]["objective_nmse"]),
        "training_trajectory_path": str(trajectory_path),
        "training_completion_contract": contract["fit"]["completion"],
        "lsmr_contract": contract["fit"]["lsmr"],
        "compression_ratio": contract["achieved_ratio"],
        "elapsed_seconds_before_full_encoding": checkpoint["fit_elapsed_seconds"],
        "elapsed_seconds": checkpoint["fit_elapsed_seconds"] + sum(float(value["elapsed_seconds"]) for value in validated),
        "elapsed_seconds_semantics": "fit wall time plus aggregate layer-worker compute time; parallel critical path is recorded by worker IDs",
        "final_encoding_aggregate_worker_seconds": sum(float(value["elapsed_seconds"]) for value in validated),
        "final_encoding_worker_seconds": {
            worker: sum(float(value["elapsed_seconds"]) for value in validated if value["worker_id"] == worker)
            for worker in sorted({str(value["worker_id"]) for value in validated})
        },
        "final_artifact_emitted": True,
        "compressed_artifact_path": str(output_root / "compressed_ple.npz"),
        "compressed_artifact_sha256": artifact_sha256(staged_artifact),
        "compressed_artifact_file_bytes": staged_artifact.stat().st_size,
        "reconstruction_path": str(reconstruction_path),
        "reconstruction_sha256": artifact_sha256(staged_reconstruction),
        "roundtrip_receipt_path": str(output_root / "requested_row_roundtrip_receipt.json"),
        "runtime_receipt_path": str(output_root / "requested_row_runtime_receipt.json"),
        "resident_bytes": actual_resident,
        "reconstruction_nmse": reconstruction_nmse,
        "parallel_final_encoding_receipt_path": str(output_root / "parallel_final_encoding_receipt.json"),
        "original_builder_sha256": checkpoint["binding"]["builder_sha256"],
        "parallel_finalizer_sha256": finalizer_sha,
    }
    staged_roundtrip = staging / "requested_row_roundtrip_receipt.json"
    staged_runtime = staging / "requested_row_runtime_receipt.json"
    staged_fan_in = staging / "parallel_final_encoding_receipt.json"
    staged_summary = staging / "summary.json"
    write_json_atomic(staged_roundtrip, roundtrip)
    write_json_atomic(staged_runtime, runtime)
    write_json_atomic(staged_fan_in, fan_in_receipt)
    write_json_atomic(staged_summary, common)
    final_method = dict(method)
    final_method.update(
        {
            "status": "ready_for_evaluation",
            "builder_sha256": checkpoint["binding"]["builder_sha256"],
            "original_builder_sha256": checkpoint["binding"]["builder_sha256"],
            "parallel_finalizer_path": str(Path(__file__).resolve()),
            "parallel_finalizer_sha256": finalizer_sha,
            "final_encoding_execution": "post_fit_parallel_layer_shards",
            "parallel_final_encoding_receipt_path": str(output_root / "parallel_final_encoding_receipt.json"),
            "parallel_final_encoding_receipt_sha256": sha256_file(staged_fan_in),
            "compressed_artifact_sha256": common["compressed_artifact_sha256"],
            "compressed_artifact_file_bytes": common["compressed_artifact_file_bytes"],
            "reconstruction_sha256": common["reconstruction_sha256"],
            "reported_reconstruction_nmse": reconstruction_nmse,
            "training_metrics_sha256": sha256_file(staged_summary),
            "roundtrip_receipt_sha256": sha256_file(staged_roundtrip),
            "online_adapter_sha256": sha256_file(ONLINE_ADAPTER_PATH),
        }
    )
    write_json_atomic(staged_method, final_method)
    require(targets[trajectory_path] == staged_trajectory, "trajectory staging path differs")
    require(targets[output_root / "requested_row_roundtrip_receipt.json"] == staged_roundtrip, "roundtrip staging path differs")
    require(targets[output_root / "requested_row_runtime_receipt.json"] == staged_runtime, "runtime staging path differs")
    require(targets[output_root / "parallel_final_encoding_receipt.json"] == staged_fan_in, "fan-in staging path differs")
    require(targets[output_root / "summary.json"] == staged_summary, "summary staging path differs")
    manifest = {
        "schema_version": "full-aq-parallel-publication-commit-v1",
        "status": "method_config_staged",
        "final_targets": {str(target): sha256_file(staged) for target, staged in targets.items()},
        "method_config_path": str(method_config_path(contract)),
        "method_config_sha256": sha256_file(staged_method),
    }
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        require(existing_manifest.get("final_targets") == manifest["final_targets"], "recovery publication manifest differs")
        require(existing_manifest.get("method_config_sha256") == manifest["method_config_sha256"], "recovery method config differs")
    else:
        write_json_atomic(manifest_path, manifest)
    for target, staged in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = manifest["final_targets"][str(target)]
        if target.exists():
            require(sha256_file(target) == expected_hash, f"existing recovery target differs: {target}")
        else:
            os.replace(staged, target)
            require(sha256_file(target) == expected_hash, f"committed target hash differs: {target}")
    method_path = method_config_path(contract)
    current_method = read_json(method_path)
    if current_method.get("status") == "ready_for_evaluation":
        require(sha256_file(method_path) == manifest["method_config_sha256"], "existing method config differs")
    else:
        require(current_method == method, "method config changed during finalization")
        os.replace(staged_method, method_path)
        require(sha256_file(method_path) == manifest["method_config_sha256"], "method config hash differs")
    manifest["status"] = "method_config_written"
    write_json_atomic(manifest_path, manifest)
    validate_published(contract, output_root, reconstruction_path, read_json(method_path))
    print(
        f"[full-aq-finalizer] published {contract['model_key']} r{contract['ratio']} "
        f"NMSE={reconstruction_nmse:.8f} ratio={contract['achieved_ratio']:.6f}x",
        flush=True,
    )


def validate_published(
    contract: dict[str, Any], output_root: Path, reconstruction_path: Path, method: dict[str, Any],
) -> None:
    artifact_path = output_root / "compressed_ple.npz"
    summary_path = output_root / "summary.json"
    roundtrip_path = output_root / "requested_row_roundtrip_receipt.json"
    runtime_path = output_root / "requested_row_runtime_receipt.json"
    fan_in_path = output_root / "parallel_final_encoding_receipt.json"
    for path in (artifact_path, reconstruction_path, summary_path, roundtrip_path, runtime_path, fan_in_path):
        require(path.is_file(), f"published output is missing: {path}")
    summary = read_json(summary_path)
    roundtrip = read_json(roundtrip_path)
    runtime = read_json(runtime_path)
    fan_in_receipt = read_json(fan_in_path)
    require(method.get("status") == "ready_for_evaluation", "method is not ready")
    require(summary.get("final_artifact_emitted") is True, "summary does not record a final artifact")
    require(summary.get("training_status") == "completed_fixed_schedule", "summary terminal status differs")
    require(int(summary.get("resident_bytes", -1)) == contract["resident_bytes"], "summary resident bytes differ")
    require(method.get("compressed_artifact_sha256") == artifact_sha256(artifact_path), "method artifact hash differs")
    require(method.get("reconstruction_sha256") == artifact_sha256(reconstruction_path), "method reconstruction hash differs")
    require(method.get("training_metrics_sha256") == sha256_file(summary_path), "method summary hash differs")
    require(method.get("roundtrip_receipt_sha256") == sha256_file(roundtrip_path), "method roundtrip hash differs")
    require(method.get("parallel_final_encoding_receipt_sha256") == sha256_file(fan_in_path), "method fan-in hash differs")
    require(roundtrip.get("status") == "passed" and roundtrip.get("exact_bfloat16") is True, "roundtrip failed")
    require(runtime.get("status") == "passed" and runtime.get("full_table_materialized") is False, "runtime receipt failed")
    require(fan_in_receipt.get("status") == "passed_complete_layer_fan_in", "fan-in receipt failed")
    module = load_full_aq_ple(artifact_path, device="cpu")
    require(int(compact_resident_bytes(module)) == contract["resident_bytes"], "published compact bytes differ")


def plan(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    reconstruction_path = Path(args.save_recon).resolve()
    contract = validate_contract(config_path, args.model, args.ratio)
    require_dev_shm(reconstruction_path)
    method = validate_method_contract(contract, output_root, reconstruction_path)
    require(method.get("status") == "build_required", "method config was already written")
    assert_no_competing_builder(output_root)
    checkpoint = load_terminal_checkpoint(contract, output_root / "fit_checkpoint.npz")
    workers = int(args.workers)
    require(1 <= workers <= contract["layers"], "worker count is invalid")
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "encode",
        "--config", str(config_path),
        "--model", contract["model_key"],
        "--ratio", str(contract["ratio"]),
        "--output-root", str(output_root),
        "--save-recon", str(reconstruction_path),
        "--row-chunk", str(args.row_chunk),
        "--device", "cuda:0",
    ]
    value = {
        "status": "safe_to_launch_after_original_builder_natural_exit",
        "fit_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "binding_sha256": checkpoint["binding"]["binding_sha256"],
        "original_builder_sha256": checkpoint["binding"]["builder_sha256"],
        "finalizer_sha256": sha256_file(Path(__file__)),
        "workers": [
            {
                "worker_id": f"gpu{worker}",
                "layers": "dynamic claim over all incomplete layers",
                "argv_after_cuda_visible_devices_binding": base
                + ["--worker-id", f"gpu{worker}", "--layers", "all", "--claim-next"],
            }
            for worker in range(workers)
        ],
    }
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--config",
            default=str(ROOT / "configs" / "fit/aq_w4a16.json"),
        )
        command.add_argument("--model", choices=("e2b", "e4b"), required=True)
        command.add_argument("--ratio", choices=(20, 40, 100), type=int, required=True)
        command.add_argument("--output-root", required=True)
        command.add_argument("--save-recon", required=True)

    encode = subparsers.add_parser("encode", help="encode one or more intermediate layer shards")
    common(encode)
    encode.add_argument("--layers", required=True, help="comma/range list or 'all'")
    encode.add_argument("--device", default="cuda:0")
    encode.add_argument("--row-chunk", type=int, default=1024)
    encode.add_argument("--worker-id", default="unassigned")
    encode.add_argument(
        "--claim-next",
        action="store_true",
        help="skip layers locked by peer workers, enabling a shared dynamic layer queue",
    )
    encode.set_defaults(function=encode_layers)

    merge = subparsers.add_parser("fan-in", help="validate all layer shards and atomically publish")
    common(merge)
    merge.set_defaults(function=fan_in)

    timing = subparsers.add_parser(
        "row-chunk-timing",
        help="time one real layer without publishing it and fix the r20/r40 execution row chunk",
    )
    common(timing)
    timing.add_argument("--device", default="cuda:0")
    timing.add_argument("--receipt", required=True)
    timing.add_argument("--expected-physical-gpu-uuid", required=True)
    timing.set_defaults(function=row_chunk_timing)

    planning = subparsers.add_parser("plan", help="validate the terminal checkpoint and print layer assignments")
    common(planning)
    planning.add_argument("--workers", type=int, default=6)
    planning.add_argument("--row-chunk", type=int, default=1024)
    planning.set_defaults(function=plan)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
