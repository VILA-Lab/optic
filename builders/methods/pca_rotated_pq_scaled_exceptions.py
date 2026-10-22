#!/usr/bin/env python3
"""Global PCA-rotated PQ with C4-frequency scaled token exceptions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

BASELINE_BUILDERS = Path(__file__).resolve().parents[1] / "baselines"
sys.path.insert(0, str(BASELINE_BUILDERS))

from baseline_common import PLE_KEY, RECON_KEY, VOCAB_SIZE, load_ple_tensor, require_uint16_code_count  # noqa: E402
from baseline_pca_rotated_pq import DEFAULT_IT_WEIGHTS, kmeans, pack_nbit  # noqa: E402


METHOD_ID = "pca_rotated_pq_scaled_exceptions"
FORMAT_ID = "pca_rotated_pq_scaled_exceptions_npz"
DEFAULT_MODEL_PATH = str(Path(DEFAULT_IT_WEIGHTS).resolve().parent)
DEFAULT_FREQUENCY_MANIFEST = "tmp/benchmark_data/ppl_texts/v1/manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_nbit(np: Any, values, bits: int, count: int):
    bit_array = np.unpackbits(values.astype("uint8", copy=False), bitorder="little")[: count * bits]
    bit_array = bit_array.reshape(count, bits)
    shifts = (1 << np.arange(bits, dtype=np.uint32))[None, :]
    return (bit_array.astype(np.uint32) * shifts).sum(axis=1).astype(np.uint16, copy=False)


def require_npz_array(np: Any, npz, name: str, dtype, shape: tuple[int, ...]):
    if name not in npz.files:
        raise SystemExit(f"resident file missing required array {name!r}")
    value = npz[name]
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise SystemExit(f"resident array {name!r} dtype {value.dtype} != {expected_dtype}")
    if tuple(value.shape) != shape:
        raise SystemExit(f"resident array {name!r} shape {tuple(value.shape)} != {shape}")
    return value


def load_metadata(npz) -> dict[str, Any]:
    if "metadata" not in npz.files:
        raise SystemExit("resident file missing required array 'metadata'")
    raw = npz["metadata"]
    if raw.shape != ():
        raise SystemExit(f"resident metadata shape {raw.shape} must be scalar")
    try:
        metadata = json.loads(str(raw.item()))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"resident metadata is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit("resident metadata JSON must be an object")
    return metadata


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


def selected_c4_frequency_tokens(args: argparse.Namespace, vocab: int):
    import numpy as np
    from transformers import PreTrainedTokenizerFast

    manifest_path = Path(args.frequency_manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"--frequency-manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted_tags = {tag.strip() for tag in args.frequency_corpus_tag.split(",") if tag.strip()}
    if not wanted_tags:
        raise SystemExit("--frequency-corpus-tag must name at least one C4 corpus tag")

    tokenizer_roots = [
        args.model_path,
        manifest.get("model_path"),
        DEFAULT_MODEL_PATH,
    ]
    tokenizer_file = None
    for candidate in tokenizer_roots:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            candidate_path = candidate_path.parent
        candidate_tokenizer_file = candidate_path / "tokenizer.json"
        if candidate_tokenizer_file.exists():
            tokenizer_file = candidate_tokenizer_file
            break
    if tokenizer_file is None:
        raise SystemExit("no local tokenizer.json found for C4 frequency-token selection")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    counts = np.zeros(vocab, dtype=np.int64)
    selected_records: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        tag = str(record.get("corpus_tag", ""))
        if tag not in wanted_tags:
            continue
        source_name = str(record.get("source", {}).get("name", ""))
        if "c4" not in tag.lower() and "c4" not in source_name.lower():
            raise SystemExit(
                "this active builder scope supports C4-frequency token exceptions only; "
                f"record {tag!r} has source name {source_name!r}"
            )
        text_path = Path(record["path"])
        if not text_path.is_file():
            raise SystemExit(f"frequency manifest text path does not exist: {text_path}")
        actual_sha256 = sha256_file(text_path)
        declared_sha256 = str(record.get("sha256", ""))
        if declared_sha256 and actual_sha256 != declared_sha256:
            raise SystemExit(f"frequency text sha256 mismatch for {text_path}: {actual_sha256} != {declared_sha256}")
        token_ids = tokenizer(text_path.read_text(encoding="utf-8"), add_special_tokens=False).input_ids
        if token_ids:
            token_ids_np = np.asarray(token_ids, dtype=np.int64)
            token_ids_np = token_ids_np[(0 <= token_ids_np) & (token_ids_np < vocab)]
            if token_ids_np.size:
                counts += np.bincount(token_ids_np, minlength=vocab)[:vocab]
        selected_records.append(
            {
                "corpus_tag": tag,
                "path": str(text_path),
                "sha256": actual_sha256,
                "tokenizer_tokens": int(record.get("tokenizer_tokens", len(token_ids))),
                "source_name": source_name,
            }
        )

    if not selected_records:
        raise SystemExit(f"--frequency-corpus-tag selected no records from {manifest_path}")
    if int(counts.sum()) <= 0:
        raise SystemExit("selected C4 frequency records produced no tokenizer ids")
    if args.exception_count > vocab:
        raise SystemExit(f"--exception-count {args.exception_count} exceeds vocab size {vocab}")
    order = np.lexsort((np.arange(vocab), -counts))
    selected = order[: args.exception_count].astype(np.uint32)
    return selected, counts[selected].astype(np.int64), selected_records, int(counts.sum())


def train_pq_codebooks(torch: Any, np: Any, x_rot, m_count: int, clusters: int, iters: int, batch_size: int, device, seed: int):
    rows, d = x_rot.shape
    d_sub = d // m_count
    codebooks = np.empty((m_count, clusters, d_sub), dtype=np.float16)
    assignments = np.empty((m_count, rows), dtype=np.uint16)
    recon_rot = torch.empty_like(x_rot)
    for subspace in range(m_count):
        print(f"[pca-pq-scaled-exc] subspace {subspace + 1}/{m_count}", flush=True)
        start_dim = subspace * d_sub
        stop_dim = (subspace + 1) * d_sub
        sub = x_rot[:, start_dim:stop_dim].contiguous()
        centers, codes = kmeans(torch, sub, clusters, iters, batch_size, device, seed + 1009 * subspace)
        centers_resident = centers.to(torch.float16).to(torch.float32)
        recon_rot[:, start_dim:stop_dim] = centers_resident[codes.long()].to(torch.float32)
        codebooks[subspace] = centers_resident.to(torch.float16).numpy()
        assignments[subspace] = codes.to(torch.uint16).numpy()
        del sub, centers, centers_resident, codes
    return codebooks, assignments, recon_rot


def storage_components(vocab: int, layers: int, d: int, subspaces: int, clusters: int, exception_count: int):
    bits = int(math.ceil(math.log2(clusters)))
    rows = vocab * layers
    d_sub = d // subspaces
    components = {
        "code_bytes": math.ceil(rows * subspaces * bits / 8),
        "codebook_bytes": subspaces * clusters * d_sub * 2,
        "rotation_bytes": d * d * 2,
        "mean_bytes": d * 2,
        "exception_id_bytes": exception_count * 4,
        "exception_value_bytes": exception_count * layers * d * 2,
        "raw_bytes": rows * d * 2,
    }
    components["storage_bytes"] = (
        components["code_bytes"]
        + components["codebook_bytes"]
        + components["rotation_bytes"]
        + components["mean_bytes"]
        + components["exception_id_bytes"]
        + components["exception_value_bytes"]
    )
    return components


def build_rotation(torch: Any, x_flat, d: int, batch_size: int):
    rows = x_flat.shape[0]
    mean = x_flat.mean(dim=0)
    cov = torch.zeros((d, d), dtype=torch.float64)
    for start in range(0, rows, batch_size):
        stop = min(rows, start + batch_size)
        chunk = (x_flat[start:stop] - mean).to(torch.float64)
        cov += chunk.T @ chunk
        del chunk
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    rotation = eigvecs[:, order].to(torch.float32).contiguous()
    return mean, rotation


def rotate_rows(torch: Any, x_flat, mean, rotation, batch_size: int):
    rows = x_flat.shape[0]
    x_rot = torch.empty_like(x_flat)
    for start in range(0, rows, batch_size):
        stop = min(rows, start + batch_size)
        x_rot[start:stop] = (x_flat[start:stop] - mean) @ rotation
    return x_rot


def materialize_reconstruction_and_exceptions(
    torch: Any,
    np: Any,
    x_flat,
    recon_rot,
    rotation_resident,
    mean_resident,
    exception_token_ids,
    exception_scale: float,
    vocab: int,
    layers: int,
    d: int,
    recon_path: Path,
):
    from safetensors.torch import save_file

    token_idx = torch.from_numpy(exception_token_ids.astype(np.int64, copy=False))
    exception_count = int(exception_token_ids.shape[0])
    exception_residuals = np.empty((exception_count, layers, d), dtype=np.float16)
    recon_packed = torch.empty((vocab, layers * d), dtype=torch.bfloat16)
    raw_err = 0.0
    raw_norm = 0.0
    rotation_t = rotation_resident.T.contiguous()

    for layer in range(layers):
        rows_slice = slice(layer * vocab, (layer + 1) * vocab)
        base_layer = recon_rot[rows_slice] @ rotation_t + mean_resident
        if exception_count:
            row_ids = layer * vocab + token_idx
            residual = (x_flat[row_ids] - base_layer[token_idx]) * float(exception_scale)
            exception_residuals[:, layer, :] = residual.to(torch.float16).numpy()
            base_layer[token_idx] += torch.from_numpy(exception_residuals[:, layer, :]).to(torch.float32)
            del residual
        recon_bf16 = base_layer.to(torch.bfloat16)
        ref = x_flat[rows_slice]
        diff = recon_bf16.to(torch.float32) - ref
        raw_err += diff.double().square().sum().item()
        raw_norm += ref.double().square().sum().item()
        recon_packed[:, layer * d : (layer + 1) * d] = recon_bf16
        del base_layer, recon_bf16, ref, diff

    recon_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({RECON_KEY: recon_packed}, str(recon_path))
    return exception_residuals, raw_err / max(raw_norm, 1e-30)


def load_resident_payload(
    np: Any,
    compressed_path: Path,
    *,
    vocab: int,
    layers: int,
    d: int,
    subspaces: int,
    clusters: int,
    exception_count: int,
    expected_storage_bytes: int,
):
    bits = int(math.ceil(math.log2(clusters)))
    rows = vocab * layers
    d_sub = d // subspaces
    code_bytes_per_subspace = math.ceil(rows * bits / 8)
    expected_files = {
        "metadata",
        "codebooks_fp16",
        "rotation_fp16",
        "mean_fp16",
        "exception_token_ids_uint32",
        "exception_scaled_residuals_fp16",
    } | {f"codes_m{idx:02d}" for idx in range(subspaces)}
    with np.load(str(compressed_path), allow_pickle=False) as npz:
        actual_files = set(npz.files)
        if actual_files != expected_files:
            raise SystemExit(
                "compressed PLE representation file entries "
                f"{sorted(actual_files)} != {sorted(expected_files)}"
            )
        metadata = load_metadata(npz)
        expected_metadata = {
            "compressed_ple_format_id": FORMAT_ID,
            "method": METHOD_ID,
            "kind": METHOD_ID,
            "pca_scope": "global",
            "exception_granularity": "token",
            "exception_selector": "c4_frequency",
            "full_ple_coverage": True,
            "recon_key": RECON_KEY,
            "ple_shape": [vocab, layers * d],
            "num_subspaces": subspaces,
            "clusters": clusters,
            "bits_per_code": bits,
            "exception_count": exception_count,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise SystemExit(f"resident metadata {key!r} {metadata.get(key)!r} != {expected!r}")
        codebooks = require_npz_array(np, npz, "codebooks_fp16", np.float16, (subspaces, clusters, d_sub)).copy()
        rotation = require_npz_array(np, npz, "rotation_fp16", np.float16, (d, d)).copy()
        mean = require_npz_array(np, npz, "mean_fp16", np.float16, (d,)).copy()
        exception_token_ids = require_npz_array(
            np, npz, "exception_token_ids_uint32", np.uint32, (exception_count,)
        ).copy()
        exception_residuals = require_npz_array(
            np, npz, "exception_scaled_residuals_fp16", np.float16, (exception_count, layers, d)
        ).copy()
        if exception_token_ids.size:
            if int(exception_token_ids.max()) >= vocab:
                raise SystemExit("resident exception token id exceeds vocab size")
            if len(set(int(item) for item in exception_token_ids.tolist())) != int(exception_token_ids.size):
                raise SystemExit("resident exception token ids must be unique")
        codes = []
        code_arrays_nbytes = 0
        for subspace in range(subspaces):
            packed = require_npz_array(
                np, npz, f"codes_m{subspace:02d}", np.uint8, (code_bytes_per_subspace,)
            ).copy()
            code_arrays_nbytes += int(packed.nbytes)
            unpacked = unpack_nbit(np, packed, bits, rows)
            if unpacked.size and int(unpacked.max()) >= clusters:
                raise SystemExit(f"resident codes_m{subspace:02d} contains out-of-range code")
            codes.append(unpacked)

    resident_bytes = int(
        codebooks.nbytes
        + rotation.nbytes
        + mean.nbytes
        + exception_token_ids.nbytes
        + exception_residuals.nbytes
        + code_arrays_nbytes
    )
    if resident_bytes != expected_storage_bytes:
        raise SystemExit(f"resident tensor bytes {resident_bytes} != declared storage bytes {expected_storage_bytes}")
    return {
        "metadata": metadata,
        "codebooks": codebooks,
        "rotation": rotation,
        "mean": mean,
        "exception_token_ids": exception_token_ids,
        "exception_residuals": exception_residuals,
        "codes": codes,
        "resident_bytes": resident_bytes,
    }


def verify_resident_roundtrip(
    torch: Any,
    np: Any,
    compressed_path: Path,
    recon_path: Path,
    *,
    vocab: int,
    layers: int,
    d: int,
    subspaces: int,
    clusters: int,
    exception_count: int,
    expected_storage_bytes: int,
    eval_batch_size: int,
    atol: float,
) -> float:
    from safetensors import safe_open

    payload = load_resident_payload(
        np,
        compressed_path,
        vocab=vocab,
        layers=layers,
        d=d,
        subspaces=subspaces,
        clusters=clusters,
        exception_count=exception_count,
        expected_storage_bytes=expected_storage_bytes,
    )
    codebooks = torch.from_numpy(payload["codebooks"]).to(torch.float32)
    rotation_t = torch.from_numpy(payload["rotation"]).to(torch.float32).T.contiguous()
    mean = torch.from_numpy(payload["mean"]).to(torch.float32)
    exception_order = np.argsort(payload["exception_token_ids"].astype(np.int64, copy=False))
    exception_ids = payload["exception_token_ids"].astype(np.int64, copy=False)[exception_order]
    exception_residuals = torch.from_numpy(payload["exception_residuals"][exception_order]).to(torch.float32)
    codes = [item.astype(np.int64, copy=False) for item in payload["codes"]]
    d_sub = d // subspaces
    max_abs_diff = 0.0

    with safe_open(str(recon_path), framework="pt", device="cpu") as recon_file:
        if RECON_KEY not in set(recon_file.keys()):
            raise SystemExit(f"missing dense reconstruction tensor key in {recon_path}: {RECON_KEY}")
        recon_slice = recon_file.get_slice(RECON_KEY)
        expected_shape = [vocab, layers * d]
        if list(recon_slice.get_shape()) != expected_shape:
            raise SystemExit(f"dense reconstruction shape {list(recon_slice.get_shape())} != {expected_shape}")
        for layer in range(layers):
            row_base = layer * vocab
            for start in range(0, vocab, eval_batch_size):
                stop = min(vocab, start + eval_batch_size)
                row_slice = slice(row_base + start, row_base + stop)
                decoded_rot = torch.empty((stop - start, d), dtype=torch.float32)
                for subspace in range(subspaces):
                    start_dim = subspace * d_sub
                    stop_dim = (subspace + 1) * d_sub
                    code_ids = torch.from_numpy(codes[subspace][row_slice])
                    decoded_rot[:, start_dim:stop_dim] = codebooks[subspace, code_ids.long()]
                    del code_ids
                decoded = decoded_rot @ rotation_t + mean
                if exception_ids.size:
                    left = int(np.searchsorted(exception_ids, start, side="left"))
                    right = int(np.searchsorted(exception_ids, stop, side="left"))
                    if right > left:
                        positions = torch.from_numpy((exception_ids[left:right] - start).astype(np.int64, copy=False))
                        decoded[positions.long()] += exception_residuals[left:right, layer, :]
                        del positions
                decoded = decoded.to(torch.bfloat16)
                saved = recon_slice[start:stop, layer * d : (layer + 1) * d]
                decoded_f32 = decoded.to(torch.float32)
                saved_f32 = saved.to(torch.float32)
                if not bool(torch.isfinite(decoded_f32).all().item()):
                    raise SystemExit("resident file reload/decode produced non-finite values")
                if not bool(torch.isfinite(saved_f32).all().item()):
                    raise SystemExit("saved BF16 dense PLE reconstruction contains non-finite values")
                diff = float((decoded_f32 - saved_f32).abs().max().item())
                max_abs_diff = max(max_abs_diff, diff)
                if max_abs_diff > atol:
                    raise SystemExit(
                        f"resident file reload/decode max_abs_diff {max_abs_diff} exceeds tolerance {atol}"
                    )
                del decoded_rot, decoded, saved, decoded_f32, saved_f32
    return max_abs_diff


def reconstruction_nmse_against_path(
    torch: Any,
    safe_open: Any,
    recon_path: Path,
    reference_path: Path,
    layers: int,
    d: int,
    chunk_size: int,
) -> float:
    expected_shape = [VOCAB_SIZE, layers * d]
    err = 0.0
    norm = 0.0
    with safe_open(str(recon_path), framework="pt", device="cpu") as recon_file:
        with safe_open(str(reference_path), framework="pt", device="cpu") as ref_file:
            ref_keys = set(ref_file.keys())
            if RECON_KEY not in set(recon_file.keys()):
                raise SystemExit(f"missing dense reconstruction tensor key in {recon_path}: {RECON_KEY}")
            ref_key = PLE_KEY if PLE_KEY in ref_keys else RECON_KEY if RECON_KEY in ref_keys else None
            if ref_key is None:
                raise SystemExit(f"missing PLE tensor key in {reference_path}; expected {PLE_KEY} or {RECON_KEY}")
            recon_slice = recon_file.get_slice(RECON_KEY)
            ref_slice = ref_file.get_slice(ref_key)
            if list(recon_slice.get_shape()) != expected_shape:
                raise SystemExit(f"dense reconstruction shape {list(recon_slice.get_shape())} != {expected_shape}")
            if list(ref_slice.get_shape()) != expected_shape:
                raise SystemExit(f"reference PLE shape {list(ref_slice.get_shape())} != {expected_shape}")
            for layer in range(layers):
                for start in range(0, VOCAB_SIZE, chunk_size):
                    stop = min(VOCAB_SIZE, start + chunk_size)
                    recon = recon_slice[start:stop, layer * d : (layer + 1) * d].to(torch.float32)
                    ref = ref_slice[start:stop, layer * d : (layer + 1) * d].to(torch.float32)
                    err += (recon - ref).double().square().sum().item()
                    norm += ref.double().square().sum().item()
                    del recon, ref
    return err / max(norm, 1e-30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--it-weights", default=DEFAULT_IT_WEIGHTS)
    parser.add_argument(
        "--source-ple",
        default="",
        help="Optional source PLE tensor to compress instead of --it-weights. Supports dense PLE reconstruction safetensors.",
    )
    parser.add_argument(
        "--reference-it-weights",
        default=DEFAULT_IT_WEIGHTS,
        help="Original checkpoint used only to report reconstruction NMSE versus checkpoint PLE when --source-ple is set.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-layers", type=int, default=35)
    parser.add_argument("--d-ple", type=int, default=256)
    parser.add_argument("--num-subspaces", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--exception-count", type=int, required=True)
    parser.add_argument("--exception-scale", type=float, required=True)
    parser.add_argument("--exception-selector", choices=["c4_frequency"], default="c4_frequency")
    parser.add_argument("--frequency-manifest", default=DEFAULT_FREQUENCY_MANIFEST)
    parser.add_argument("--frequency-corpus-tag", default="c4val")
    parser.add_argument("--resident-roundtrip-atol", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-recon", required=True)
    parser.add_argument("--save-compressed-ple-npz", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_layers <= 0 or args.d_ple != 256:
        raise SystemExit("this active builder requires a positive layer count and d_ple=256")
    if args.num_subspaces <= 0:
        raise SystemExit("--num-subspaces must be positive")
    if args.d_ple % args.num_subspaces != 0:
        raise SystemExit("--num-subspaces must divide --d-ple")
    require_uint16_code_count("--clusters", args.clusters)
    if args.iters <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("--iters, --batch-size, and --eval-batch-size must be positive")
    if args.exception_count < 0:
        raise SystemExit("--exception-count must be nonnegative")
    if not (0.0 <= args.exception_scale <= 1.0) or not math.isfinite(args.exception_scale):
        raise SystemExit("--exception-scale must be finite and in [0, 1]")
    if args.resident_roundtrip_atol < 0.0 or not math.isfinite(args.resident_roundtrip_atol):
        raise SystemExit("--resident-roundtrip-atol must be finite and nonnegative")
    if Path(args.save_recon).suffix != ".safetensors":
        raise SystemExit("--save-recon must write a dense reconstruction safetensors file")
    if Path(args.save_compressed_ple_npz).suffix != ".npz":
        raise SystemExit("--save-compressed-ple-npz must use a .npz filename")
    if Path(args.output).suffix != ".json":
        raise SystemExit("--output must write a metrics JSON file")


def main() -> None:
    args = parse_args()
    validate_args(args)

    import numpy as np
    import torch
    from safetensors import safe_open

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    source_path = args.source_ple or args.it_weights
    print(f"[pca-pq-scaled-exc] loading PLE source={source_path}", flush=True)
    packed, ple_key, ple_shape = load_ple_tensor(
        safe_open,
        source_path,
        args.num_layers,
        args.d_ple,
        allow_recon_key=bool(args.source_ple),
    )
    if tuple(ple_shape) != (VOCAB_SIZE, args.num_layers * args.d_ple):
        raise SystemExit(f"loaded PLE shape metadata {tuple(ple_shape)} != {(VOCAB_SIZE, args.num_layers * args.d_ple)}")
    packed = packed.to(torch.float32)

    vocab = packed.shape[0]
    layers = args.num_layers
    d = args.d_ple
    rows = vocab * layers
    subspaces = args.num_subspaces
    clusters = args.clusters
    bits = int(math.ceil(math.log2(clusters)))
    d_sub = d // subspaces
    device = torch.device(args.device)
    components = storage_components(vocab, layers, d, subspaces, clusters, args.exception_count)

    print(
        f"[pca-pq-scaled-exc] V={vocab} L={layers} d={d} rows={rows} "
        f"M={subspaces} K={clusters} bits={bits} exceptions={args.exception_count} "
        f"scale={args.exception_scale}",
        flush=True,
    )

    exception_token_ids, exception_counts, frequency_records, selector_total_tokens = selected_c4_frequency_tokens(
        args,
        vocab,
    )
    x_flat = packed.reshape(vocab, layers, d).permute(1, 0, 2).reshape(rows, d).contiguous()
    del packed

    mean, rotation = build_rotation(torch, x_flat, d, args.batch_size)
    print("[pca-pq-scaled-exc] global PCA rotation ready", flush=True)
    x_rot = rotate_rows(torch, x_flat, mean, rotation, args.batch_size)
    codebooks, assignments, recon_rot = train_pq_codebooks(
        torch,
        np,
        x_rot,
        subspaces,
        clusters,
        args.iters,
        args.batch_size,
        device,
        args.seed,
    )
    del x_rot

    rotation_resident = rotation.to(torch.float16).to(torch.float32)
    mean_resident = mean.to(torch.float16).to(torch.float32)
    rotation_orthogonality_max_abs = float(
        (rotation_resident.T @ rotation_resident - torch.eye(d, dtype=torch.float32)).abs().max().item()
    )

    recon_path = Path(args.save_recon)
    compressed_path = Path(args.save_compressed_ple_npz)
    output_path = Path(args.output)
    exception_residuals, decoded_source_ple_nmse = materialize_reconstruction_and_exceptions(
        torch,
        np,
        x_flat,
        recon_rot,
        rotation_resident,
        mean_resident,
        exception_token_ids,
        args.exception_scale,
        vocab,
        layers,
        d,
        recon_path,
    )
    del recon_rot

    resident_tensors = [
        "codebooks_fp16",
        "rotation_fp16",
        "mean_fp16",
        "exception_token_ids_uint32",
        "exception_scaled_residuals_fp16",
    ] + [f"codes_m{idx:02d}" for idx in range(subspaces)]
    metadata = {
        "compressed_ple_format_id": FORMAT_ID,
        "method": METHOD_ID,
        "kind": METHOD_ID,
        "pca_scope": "global",
        "exception_granularity": "token",
        "exception_selector": args.exception_selector,
        "frequency_manifest": args.frequency_manifest,
        "frequency_corpus_tag": args.frequency_corpus_tag,
        "num_subspaces": subspaces,
        "clusters": clusters,
        "bits_per_code": bits,
        "bits": bits,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "exception_count": args.exception_count,
        "exception_scale": args.exception_scale,
        "rotation_orthogonality_max_abs": rotation_orthogonality_max_abs,
        "ple_key": ple_key,
        "ple_shape": list(ple_shape),
        "recon_key": RECON_KEY,
        "full_ple_coverage": True,
        "resident_tensors": resident_tensors,
        "seed": args.seed,
    }
    payload = {
        "metadata": np.array(json.dumps(metadata)),
        "codebooks_fp16": codebooks,
        "rotation_fp16": rotation_resident.to(torch.float16).numpy(),
        "mean_fp16": mean_resident.to(torch.float16).numpy(),
        "exception_token_ids_uint32": exception_token_ids,
        "exception_scaled_residuals_fp16": exception_residuals,
    }
    expected_code_bytes_per_subspace = math.ceil(rows * bits / 8)
    for subspace in range(subspaces):
        packed_codes = pack_nbit(assignments[subspace], bits)
        if int(packed_codes.nbytes) != expected_code_bytes_per_subspace:
            raise SystemExit(
                f"packed codes_m{subspace:02d} nbytes {int(packed_codes.nbytes)} "
                f"!= expected {expected_code_bytes_per_subspace}"
            )
        payload[f"codes_m{subspace:02d}"] = packed_codes

    compressed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(compressed_path), **payload)
    roundtrip_max_abs_diff = verify_resident_roundtrip(
        torch,
        np,
        compressed_path,
        recon_path,
        vocab=vocab,
        layers=layers,
        d=d,
        subspaces=subspaces,
        clusters=clusters,
        exception_count=args.exception_count,
        expected_storage_bytes=components["storage_bytes"],
        eval_batch_size=args.eval_batch_size,
        atol=args.resident_roundtrip_atol,
    )

    decoded_original_checkpoint_ple_nmse = None
    if args.source_ple:
        decoded_original_checkpoint_ple_nmse = reconstruction_nmse_against_path(
            torch,
            safe_open,
            recon_path,
            Path(args.reference_it_weights),
            layers,
            d,
            args.eval_batch_size,
        )

    compression_ratio = components["raw_bytes"] / components["storage_bytes"]
    out = dict(metadata)
    out.update(
        {
            "benchmark": benchmark_id_for_metrics_output(output_path),
            "source_ple": args.source_ple or "",
            "validated_full_ple_shape": True,
            "decoded_source_ple_nmse": decoded_source_ple_nmse,
            "decoded_original_checkpoint_ple_nmse": decoded_original_checkpoint_ple_nmse,
            "target_nmse": decoded_source_ple_nmse,
            "raw_nmse": decoded_source_ple_nmse,
            "d_subspace": d_sub,
            "frequency_manifest_sha256": sha256_file(args.frequency_manifest),
            "frequency_records": frequency_records,
            "frequency_selector_total_tokens": selector_total_tokens,
            "exception_token_frequency_count_min": int(exception_counts.min()) if exception_counts.size else None,
            "exception_token_frequency_count_max": int(exception_counts.max()) if exception_counts.size else None,
            "compressed_file_schema_checked": True,
            "resident_tensor_byte_check_passed": True,
            "resident_file_roundtrip_max_abs_diff": roundtrip_max_abs_diff,
            "resident_roundtrip_atol": args.resident_roundtrip_atol,
            "resident_file_roundtrip_passed": roundtrip_max_abs_diff <= args.resident_roundtrip_atol,
            **components,
            "storage_gib": components["storage_bytes"] / 1024**3,
            "compression_ratio": compression_ratio,
            "input_sha256": sha256_file(source_path),
            "reference_it_weights_sha256": sha256_file(args.reference_it_weights) if args.source_ple else None,
            "dense_recon_sha256": sha256_file(recon_path),
            "compressed_ple_sha256": sha256_file(compressed_path),
            "elapsed_seconds": time.time() - t0,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"[pca-pq-scaled-exc] decoded-source-PLE NMSE={decoded_source_ple_nmse:.6f}", flush=True)
    if decoded_original_checkpoint_ple_nmse is not None:
        print(
            f"[pca-pq-scaled-exc] decoded-original-checkpoint-PLE NMSE="
            f"{decoded_original_checkpoint_ple_nmse:.6f}",
            flush=True,
        )
    print(f"[pca-pq-scaled-exc] storage={components['storage_bytes']} bytes ratio={compression_ratio:.4f}x", flush=True)


if __name__ == "__main__":
    main()
