#!/usr/bin/env python3
"""D_out-weighted global PCA-rotated PQ with scaled token exceptions."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pca_rotated_pq_scaled_exceptions as pca_scaled


METHOD_ID = "dout_weighted_pca_rotated_pq_scaled_exceptions"
FORMAT_ID = "dout_weighted_pca_rotated_pq_scaled_exceptions_npz"
LOG_PREFIX = "[dout-weighted-pca-pq-scaled-exc]"

DEFAULT_IT_WEIGHTS = pca_scaled.DEFAULT_IT_WEIGHTS
DEFAULT_MODEL_PATH = pca_scaled.DEFAULT_MODEL_PATH
DEFAULT_FREQUENCY_MANIFEST = pca_scaled.DEFAULT_FREQUENCY_MANIFEST
PLE_KEY = pca_scaled.PLE_KEY
RECON_KEY = pca_scaled.RECON_KEY
VOCAB_SIZE = pca_scaled.VOCAB_SIZE
build_rotation = pca_scaled.build_rotation
kmeans = pca_scaled.kmeans
load_metadata = pca_scaled.load_metadata
load_ple_tensor = pca_scaled.load_ple_tensor
pack_nbit = pca_scaled.pack_nbit
reconstruction_nmse_against_path = pca_scaled.reconstruction_nmse_against_path
require_npz_array = pca_scaled.require_npz_array
require_uint16_code_count = pca_scaled.require_uint16_code_count
rotate_rows = pca_scaled.rotate_rows
sha256_file = pca_scaled.sha256_file
train_pq_codebooks = pca_scaled.train_pq_codebooks
unpack_nbit = pca_scaled.unpack_nbit


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def require_existing_file(option: str, value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise SystemExit(f"{option} does not exist: {path}")
    return path


def parse_csv_tags(value: str, label: str) -> set[str]:
    tags = {part.strip() for part in value.split(",") if part.strip()}
    if not tags:
        raise SystemExit(f"{label} must name at least one corpus tag")
    return tags


def parse_int_list(spec: str, label: str) -> list[int]:
    values: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise SystemExit(f"invalid {label} value {part!r}") from exc
    return values


def load_coord_scales(np: Any, args: argparse.Namespace, *, layers: int, d: int):
    coord_weights_path = require_existing_file("--coord-weights", args.coord_weights)
    coord_weights_np = np.load(str(coord_weights_path), allow_pickle=False).astype("float32", copy=False)
    if tuple(coord_weights_np.shape) != (layers, d):
        raise SystemExit(f"--coord-weights shape {tuple(coord_weights_np.shape)} != {(layers, d)}")
    if not bool(np.isfinite(coord_weights_np).all()) or float(coord_weights_np.min()) <= 0.0:
        raise SystemExit("--coord-weights must be finite and strictly positive")

    source_mean = max(float(coord_weights_np.mean()), 1e-30)
    normalized = coord_weights_np / source_mean
    effective = np.power(normalized, float(args.coord_weight_alpha)).astype("float32", copy=False)
    effective_mean = max(float(effective.mean()), 1e-30)
    effective = effective / effective_mean
    coord_scales = np.sqrt(effective).astype("float32", copy=False)
    coord_scales_fp16 = coord_scales.astype(np.float16, copy=False)
    coord_scales_resident = coord_scales_fp16.astype("float32", copy=False)
    if not bool(np.isfinite(coord_scales_resident).all()) or float(coord_scales_resident.min()) <= 0.0:
        raise SystemExit("resident D_out coordinate scales must be finite and positive")

    metadata_path = Path(args.coord_weights_metadata) if args.coord_weights_metadata else None
    if metadata_path is not None and not metadata_path.is_file():
        raise SystemExit(f"--coord-weights-metadata does not exist: {metadata_path}")

    report = {
        "coord_weights_path": str(coord_weights_path),
        "coord_weights_sha256": sha256_file(coord_weights_path),
        "coord_weights_provenance": args.coord_weights_provenance,
        "coord_weights_metadata_path": str(metadata_path) if metadata_path is not None else "",
        "coord_weights_metadata_sha256": sha256_file(metadata_path) if metadata_path is not None else "",
        "coord_weight_alpha": float(args.coord_weight_alpha),
        "coord_scales_resident": True,
        "coord_weights_source_mean": source_mean,
        "coord_weights_source_min": float(coord_weights_np.min()),
        "coord_weights_source_max": float(coord_weights_np.max()),
        "coord_weights_normalized_min": float(normalized.min()),
        "coord_weights_normalized_max": float(normalized.max()),
        "coord_weights_effective_mean": float(effective.mean()),
        "coord_weights_effective_min": float(effective.min()),
        "coord_weights_effective_max": float(effective.max()),
        "coord_scales_min": float(coord_scales_resident.min()),
        "coord_scales_max": float(coord_scales_resident.max()),
    }
    return coord_scales_resident, coord_scales_fp16, report


def frequency_counts(np: Any, args: argparse.Namespace, vocab: int):
    from transformers import PreTrainedTokenizerFast

    manifest_path = require_existing_file("--frequency-manifest", args.frequency_manifest)
    manifest = load_json(manifest_path)
    wanted_tags = parse_csv_tags(args.frequency_corpus_tag, "--frequency-corpus-tag")

    tokenizer_roots = [args.model_path, manifest.get("model_path"), DEFAULT_MODEL_PATH]
    tokenizer_file = None
    for candidate in tokenizer_roots:
        if not candidate:
            continue
        candidate_path = Path(str(candidate))
        if candidate_path.is_file():
            candidate_path = candidate_path.parent
        candidate_tokenizer_file = candidate_path / "tokenizer.json"
        if candidate_tokenizer_file.exists():
            tokenizer_file = candidate_tokenizer_file
            break
    if tokenizer_file is None:
        raise SystemExit("no local tokenizer.json found for frequency-token selection")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    counts = np.zeros(vocab, dtype=np.int64)
    selected_records: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        tag = str(record.get("corpus_tag", ""))
        if tag not in wanted_tags:
            continue
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
                "source_name": str(record.get("source", {}).get("name", "")),
            }
        )

    if not selected_records:
        raise SystemExit(f"--frequency-corpus-tag selected no records from {manifest_path}")
    if int(counts.sum()) <= 0:
        raise SystemExit("selected frequency records produced no tokenizer ids")
    report = {
        "frequency_manifest": str(manifest_path),
        "frequency_manifest_sha256": sha256_file(manifest_path),
        "frequency_corpus_tag": args.frequency_corpus_tag,
        "frequency_records": selected_records,
        "frequency_selector_total_tokens": int(counts.sum()),
    }
    return counts, report


def dout_gate_scores_and_counts_from_paths(
    np: Any,
    gate_cache_path_raw: str,
    gate_cache_config_path_raw: str,
    corpus_tag: str,
    vocab: int,
):
    import torch
    from safetensors import safe_open

    gate_cache_path = require_existing_file("--dout-gate-cache", gate_cache_path_raw)
    gate_cache_config_path = require_existing_file("--dout-gate-cache-config", gate_cache_config_path_raw)
    config = load_json(gate_cache_config_path)
    wanted_tags = parse_csv_tags(corpus_tag, "--dout-corpus-tag")
    scores = np.zeros(vocab, dtype=np.float64)
    counts = np.zeros(vocab, dtype=np.int64)
    selected_records: list[dict[str, Any]] = []

    with safe_open(str(gate_cache_path), framework="pt", device="cpu") as cache:
        for record in config.get("records", []):
            tag = str(record.get("corpus_tag", ""))
            if tag not in wanted_tags:
                continue
            token_ids = cache.get_tensor(f"token_ids_{tag}").to(torch.long)
            gates = cache.get_tensor(f"g_{tag}").to(torch.float32)
            if token_ids.numel() != gates.shape[0]:
                raise SystemExit(f"D_out gate cache token/gate length mismatch for {tag}")
            if gates.ndim != 3:
                raise SystemExit(f"D_out gate cache g_{tag} must have shape (tokens, layers, d)")
            token_np = token_ids.numpy()
            if token_np.size and (int(token_np.min()) < 0 or int(token_np.max()) >= vocab):
                raise SystemExit(f"D_out gate cache token ids for {tag} are outside the vocab")
            token_score = gates.square().mean(dim=(1, 2)).numpy().astype(np.float64)
            scores += np.bincount(token_np, weights=token_score, minlength=vocab)[:vocab]
            counts += np.bincount(token_np, minlength=vocab)[:vocab]
            selected_records.append(
                {
                    "corpus_tag": tag,
                    "token_count": int(token_ids.numel()),
                    "gate_shape": list(gates.shape),
                }
            )

    if not selected_records:
        raise SystemExit(f"D_out corpus tag selection produced no records from {gate_cache_config_path}")
    report = {
        "dout_gate_cache": str(gate_cache_path),
        "dout_gate_cache_sha256": sha256_file(gate_cache_path),
        "dout_gate_cache_config": str(gate_cache_config_path),
        "dout_gate_cache_config_sha256": sha256_file(gate_cache_config_path),
        "dout_corpus_tag": corpus_tag,
        "dout_gate_records": selected_records,
        "dout_gate_positive_token_count": int((scores > 0.0).sum()),
    }
    return scores, counts, report


def dout_gate_scores_and_counts(np: Any, args: argparse.Namespace, vocab: int):
    return dout_gate_scores_and_counts_from_paths(
        np,
        args.dout_gate_cache,
        args.dout_gate_cache_config,
        args.dout_corpus_tag,
        vocab,
    )


def select_exception_tokens(np: Any, args: argparse.Namespace, vocab: int):
    if args.exception_count == 0:
        return (
            np.empty((0,), dtype=np.uint32),
            np.empty((0,), dtype=np.int64),
            None,
            {"exception_selector_inputs": []},
        )
    if args.exception_count > vocab:
        raise SystemExit(f"--exception-count {args.exception_count} exceeds vocab size {vocab}")

    selector = args.exception_selector
    if selector in {"ppl_text_frequency", "c4_frequency"}:
        counts, frequency_report = frequency_counts(np, args, vocab)
        order = np.lexsort((np.arange(vocab), -counts))
        selected = order[: args.exception_count].astype(np.uint32)
        report = {
            **frequency_report,
            "exception_selector_inputs": [
                {
                    "kind": "frequency_manifest",
                    "path": frequency_report["frequency_manifest"],
                    "sha256": frequency_report["frequency_manifest_sha256"],
                }
            ],
        }
        return selected, counts[selected].astype(np.int64), None, report

    if selector == "dout_gate_sensitivity":
        scores, counts, dout_report = dout_gate_scores_and_counts(np, args, vocab)
        order = np.lexsort((np.arange(vocab), -scores))
        selected = order[: args.exception_count].astype(np.uint32)
        report = {
            **dout_report,
            "exception_selector_inputs": [
                {"kind": "dout_gate_cache", "path": dout_report["dout_gate_cache"], "sha256": dout_report["dout_gate_cache_sha256"]},
                {
                    "kind": "dout_gate_cache_config",
                    "path": dout_report["dout_gate_cache_config"],
                    "sha256": dout_report["dout_gate_cache_config_sha256"],
                },
            ],
        }
        return selected, counts[selected].astype(np.int64), scores[selected].astype(np.float64), report

    if selector == "dual_dout_consensus":
        primary_scores, primary_counts, primary_report = dout_gate_scores_and_counts(np, args, vocab)
        secondary_scores, secondary_counts, secondary_report = dout_gate_scores_and_counts_from_paths(
            np,
            args.secondary_dout_gate_cache,
            args.secondary_dout_gate_cache_config,
            args.secondary_dout_corpus_tag,
            vocab,
        )
        primary_max = float(primary_scores.max()) if primary_scores.size else 0.0
        secondary_max = float(secondary_scores.max()) if secondary_scores.size else 0.0
        if primary_max <= 0.0 or secondary_max <= 0.0:
            raise SystemExit("dual D_out consensus requires positive scores from both gate caches")
        primary_norm = primary_scores / primary_max
        secondary_norm = secondary_scores / secondary_max
        consensus = np.sqrt(primary_norm * secondary_norm)
        support = np.minimum(primary_norm, secondary_norm)
        order = np.lexsort((np.arange(vocab), -support, -consensus))
        selected = order[: args.exception_count].astype(np.uint32)
        report = {
            **primary_report,
            "secondary_dout_gate_cache": secondary_report["dout_gate_cache"],
            "secondary_dout_gate_cache_sha256": secondary_report["dout_gate_cache_sha256"],
            "secondary_dout_gate_cache_config": secondary_report["dout_gate_cache_config"],
            "secondary_dout_gate_cache_config_sha256": secondary_report["dout_gate_cache_config_sha256"],
            "secondary_dout_corpus_tag": args.secondary_dout_corpus_tag,
            "secondary_dout_gate_records": secondary_report["dout_gate_records"],
            "secondary_dout_gate_positive_token_count": secondary_report["dout_gate_positive_token_count"],
            "exception_selector_inputs": [
                {"kind": "dout_gate_cache", "path": primary_report["dout_gate_cache"], "sha256": primary_report["dout_gate_cache_sha256"]},
                {
                    "kind": "dout_gate_cache_config",
                    "path": primary_report["dout_gate_cache_config"],
                    "sha256": primary_report["dout_gate_cache_config_sha256"],
                },
                {
                    "kind": "secondary_dout_gate_cache",
                    "path": secondary_report["dout_gate_cache"],
                    "sha256": secondary_report["dout_gate_cache_sha256"],
                },
                {
                    "kind": "secondary_dout_gate_cache_config",
                    "path": secondary_report["dout_gate_cache_config"],
                    "sha256": secondary_report["dout_gate_cache_config_sha256"],
                },
            ],
        }
        counts = primary_counts[selected] + secondary_counts[selected]
        return selected, counts.astype(np.int64), consensus[selected].astype(np.float64), report

    if selector == "frequency_dout_hybrid":
        counts, frequency_report = frequency_counts(np, args, vocab)
        scores, dout_counts, dout_report = dout_gate_scores_and_counts(np, args, vocab)
        frequency_quota = args.frequency_exception_quota
        if frequency_quota < 0:
            frequency_quota = args.exception_count // 2
        if frequency_quota > args.exception_count:
            raise SystemExit("--frequency-exception-quota cannot exceed --exception-count")
        freq_order = np.lexsort((np.arange(vocab), -counts))
        dout_order = np.lexsort((np.arange(vocab), -scores))
        selected_list: list[int] = []
        seen: set[int] = set()
        for token_id in freq_order[:frequency_quota]:
            tid = int(token_id)
            selected_list.append(tid)
            seen.add(tid)
        for token_id in dout_order:
            tid = int(token_id)
            if tid in seen:
                continue
            selected_list.append(tid)
            seen.add(tid)
            if len(selected_list) >= args.exception_count:
                break
        if len(selected_list) != args.exception_count:
            raise SystemExit("frequency+D_out selector could not fill exception budget")
        selected = np.asarray(selected_list, dtype=np.uint32)
        report = {
            **frequency_report,
            **dout_report,
            "frequency_exception_quota": frequency_quota,
            "exception_selector_inputs": [
                {
                    "kind": "frequency_manifest",
                    "path": frequency_report["frequency_manifest"],
                    "sha256": frequency_report["frequency_manifest_sha256"],
                },
                {"kind": "dout_gate_cache", "path": dout_report["dout_gate_cache"], "sha256": dout_report["dout_gate_cache_sha256"]},
                {
                    "kind": "dout_gate_cache_config",
                    "path": dout_report["dout_gate_cache_config"],
                    "sha256": dout_report["dout_gate_cache_config_sha256"],
                },
            ],
        }
        selected_counts = counts[selected].astype(np.int64)
        selected_scores = scores[selected].astype(np.float64)
        if not selected_counts.any() and dout_counts[selected].any():
            selected_counts = dout_counts[selected].astype(np.int64)
        return selected, selected_counts, selected_scores, report

    if selector == "frequency_stratified_dout":
        counts, frequency_report = frequency_counts(np, args, vocab)
        scores, dout_counts, dout_report = dout_gate_scores_and_counts(np, args, vocab)
        cutoffs = parse_int_list(args.frequency_stratified_cutoffs, "--frequency-stratified-cutoffs")
        if sorted(cutoffs) != cutoffs or len(set(cutoffs)) != len(cutoffs):
            raise SystemExit("--frequency-stratified-cutoffs must be strictly increasing")
        if cutoffs and (cutoffs[0] <= 0 or cutoffs[-1] >= vocab):
            raise SystemExit(f"--frequency-stratified-cutoffs must be inside (0, {vocab})")
        quotas = parse_int_list(args.frequency_stratified_quotas, "--frequency-stratified-quotas")
        if len(quotas) != len(cutoffs) + 1:
            raise SystemExit("--frequency-stratified-quotas must have one value per frequency band")
        if any(quota < 0 for quota in quotas):
            raise SystemExit("--frequency-stratified-quotas must be nonnegative")
        if sum(quotas) != args.exception_count:
            raise SystemExit("--frequency-stratified-quotas must sum to --exception-count")

        freq_order = np.lexsort((np.arange(vocab), -counts))
        bounds = [0, *cutoffs, vocab]
        selected_list: list[int] = []
        seen: set[int] = set()
        band_summaries: list[dict[str, Any]] = []
        for band_idx, (start, stop, quota) in enumerate(zip(bounds[:-1], bounds[1:], quotas, strict=True)):
            if quota == 0:
                band_summaries.append({"band": band_idx, "rank_start": start, "rank_stop": stop, "quota": 0})
                continue
            candidates = freq_order[start:stop]
            positive = candidates[scores[candidates] > 0.0]
            pool = positive if positive.size >= quota else candidates
            band_order = pool[np.lexsort((pool, -scores[pool]))]
            added = 0
            for token_id in band_order:
                tid = int(token_id)
                if tid in seen:
                    continue
                selected_list.append(tid)
                seen.add(tid)
                added += 1
                if added >= quota:
                    break
            band_summaries.append(
                {
                    "band": band_idx,
                    "rank_start": start,
                    "rank_stop": stop,
                    "quota": quota,
                    "positive_score_candidates": int(positive.size),
                    "selected": added,
                }
            )
        if len(selected_list) < args.exception_count:
            global_order = np.lexsort((np.arange(vocab), -scores))
            for token_id in global_order:
                tid = int(token_id)
                if tid in seen:
                    continue
                selected_list.append(tid)
                seen.add(tid)
                if len(selected_list) >= args.exception_count:
                    break
        if len(selected_list) != args.exception_count:
            raise SystemExit("frequency-stratified selector could not fill exception budget")
        selected = np.asarray(selected_list, dtype=np.uint32)
        report = {
            **frequency_report,
            **dout_report,
            "frequency_stratified_cutoffs": args.frequency_stratified_cutoffs,
            "frequency_stratified_quotas": args.frequency_stratified_quotas,
            "frequency_stratified_band_summaries": band_summaries,
            "exception_selector_inputs": [
                {
                    "kind": "frequency_manifest",
                    "path": frequency_report["frequency_manifest"],
                    "sha256": frequency_report["frequency_manifest_sha256"],
                },
                {"kind": "dout_gate_cache", "path": dout_report["dout_gate_cache"], "sha256": dout_report["dout_gate_cache_sha256"]},
                {
                    "kind": "dout_gate_cache_config",
                    "path": dout_report["dout_gate_cache_config"],
                    "sha256": dout_report["dout_gate_cache_config_sha256"],
                },
            ],
        }
        selected_counts = counts[selected].astype(np.int64)
        if not selected_counts.any() and dout_counts[selected].any():
            selected_counts = dout_counts[selected].astype(np.int64)
        return selected, selected_counts, scores[selected].astype(np.float64), report

    raise SystemExit(f"unsupported exception selector: {selector}")


def storage_components(vocab: int, layers: int, d: int, subspaces: int, clusters: int, exception_count: int):
    bits = int(math.ceil(math.log2(clusters)))
    rows = vocab * layers
    d_sub = d // subspaces
    components = {
        "code_bytes": math.ceil(rows * subspaces * bits / 8),
        "codebook_bytes": subspaces * clusters * d_sub * 2,
        "rotation_bytes": d * d * 2,
        "mean_bytes": d * 2,
        "coord_scale_bytes": layers * d * 2,
        "exception_id_bytes": exception_count * 4,
        "exception_value_bytes": exception_count * layers * d * 2,
        "raw_bytes": rows * d * 2,
    }
    components["coord_weight_bytes"] = components["coord_scale_bytes"]
    components["storage_bytes"] = (
        components["code_bytes"]
        + components["codebook_bytes"]
        + components["rotation_bytes"]
        + components["mean_bytes"]
        + components["coord_scale_bytes"]
        + components["exception_id_bytes"]
        + components["exception_value_bytes"]
    )
    return components


def apply_coord_scales(torch: Any, x_flat, coord_scales_resident, *, vocab: int, layers: int, d: int):
    coord_scales = torch.from_numpy(coord_scales_resident.astype("float32", copy=False))
    train_flat = torch.empty_like(x_flat)
    for layer in range(layers):
        rows_slice = slice(layer * vocab, (layer + 1) * vocab)
        train_flat[rows_slice] = x_flat[rows_slice] * coord_scales[layer].reshape(1, d)
    return train_flat


def expected_resident_tensors(subspaces: int) -> list[str]:
    return [
        "codebooks_fp16",
        "rotation_fp16",
        "mean_fp16",
        "coord_scales_fp16",
        "exception_token_ids_uint32",
        "exception_scaled_residuals_fp16",
    ] + [f"codes_m{idx:02d}" for idx in range(subspaces)]


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
    expected_files = {"metadata", *expected_resident_tensors(subspaces)}
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
            "schema_version": "1.0",
            "hidden_raw_copy_allowed": False,
            "pca_scope": "global",
            "exception_granularity": "token",
            "coord_scales_resident": True,
            "full_ple_coverage": True,
            "recon_key": RECON_KEY,
            "ple_shape": [vocab, layers * d],
            "num_subspaces": subspaces,
            "clusters": clusters,
            "bits_per_code": bits,
            "exception_count": exception_count,
            "storage_bytes": expected_storage_bytes,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise SystemExit(f"resident metadata {key!r} {metadata.get(key)!r} != {expected!r}")
        if metadata.get("resident_tensors") != expected_resident_tensors(subspaces):
            raise SystemExit("resident metadata resident_tensors does not match the resident schema")
        codebooks = require_npz_array(np, npz, "codebooks_fp16", np.float16, (subspaces, clusters, d_sub)).copy()
        rotation = require_npz_array(np, npz, "rotation_fp16", np.float16, (d, d)).copy()
        mean = require_npz_array(np, npz, "mean_fp16", np.float16, (d,)).copy()
        coord_scales = require_npz_array(np, npz, "coord_scales_fp16", np.float16, (layers, d)).copy()
        if not bool(np.isfinite(coord_scales.astype("float32", copy=False)).all()) or float(coord_scales.min()) <= 0.0:
            raise SystemExit("resident coord_scales_fp16 must be finite and positive")
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
        + coord_scales.nbytes
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
        "coord_scales": coord_scales,
        "exception_token_ids": exception_token_ids,
        "exception_residuals": exception_residuals,
        "codes": codes,
        "resident_bytes": resident_bytes,
    }


def materialize_reconstruction_and_exceptions(
    torch: Any,
    np: Any,
    x_flat,
    recon_rot,
    rotation_resident,
    mean_resident,
    coord_scales_resident,
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
    coord_scales = torch.from_numpy(coord_scales_resident.astype("float32", copy=False))
    rotation_t = rotation_resident.T.contiguous()
    raw_err = 0.0
    raw_norm = 0.0

    for layer in range(layers):
        rows_slice = slice(layer * vocab, (layer + 1) * vocab)
        base_layer = recon_rot[rows_slice] @ rotation_t + mean_resident
        base_layer = base_layer / coord_scales[layer].reshape(1, d)
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
    save_file({RECON_KEY: recon_packed.contiguous()}, str(recon_path))
    return exception_residuals, raw_err / max(raw_norm, 1e-30)


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
    coord_scales = torch.from_numpy(payload["coord_scales"].astype("float32", copy=False))
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
                decoded = decoded / coord_scales[layer].reshape(1, d)
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
                diff_tensor = (decoded_f32 - saved_f32).abs()
                if not bool(torch.isfinite(diff_tensor).all().item()):
                    raise SystemExit("resident file reload/decode diff contains non-finite values")
                diff = float(diff_tensor.max().item()) if diff_tensor.numel() else 0.0
                max_abs_diff = max(max_abs_diff, diff)
                if max_abs_diff > atol:
                    raise SystemExit(
                        f"resident file reload/decode max_abs_diff {max_abs_diff} exceeds tolerance {atol}"
                    )
                del decoded_rot, decoded, saved, decoded_f32, saved_f32, diff_tensor
    return max_abs_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-name", default=METHOD_ID)
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
    parser.add_argument("--exception-granularity", choices=["token"], default="token")
    parser.add_argument(
        "--exception-selector",
        choices=[
            "ppl_text_frequency",
            "c4_frequency",
            "dout_gate_sensitivity",
            "dual_dout_consensus",
            "frequency_dout_hybrid",
            "frequency_stratified_dout",
        ],
        required=True,
    )
    parser.add_argument("--frequency-manifest", default=DEFAULT_FREQUENCY_MANIFEST)
    parser.add_argument("--frequency-corpus-tag", default="c4val")
    parser.add_argument("--frequency-exception-quota", type=int, default=-1)
    parser.add_argument("--frequency-stratified-cutoffs", default="1024,8192")
    parser.add_argument("--frequency-stratified-quotas", default="")
    parser.add_argument("--dout-gate-cache", default="")
    parser.add_argument("--dout-gate-cache-config", default="")
    parser.add_argument("--dout-corpus-tag", default="")
    parser.add_argument("--secondary-dout-gate-cache", default="")
    parser.add_argument("--secondary-dout-gate-cache-config", default="")
    parser.add_argument("--secondary-dout-corpus-tag", default="")
    parser.add_argument("--coord-weights", required=True, help="D_out coordinate weights .npy with shape (L, d).")
    parser.add_argument(
        "--coord-weights-provenance",
        required=True,
        help="Explicit provenance path for --coord-weights.",
    )
    parser.add_argument(
        "--coord-weights-metadata",
        default="",
        help="Optional metadata JSON for --coord-weights. Its SHA is recorded when supplied.",
    )
    parser.add_argument("--coord-weight-alpha", type=float, required=True)
    parser.add_argument("--pca-scope", choices=["global"], default="global")
    parser.add_argument("--resident-roundtrip-atol", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-recon", required=True)
    parser.add_argument("--save-compressed-ple-npz", "--save-artifact", dest="save_compressed_ple_npz", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.method_name != METHOD_ID:
        raise SystemExit(f"--method-name must be {METHOD_ID!r}")
    if args.num_layers <= 0 or args.d_ple != 256:
        raise SystemExit("this active builder requires a positive layer count and d_ple=256")
    if args.num_subspaces <= 0:
        raise SystemExit("--num-subspaces must be positive")
    if args.d_ple % args.num_subspaces:
        raise SystemExit("--num-subspaces must divide --d-ple")
    require_uint16_code_count("--clusters", args.clusters)
    if args.iters <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("--iters, --batch-size, and --eval-batch-size must be positive")
    if args.exception_count < 0:
        raise SystemExit("--exception-count must be nonnegative")
    if not (0.0 <= args.exception_scale <= 1.0) or not math.isfinite(args.exception_scale):
        raise SystemExit("--exception-scale must be finite and in [0, 1]")
    if not math.isfinite(args.coord_weight_alpha) or not (0.0 < args.coord_weight_alpha <= 1.0):
        raise SystemExit("--coord-weight-alpha must be finite and in (0, 1]")
    if not args.coord_weights_provenance.strip():
        raise SystemExit("--coord-weights-provenance must be a nonempty explicit provenance string")
    if args.resident_roundtrip_atol < 0.0 or not math.isfinite(args.resident_roundtrip_atol):
        raise SystemExit("--resident-roundtrip-atol must be finite and nonnegative")
    if args.exception_count and args.exception_selector in {
        "dout_gate_sensitivity",
        "dual_dout_consensus",
        "frequency_dout_hybrid",
        "frequency_stratified_dout",
    }:
        for option_name, value in [
            ("--dout-gate-cache", args.dout_gate_cache),
            ("--dout-gate-cache-config", args.dout_gate_cache_config),
            ("--dout-corpus-tag", args.dout_corpus_tag),
        ]:
            if not value:
                raise SystemExit(f"{args.exception_selector} requires {option_name}")
    if args.exception_count and args.exception_selector == "dual_dout_consensus":
        for option_name, value in [
            ("--secondary-dout-gate-cache", args.secondary_dout_gate_cache),
            ("--secondary-dout-gate-cache-config", args.secondary_dout_gate_cache_config),
            ("--secondary-dout-corpus-tag", args.secondary_dout_corpus_tag),
        ]:
            if not value:
                raise SystemExit(f"dual_dout_consensus requires {option_name}")
    if Path(args.save_recon).name != "dense_ple_reconstruction.safetensors":
        raise SystemExit("--save-recon must write dense_ple_reconstruction.safetensors")
    if Path(args.save_compressed_ple_npz).name != "compressed_ple.npz":
        raise SystemExit("--save-compressed-ple-npz must write compressed_ple.npz")
    if Path(args.output).name != "metrics.json":
        raise SystemExit("--output must write metrics.json")


def main() -> None:
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

    source_path = args.source_ple or args.it_weights
    print(f"{LOG_PREFIX} loading PLE source={source_path}", flush=True)
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

    vocab = int(packed.shape[0])
    layers = int(args.num_layers)
    d = int(args.d_ple)
    rows = vocab * layers
    subspaces = int(args.num_subspaces)
    clusters = int(args.clusters)
    bits = int(math.ceil(math.log2(clusters)))
    d_sub = d // subspaces
    device = torch.device(args.device)
    components = storage_components(vocab, layers, d, subspaces, clusters, args.exception_count)
    coord_scales_resident, coord_scales_fp16, coord_report = load_coord_scales(np, args, layers=layers, d=d)

    print(
        f"{LOG_PREFIX} V={vocab} L={layers} d={d} rows={rows} "
        f"M={subspaces} K={clusters} bits={bits} exceptions={args.exception_count} "
        f"exception_scale={args.exception_scale:g} coord_alpha={args.coord_weight_alpha:g}",
        flush=True,
    )

    exception_token_ids, exception_counts, exception_scores, selector_report = select_exception_tokens(np, args, vocab)
    x_flat = packed.reshape(vocab, layers, d).permute(1, 0, 2).reshape(rows, d).contiguous()
    del packed
    train_flat = apply_coord_scales(torch, x_flat, coord_scales_resident, vocab=vocab, layers=layers, d=d)

    mean, rotation = build_rotation(torch, train_flat, d, args.batch_size)
    print(f"{LOG_PREFIX} global D_out-weighted PCA rotation ready", flush=True)
    x_rot = rotate_rows(torch, train_flat, mean, rotation, args.batch_size)
    del train_flat
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
        coord_scales_resident,
        exception_token_ids,
        args.exception_scale,
        vocab,
        layers,
        d,
        recon_path,
    )
    del recon_rot

    resident_tensors = expected_resident_tensors(subspaces)
    metadata = {
        "compressed_ple_format_id": FORMAT_ID,
        "method": METHOD_ID,
        "kind": METHOD_ID,
        "schema_version": "1.0",
        "hidden_raw_copy_allowed": False,
        "pca_scope": "global",
        "exception_granularity": "token",
        "exception_selector": args.exception_selector,
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
        "coord_scales_resident": True,
        "storage_bytes": components["storage_bytes"],
        "resident_tensors": resident_tensors,
        "seed": args.seed,
        "nonresident_file_content": [
            "metadata JSON string",
            "D_out coordinate-weight provenance and selector provenance fields",
        ],
    }
    metadata.update(coord_report)
    metadata.update(selector_report)

    payload = {
        "metadata": np.array(json.dumps(metadata)),
        "codebooks_fp16": codebooks,
        "rotation_fp16": rotation_resident.to(torch.float16).numpy(),
        "mean_fp16": mean_resident.to(torch.float16).numpy(),
        "coord_scales_fp16": coord_scales_fp16,
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
    del payload, assignments, codebooks, coord_scales_fp16

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
    del x_flat

    compression_ratio = components["raw_bytes"] / components["storage_bytes"]
    out = dict(metadata)
    out.update(
        {
            "benchmark": pca_scaled.benchmark_id_for_metrics_output(output_path),
            "source_ple": args.source_ple or "",
            "validated_full_ple_shape": True,
            "decoded_source_ple_nmse": decoded_source_ple_nmse,
            "decoded_original_checkpoint_ple_nmse": decoded_original_checkpoint_ple_nmse,
            "target_nmse": decoded_source_ple_nmse,
            "raw_nmse": decoded_source_ple_nmse,
            "d_subspace": d_sub,
            "exception_token_selector_count_min": int(exception_counts.min()) if exception_counts.size else None,
            "exception_token_selector_count_max": int(exception_counts.max()) if exception_counts.size else None,
            "exception_token_frequency_count_min": (
                int(exception_counts.min())
                if exception_counts.size and args.exception_selector in {"ppl_text_frequency", "c4_frequency"}
                else None
            ),
            "exception_token_frequency_count_max": (
                int(exception_counts.max())
                if exception_counts.size and args.exception_selector in {"ppl_text_frequency", "c4_frequency"}
                else None
            ),
            "exception_dout_score_min": float(exception_scores.min()) if exception_scores is not None and exception_scores.size else None,
            "exception_dout_score_max": float(exception_scores.max()) if exception_scores is not None and exception_scores.size else None,
            "compressed_file_schema_checked": True,
            "resident_tensor_byte_check_passed": True,
            "resident_tensor_bytes": components["storage_bytes"],
            "resident_file_roundtrip_max_abs_diff": roundtrip_max_abs_diff,
            "resident_roundtrip_max_abs_diff": roundtrip_max_abs_diff,
            "resident_roundtrip_atol": args.resident_roundtrip_atol,
            "resident_file_roundtrip_passed": roundtrip_max_abs_diff <= args.resident_roundtrip_atol,
            "resident_roundtrip_passed": roundtrip_max_abs_diff <= args.resident_roundtrip_atol,
            **components,
            "storage_gib": components["storage_bytes"] / 1024**3,
            "compression_ratio": compression_ratio,
            "input_sha256": sha256_file(source_path),
            "reference_it_weights_sha256": sha256_file(args.reference_it_weights) if args.source_ple else None,
            "dense_recon_sha256": sha256_file(recon_path),
            "compressed_ple_sha256": sha256_file(compressed_path),
            "compressed_ple_npz_total_bytes": compressed_path.stat().st_size,
            "elapsed_seconds": time.time() - t0,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"{LOG_PREFIX} decoded-source-PLE NMSE={decoded_source_ple_nmse:.6f}", flush=True)
    if decoded_original_checkpoint_ple_nmse is not None:
        print(
            f"{LOG_PREFIX} decoded-original-checkpoint-PLE NMSE="
            f"{decoded_original_checkpoint_ple_nmse:.6f}",
            flush=True,
        )
    print(
        f"{LOG_PREFIX} storage={components['storage_bytes']} bytes "
        f"ratio={compression_ratio:.4f}x roundtrip_max_abs_diff={roundtrip_max_abs_diff:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
