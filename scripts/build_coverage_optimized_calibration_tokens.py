#!/usr/bin/env python3
"""Build a broad calibration stream chosen for natural tokenizer coverage."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/calibration/coverage.json"
sys.path.insert(0, str(ROOT / "scripts"))
from build_broad_calibration_tokens import (  # noqa: E402
    document_is_contaminated,
    load_signature_set,
    source_seed,
)
from build_eval_contamination_signatures import sha256_file  # noqa: E402
from check_disk_space import evaluate_storage_policy  # noqa: E402


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def check_storage(
    config: dict[str, Any], phase: str, reserve_bytes: int = 0
) -> dict[str, Any]:
    policy = json.loads(Path(config["storage"]["policy"]).read_text(encoding="utf-8"))
    report = evaluate_storage_policy(
        policy, reserve_bytes=reserve_bytes, reserve_scope="corpus", phase=phase
    )
    if not report["ok"]:
        raise RuntimeError(f"storage policy failed at {phase}: {report['checks']}")
    return report


def validate_config(config: dict[str, Any]) -> None:
    sequence = config["sequence"]
    pool = config["candidate_pool"]
    sources = config["sources"]
    candidate_count = sum(int(source["candidate_sequences"]) for source in sources)
    selected_floor = sum(int(source["minimum_selected_sequences"]) for source in sources)
    if candidate_count != int(pool["sequence_count"]):
        raise ValueError(f"candidate quota mismatch: {candidate_count}")
    if int(pool["positions"]) != candidate_count * int(sequence["length"]):
        raise ValueError("candidate position count is inconsistent")
    if int(sequence["selected_positions"]) != (
        int(sequence["selected_count"]) * int(sequence["length"])
    ):
        raise ValueError("selected position count is inconsistent")
    if selected_floor > int(sequence["selected_count"]):
        raise ValueError("source floors exceed selected sequence count")
    tags = [source["tag"] for source in sources]
    if len(tags) != len(set(tags)):
        raise ValueError("source tags must be unique")
    vocab_size = int(config["token_coverage"]["vocab_size"])
    pool_minimum = int(config["token_coverage"]["candidate_pool_minimum_unique_token_ids"])
    selected_minimum = int(
        config["token_coverage"]["selected_context_minimum_unique_token_ids"]
    )
    if pool_minimum != math.ceil(
        vocab_size * float(pool["minimum_unique_token_fraction"])
    ):
        raise ValueError("candidate coverage count and fraction differ")
    if selected_minimum != math.ceil(
        vocab_size * float(pool["minimum_selected_unique_token_fraction"])
    ):
        raise ValueError("selected coverage count and fraction differ")


def candidate_windows(
    token_ids: Sequence[int], *, content_length: int, stride: int
) -> Iterable[np.ndarray]:
    if len(token_ids) < content_length:
        return
    final_start = len(token_ids) - content_length
    starts = list(range(0, final_start + 1, stride))
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    for start in starts:
        yield np.asarray(token_ids[start : start + content_length], dtype=np.uint32)


def select_document_chunks(
    token_ids: Sequence[int], *, content_length: int, stride: int,
    maximum: int, covered: np.ndarray
) -> list[np.ndarray]:
    """Choose non-identical document windows with the largest new-ID gain."""
    windows = list(candidate_windows(token_ids, content_length=content_length, stride=stride))
    selected: list[np.ndarray] = []
    selected_starts: set[bytes] = set()
    for _ in range(min(maximum, len(windows))):
        best: np.ndarray | None = None
        best_key: tuple[int, int] | None = None
        for order, window in enumerate(windows):
            identity = window.tobytes()
            if identity in selected_starts:
                continue
            unique = np.unique(window)
            gain = int(np.count_nonzero(~covered[unique]))
            key = (gain, -order)
            if best_key is None or key > best_key:
                best = window
                best_key = key
        if best is None:
            break
        selected.append(best)
        selected_starts.add(best.tobytes())
        covered[np.unique(best)] = True
    return selected


def sequence_unique_ids(array: np.ndarray) -> list[np.ndarray]:
    """Exclude column zero because it is the manually prepended BOS token."""
    return [np.unique(row[1:]).astype(np.uint32, copy=False) for row in array]


def lazy_greedy_select(
    uniques: Sequence[np.ndarray], candidate_indices: Iterable[int], count: int,
    covered: np.ndarray, selected_mask: np.ndarray
) -> list[int]:
    """Deterministic lazy greedy maximum coverage with exact marginal gains."""
    heap: list[tuple[int, int]] = []
    for index in candidate_indices:
        if selected_mask[index]:
            continue
        gain = int(np.count_nonzero(~covered[uniques[index]]))
        heapq.heappush(heap, (-gain, int(index)))
    selected: list[int] = []
    while len(selected) < count:
        if not heap:
            raise RuntimeError("candidate heap exhausted before selection quota")
        negative_bound, index = heapq.heappop(heap)
        if selected_mask[index]:
            continue
        exact_gain = int(np.count_nonzero(~covered[uniques[index]]))
        next_bound = -heap[0][0] if heap else -1
        if exact_gain >= next_bound:
            selected_mask[index] = True
            covered[uniques[index]] = True
            selected.append(index)
        else:
            heapq.heappush(heap, (-exact_gain, index))
    return selected


def source_ranges(sources: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    result = []
    start = 0
    for source in sources:
        end = start + int(source["candidate_sequences"])
        result.append((start, end))
        start = end
    return result


def write_checkpoint(path: Path, **values: Any) -> None:
    atomic_json(path, {"schema_version": "ple-coverage-pool-checkpoint-v1", **values})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)

    sequence = config["sequence"]
    pool = config["candidate_pool"]
    coverage_config = config["token_coverage"]
    sources = config["sources"]
    output_dir = Path(config["storage"]["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_token_ids_le_u32.bin.partial"
    source_id_path = output_dir / "candidate_source_ids_le_u16.bin.partial"
    checkpoint_path = output_dir / "candidate_pool_checkpoint.json"
    manifest_path = output_dir / "calibration_token_manifest.json"
    selected_path = output_dir / "calibration_token_ids_le_u32.bin"
    vocabulary_path = output_dir / "complete_tokenizer_row_ids_le_u32.bin"
    if manifest_path.exists() or selected_path.exists():
        raise SystemExit("refusing to overwrite an existing coverage-optimized calibration asset")

    signature_path = Path(config["decontamination"]["signature_file"])
    signature_manifest_path = Path(config["decontamination"]["signature_manifest"])
    if sha256_file(signature_path) != config["decontamination"]["signature_file_sha256"]:
        raise RuntimeError("evaluation signature hash differs")
    if sha256_file(signature_manifest_path) != config["decontamination"]["signature_manifest_sha256"]:
        raise RuntimeError("evaluation signature manifest hash differs")

    length = int(sequence["length"])
    candidate_count = int(pool["sequence_count"])
    candidate_bytes = candidate_count * length * np.dtype("<u4").itemsize
    source_id_bytes = candidate_count * np.dtype("<u2").itemsize
    reserve = candidate_bytes + source_id_bytes + int(config["storage"]["per_source_reserve_bytes"])
    check_storage(config, "pre_candidate_pool", reserve)

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config_sha256") != sha256_file(args.config):
            raise RuntimeError("checkpoint belongs to a different configuration")
        completed_count = int(checkpoint["completed_candidate_sequences"])
        source_records = list(checkpoint["completed_sources"])
        current = checkpoint.get("current_source")
    else:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate = np.memmap(
            candidate_path, dtype="<u4", mode="w+", shape=(candidate_count, length)
        )
        candidate.flush()
        del candidate
        source_ids = np.memmap(
            source_id_path, dtype="<u2", mode="w+", shape=(candidate_count,)
        )
        source_ids.flush()
        del source_ids
        completed_count = 0
        source_records: list[dict[str, Any]] = []
        current = None
        write_checkpoint(
            checkpoint_path,
            config_sha256=sha256_file(args.config),
            completed_candidate_sequences=0,
            completed_sources=[],
            current_source=None,
        )

    if candidate_path.stat().st_size != candidate_bytes:
        raise RuntimeError("candidate token pool has an unexpected size")
    if source_id_path.stat().st_size != source_id_bytes:
        raise RuntimeError("candidate source-id pool has an unexpected size")
    candidates = np.memmap(
        candidate_path, dtype="<u4", mode="r+", shape=(candidate_count, length)
    )
    source_ids = np.memmap(
        source_id_path, dtype="<u2", mode="r+", shape=(candidate_count,)
    )
    vocab_size = int(coverage_config["vocab_size"])
    candidate_covered = np.zeros(vocab_size, dtype=bool)
    if completed_count:
        values = np.unique(candidates[:completed_count, 1:])
        if values.size and int(values[-1]) >= vocab_size:
            raise RuntimeError("checkpoint contains an out-of-vocabulary token ID")
        candidate_covered[values] = True

    cache_root = Path(config["storage"]["isolated_huggingface_cache"])
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["local_snapshot"], local_files_only=True, trust_remote_code=True
    )
    if tokenizer.bos_token_id is None:
        raise RuntimeError("tokenizer has no BOS token")
    if int(tokenizer.vocab_size) != vocab_size:
        raise RuntimeError("tokenizer vocabulary differs from the configuration")
    signatures = load_signature_set(signature_path)
    ngram_width = int(config["decontamination"]["ngram_tokens"])
    content_length = length - 1
    stride = int(pool["window_stride"])
    maximum_document_tokens = int(pool["maximum_tokens_per_document"])
    checkpoint_every = int(pool["checkpoint_every_sequences"])
    maximum_per_document = int(sequence["max_sequences_per_document"])
    ranges = source_ranges(sources)

    for source_index, (source, bounds) in enumerate(zip(sources, ranges)):
        start, end = bounds
        if end <= completed_count:
            if not any(record["tag"] == source["tag"] for record in source_records):
                if not current or current.get("tag") != source["tag"]:
                    raise RuntimeError("completed source is missing checkpoint statistics")
                source_unique = np.unique(candidates[start:end, 1:])
                source_records.append({
                    "tag": source["tag"],
                    "dataset": source["dataset"],
                    "config": source.get("config"),
                    "revision": source["revision"],
                    "split": source["split"],
                    "candidate_start": start,
                    "candidate_end": end,
                    "candidate_sequences": end - start,
                    "minimum_selected_sequences": int(source["minimum_selected_sequences"]),
                    "seen_documents": int(current["seen_documents"]),
                    "rejected_contamination_documents": int(current["rejected_contamination_documents"]),
                    "rejected_too_short_documents": int(current["rejected_too_short_documents"]),
                    "natural_unique_token_ids": int(source_unique.size),
                    "natural_unique_token_fraction": int(source_unique.size) / vocab_size,
                    "cumulative_natural_unique_token_ids": int(np.count_nonzero(candidate_covered)),
                    "cumulative_natural_unique_token_fraction": int(np.count_nonzero(candidate_covered)) / vocab_size,
                })
                current = None
                write_checkpoint(
                    checkpoint_path,
                    config_sha256=sha256_file(args.config),
                    completed_candidate_sequences=end,
                    completed_sources=source_records,
                    current_source=None,
                )
            continue
        if completed_count < start:
            raise RuntimeError("checkpoint candidate count does not align to source ranges")
        check_storage(config, f"pre_source_{source['tag']}", int(config["storage"]["per_source_reserve_bytes"]))
        kwargs: dict[str, Any] = {
            "path": source["dataset"],
            "split": source["split"],
            "streaming": True,
            "revision": source["revision"],
        }
        if source.get("config"):
            kwargs["name"] = source["config"]
        stream = iter(load_dataset(**kwargs).shuffle(
            seed=source_seed(int(pool["seed"]), source["tag"]),
            buffer_size=int(pool["stream_shuffle_buffer"]),
        ))
        accepted = completed_count - start
        seen_documents = 0
        rejected_contamination = 0
        rejected_too_short = 0
        if current and current.get("tag") == source["tag"]:
            seen_documents = int(current["seen_documents"])
            rejected_contamination = int(current["rejected_contamination_documents"])
            rejected_too_short = int(current["rejected_too_short_documents"])
            for _ in range(seen_documents):
                next(stream)
        elif accepted:
            raise RuntimeError("partial source checkpoint is missing resume metadata")

        while start + accepted < end:
            row = next(stream)
            seen_documents += 1
            text = row.get(source["text_field"])
            if not isinstance(text, str) or not text.strip():
                rejected_too_short += 1
                continue
            if document_is_contaminated(text, signatures, ngram_width):
                rejected_contamination += 1
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids = [int(value) for value in token_ids[:maximum_document_tokens]]
            chunks = select_document_chunks(
                token_ids,
                content_length=content_length,
                stride=stride,
                maximum=min(maximum_per_document, end - start - accepted),
                covered=candidate_covered,
            )
            if not chunks:
                rejected_too_short += 1
                continue
            for chunk in chunks:
                if chunk.size != content_length or (chunk.size and int(chunk.max()) >= vocab_size):
                    raise RuntimeError("candidate chunk is invalid")
                write_index = start + accepted
                candidates[write_index, 0] = int(tokenizer.bos_token_id)
                candidates[write_index, 1:] = chunk
                source_ids[write_index] = source_index
                accepted += 1
            completed_count = start + accepted
            if accepted % checkpoint_every < len(chunks) or completed_count == end:
                candidates.flush()
                source_ids.flush()
                current = {
                    "tag": source["tag"],
                    "source_index": source_index,
                    "accepted_sequences": accepted,
                    "seen_documents": seen_documents,
                    "rejected_contamination_documents": rejected_contamination,
                    "rejected_too_short_documents": rejected_too_short,
                }
                write_checkpoint(
                    checkpoint_path,
                    config_sha256=sha256_file(args.config),
                    completed_candidate_sequences=completed_count,
                    completed_sources=source_records,
                    current_source=current,
                )
                check_storage(config, f"streaming_{source['tag']}")

        source_unique = np.unique(candidates[start:end, 1:])
        source_record = {
            "tag": source["tag"],
            "dataset": source["dataset"],
            "config": source.get("config"),
            "revision": source["revision"],
            "split": source["split"],
            "candidate_start": start,
            "candidate_end": end,
            "candidate_sequences": end - start,
            "minimum_selected_sequences": int(source["minimum_selected_sequences"]),
            "seen_documents": seen_documents,
            "rejected_contamination_documents": rejected_contamination,
            "rejected_too_short_documents": rejected_too_short,
            "natural_unique_token_ids": int(source_unique.size),
            "natural_unique_token_fraction": int(source_unique.size) / vocab_size,
            "cumulative_natural_unique_token_ids": int(np.count_nonzero(candidate_covered)),
            "cumulative_natural_unique_token_fraction": int(np.count_nonzero(candidate_covered)) / vocab_size,
        }
        source_records.append(source_record)
        current = None
        write_checkpoint(
            checkpoint_path,
            config_sha256=sha256_file(args.config),
            completed_candidate_sequences=end,
            completed_sources=source_records,
            current_source=None,
        )
        print(json.dumps(source_record, sort_keys=True), flush=True)
        check_storage(config, f"post_source_{source['tag']}")

    candidates.flush()
    source_ids.flush()
    if completed_count != candidate_count:
        raise RuntimeError("candidate pool ended before its configured size")
    pool_unique_count = int(np.count_nonzero(candidate_covered))
    pool_unique_fraction = pool_unique_count / vocab_size
    pool_minimum = int(coverage_config["candidate_pool_minimum_unique_token_ids"])

    final_candidate_path = output_dir / "candidate_token_ids_le_u32.bin"
    final_source_id_path = output_dir / "candidate_source_ids_le_u16.bin"
    del candidates
    del source_ids
    candidate_path.replace(final_candidate_path)
    source_id_path.replace(final_source_id_path)
    checkpoint_path.unlink()
    candidates = np.memmap(
        final_candidate_path, dtype="<u4", mode="r", shape=(candidate_count, length)
    )

    manifest: dict[str, Any] = {
        "schema_version": "ple-coverage-optimized-calibration-token-stream-v2",
        "protocol_id": config["protocol_id"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "candidate_token_file": str(final_candidate_path.resolve()),
        "candidate_token_file_sha256": sha256_file(final_candidate_path),
        "candidate_token_file_bytes": final_candidate_path.stat().st_size,
        "candidate_source_id_file": str(final_source_id_path.resolve()),
        "candidate_source_id_file_sha256": sha256_file(final_source_id_path),
        "candidate_sequence_count": candidate_count,
        "candidate_positions": int(pool["positions"]),
        "candidate_natural_unique_token_ids": pool_unique_count,
        "candidate_natural_unique_token_fraction": pool_unique_fraction,
        "candidate_minimum_unique_token_ids": pool_minimum,
        "candidate_coverage_gate_passed": pool_unique_count >= pool_minimum,
        "manual_bos_excluded_from_natural_coverage": True,
        "raw_source_text_persisted": False,
        "signature_file": str(signature_path.resolve()),
        "signature_file_sha256": sha256_file(signature_path),
        "sources": source_records,
    }
    if pool_unique_count < pool_minimum:
        manifest["gate_cache_allowed"] = False
        manifest["status"] = "candidate_coverage_gate_failed"
        atomic_json(manifest_path, manifest)
        print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2, sort_keys=True))
        raise SystemExit(3)

    uniques = sequence_unique_ids(candidates)
    selected_mask = np.zeros(candidate_count, dtype=bool)
    selected_covered = np.zeros(vocab_size, dtype=bool)
    selected_indices: list[int] = []
    per_source_floor_selected: dict[str, int] = {}
    for source, (start, end) in zip(sources, ranges):
        count = int(source["minimum_selected_sequences"])
        chosen = lazy_greedy_select(
            uniques, range(start, end), count, selected_covered, selected_mask
        )
        selected_indices.extend(chosen)
        per_source_floor_selected[source["tag"]] = len(chosen)
    remaining = int(sequence["selected_count"]) - len(selected_indices)
    chosen = lazy_greedy_select(
        uniques, range(candidate_count), remaining, selected_covered, selected_mask
    )
    selected_indices.extend(chosen)
    per_source_selected = {
        source["tag"]: sum(start <= index < end for index in selected_indices)
        for source, (start, end) in zip(sources, ranges)
    }
    selected_array = np.asarray(candidates[selected_indices], dtype="<u4")
    selected_temporary = selected_path.with_suffix(selected_path.suffix + ".tmp")
    selected_array.tofile(selected_temporary)
    selected_temporary.replace(selected_path)
    selected_unique_count = int(np.count_nonzero(selected_covered))
    selected_minimum = int(coverage_config["selected_context_minimum_unique_token_ids"])
    selected_passed = selected_unique_count >= selected_minimum

    vocabulary = np.arange(vocab_size, dtype="<u4")
    vocabulary_temporary = vocabulary_path.with_suffix(vocabulary_path.suffix + ".tmp")
    vocabulary.tofile(vocabulary_temporary)
    vocabulary_temporary.replace(vocabulary_path)
    manifest.update({
        "status": "ready_for_gate_cache" if selected_passed else "selected_coverage_gate_failed",
        "gate_cache_allowed": selected_passed,
        "selection_method": pool["selection"],
        "selected_sequence_count": int(selected_array.shape[0]),
        "selected_sequence_length": int(selected_array.shape[1]),
        "selected_positions": int(selected_array.size),
        "selected_token_file": str(selected_path.resolve()),
        "selected_token_file_sha256": sha256_file(selected_path),
        "selected_token_file_bytes": selected_path.stat().st_size,
        "selected_candidate_indices": selected_indices,
        "selected_source_floor_counts": per_source_floor_selected,
        "selected_per_source": per_source_selected,
        "selected_natural_unique_token_ids": selected_unique_count,
        "selected_natural_unique_token_fraction": selected_unique_count / vocab_size,
        "selected_minimum_unique_token_ids": selected_minimum,
        "selected_coverage_gate_passed": selected_passed,
        "complete_tokenizer_row_file": str(vocabulary_path.resolve()),
        "complete_tokenizer_row_file_sha256": sha256_file(vocabulary_path),
        "complete_tokenizer_row_count": vocab_size,
        "complete_tokenizer_row_coverage_fraction": 1.0,
        "complete_tokenizer_row_stratum_is_natural_context": False,
    })
    atomic_json(manifest_path, manifest)
    check_storage(config, "post_coverage_selection")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"sources", "selected_candidate_indices"}}, indent=2, sort_keys=True))
    if not selected_passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
