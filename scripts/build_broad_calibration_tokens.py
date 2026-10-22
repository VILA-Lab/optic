#!/usr/bin/env python3
"""Stream a small, broad, decontaminated calibration token set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/calibration/broad.json"
sys.path.insert(0, str(ROOT / "scripts"))
from build_eval_contamination_signatures import sha256_file, text_ngram_hashes  # noqa: E402
from check_disk_space import evaluate_storage_policy  # noqa: E402


def check_storage(config: dict[str, Any], phase: str, reserve_bytes: int = 0) -> dict[str, Any]:
    policy_path = Path(config["storage"]["policy"])
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    report = evaluate_storage_policy(
        policy, reserve_bytes=reserve_bytes, reserve_scope="corpus", phase=phase
    )
    if not report["ok"]:
        raise RuntimeError(f"storage policy failed at {phase}: {report['checks']}")
    return report


def load_signature_set(path: Path) -> set[int]:
    values = np.fromfile(path, dtype="<u8")
    if values.size and not bool(np.all(values[:-1] < values[1:])):
        raise RuntimeError("contamination signature file is not strictly sorted")
    return {int(value) for value in values}


def document_is_contaminated(text: str, signatures: set[int], width: int) -> bool:
    return any(value in signatures for value in text_ngram_hashes(text, width))


def sequence_chunks(
    token_ids: list[int], *, content_length: int, maximum: int
) -> list[list[int]]:
    chunks = []
    for start in range(0, len(token_ids) - content_length + 1, content_length):
        chunks.append(token_ids[start : start + content_length])
        if len(chunks) == maximum:
            break
    return chunks


def source_seed(base_seed: int, tag: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sequence = config["sequence"]
    if sum(int(source["sequences"]) for source in config["sources"]) != int(sequence["count"]):
        raise SystemExit("source sequence quotas do not sum to the configured total")
    if int(sequence["positions"]) != int(sequence["count"]) * int(sequence["length"]):
        raise SystemExit("configured position count is inconsistent")

    output_dir = Path(config["storage"]["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = output_dir / "eval_13gram_hashes_le_u64.bin"
    signature_manifest_path = output_dir / "eval_13gram_manifest.json"
    if not signature_path.is_file() or not signature_manifest_path.is_file():
        raise SystemExit("build the evaluation signature set first")
    output_path = output_dir / "calibration_token_ids_le_u32.bin"
    vocabulary_path = output_dir / "complete_tokenizer_row_ids_le_u32.bin"
    manifest_path = output_dir / "calibration_token_manifest.json"
    if output_path.exists() or vocabulary_path.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite an existing broad calibration token asset")

    reserve = int(config["storage"]["per_source_reserve_bytes"])
    check_storage(config, "pre_token_stream", reserve)
    signatures = load_signature_set(signature_path)
    width = int(config["decontamination"]["ngram_tokens"])

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

    length = int(sequence["length"])
    content_length = length - 1
    maximum_per_document = int(sequence["max_sequences_per_document"])
    all_sequences: list[list[int]] = []
    cumulative_unique_ids: set[int] = set()
    source_records: list[dict[str, Any]] = []
    base_seed = int(config["sampling"]["seed"])
    for source in config["sources"]:
        check_storage(config, f"pre_source_{source['tag']}", reserve)
        kwargs: dict[str, Any] = {
            "path": source["dataset"],
            "split": source["split"],
            "streaming": True,
            "revision": source["revision"],
        }
        if source.get("config"):
            kwargs["name"] = source["config"]
        stream = load_dataset(**kwargs).shuffle(
            seed=source_seed(base_seed, source["tag"]),
            buffer_size=int(config["sampling"]["stream_shuffle_buffer"]),
        )
        target = int(source["sequences"])
        accepted: list[list[int]] = []
        seen_documents = 0
        rejected_contamination = 0
        rejected_too_short = 0
        document_hashes: list[str] = []
        for row in stream:
            seen_documents += 1
            text = row.get(source["text_field"])
            if not isinstance(text, str) or not text.strip():
                rejected_too_short += 1
                continue
            if document_is_contaminated(text, signatures, width):
                rejected_contamination += 1
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            chunks = sequence_chunks(
                [int(value) for value in token_ids],
                content_length=content_length,
                maximum=maximum_per_document,
            )
            if not chunks:
                rejected_too_short += 1
                continue
            document_id = str(row.get(source.get("id_field", ""), ""))
            identity = hashlib.sha256(
                (source["tag"] + "\x1f" + document_id + "\x1f" + text).encode("utf-8")
            ).hexdigest()
            for chunk in chunks:
                accepted.append([int(tokenizer.bos_token_id), *chunk])
                document_hashes.append(identity)
                if len(accepted) == target:
                    break
            if seen_documents % 32 == 0:
                check_storage(config, f"streaming_{source['tag']}")
            if len(accepted) == target:
                break
        if len(accepted) != target:
            raise RuntimeError(
                f"source {source['tag']} exhausted before quota: {len(accepted)} != {target}"
            )
        all_sequences.extend(accepted)
        source_unique_ids = {
            token_id for accepted_sequence in accepted for token_id in accepted_sequence
        }
        cumulative_unique_ids.update(source_unique_ids)
        source_records.append({
            "tag": source["tag"],
            "dataset": source["dataset"],
            "config": source.get("config"),
            "revision": source["revision"],
            "split": source["split"],
            "target_sequences": target,
            "accepted_sequences": len(accepted),
            "seen_documents": seen_documents,
            "rejected_contamination_documents": rejected_contamination,
            "rejected_too_short_documents": rejected_too_short,
            "unique_token_ids": len(source_unique_ids),
            "unique_token_fraction": len(source_unique_ids) / int(config["token_coverage"]["vocab_size"]),
            "cumulative_unique_token_ids": len(cumulative_unique_ids),
            "cumulative_unique_token_fraction": len(cumulative_unique_ids) / int(config["token_coverage"]["vocab_size"]),
            "accepted_document_identity_hashes": document_hashes,
        })
        check_storage(config, f"post_source_{source['tag']}")
        progress_record = {
            key: value
            for key, value in source_records[-1].items()
            if key != "accepted_document_identity_hashes"
        }
        print(json.dumps(progress_record, sort_keys=True), flush=True)

    array = np.asarray(all_sequences, dtype="<u4")
    expected_shape = (int(sequence["count"]), length)
    if array.shape != expected_shape:
        raise RuntimeError(f"token array shape differs: {array.shape} != {expected_shape}")
    declared_vocab = int(config["token_coverage"]["vocab_size"])
    if int(tokenizer.vocab_size) != declared_vocab:
        raise RuntimeError(
            f"tokenizer vocabulary differs: {tokenizer.vocab_size} != {declared_vocab}"
        )
    if int(array.max()) >= declared_vocab:
        raise RuntimeError("calibration token id is outside tokenizer vocabulary")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    array.tofile(temporary)
    temporary.replace(output_path)
    vocabulary = np.arange(declared_vocab, dtype="<u4")
    vocabulary_temporary = vocabulary_path.with_suffix(vocabulary_path.suffix + ".tmp")
    vocabulary.tofile(vocabulary_temporary)
    vocabulary_temporary.replace(vocabulary_path)
    natural_unique = np.unique(array)
    combined_unique = np.union1d(natural_unique, vocabulary)
    combined_fraction = int(combined_unique.size) / declared_vocab
    required_combined_fraction = float(
        config["token_coverage"]["combined_fit_token_row_coverage_required"]
    )
    if combined_fraction != required_combined_fraction:
        raise RuntimeError(
            f"combined token-row coverage differs: {combined_fraction} != {required_combined_fraction}"
        )
    manifest = {
        "schema_version": "ple-broad-calibration-token-stream-v1",
        "protocol_id": config["protocol_id"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "signature_manifest": str(signature_manifest_path.resolve()),
        "signature_manifest_sha256": sha256_file(signature_manifest_path),
        "signature_file_sha256": sha256_file(signature_path),
        "tokenizer_identifier": config["model"]["identifier"],
        "tokenizer_revision": config["model"]["revision"],
        "bos_token_id": int(tokenizer.bos_token_id),
        "tokenizer_add_special_tokens": False,
        "context_reset_between_sequences": True,
        "sequence_count": expected_shape[0],
        "sequence_length": expected_shape[1],
        "positions": int(array.size),
        "token_file": str(output_path.resolve()),
        "token_file_sha256": sha256_file(output_path),
        "token_file_bytes": output_path.stat().st_size,
        "natural_context_unique_token_ids": int(natural_unique.size),
        "natural_context_unique_token_fraction": int(natural_unique.size) / declared_vocab,
        "complete_tokenizer_row_file": str(vocabulary_path.resolve()),
        "complete_tokenizer_row_file_sha256": sha256_file(vocabulary_path),
        "complete_tokenizer_row_file_bytes": vocabulary_path.stat().st_size,
        "complete_tokenizer_row_count": int(vocabulary.size),
        "complete_tokenizer_row_coverage_fraction": int(vocabulary.size) / declared_vocab,
        "combined_fit_unique_token_rows": int(combined_unique.size),
        "combined_fit_token_row_coverage_fraction": combined_fraction,
        "complete_tokenizer_row_stratum_is_natural_context": False,
        "raw_source_text_persisted": False,
        "sources": source_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    check_storage(config, "post_token_stream")
    summary = {
        key: value for key, value in manifest.items()
        if key not in {"sources"}
    }
    summary["source_count"] = len(source_records)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
