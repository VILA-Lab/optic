#!/usr/bin/env python3
"""Count calibration-token frequencies on the fitting split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/calibration/token_counts.json"
SPLIT_MODULUS = 5
VALIDATION_REMAINDER = 0
TOKENIZER_IDENTITY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
)


class InputUncertaintyError(RuntimeError):
    """The declared inputs do not uniquely determine the requested split."""


class SplitLeakageError(RuntimeError):
    """Identical sequence content would cross the fit/validation boundary."""


@dataclass(frozen=True)
class SequenceRecord:
    collection: str
    ordinal: int
    identity: str
    content_sha256: str
    partition: str
    token_count: int


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_partition(identity: str) -> str:
    digest_value = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big")
    return "validation" if digest_value % SPLIT_MODULUS == VALIDATION_REMAINDER else "fit"


def validate_natural_identity_geometry(
    declared_input: Mapping[str, Any], token_manifest: Mapping[str, Any]
) -> tuple[int, int]:
    """Require a one-to-one mapping from declared contexts to candidate IDs."""
    manifest_count = int(token_manifest.get("selected_sequence_count", -1))
    manifest_length = int(token_manifest.get("selected_sequence_length", -1))
    declared_count = int(declared_input.get("sequence_count", -1))
    declared_length = int(declared_input.get("sequence_length", -1))
    if manifest_count <= 0 or manifest_length <= 0:
        raise InputUncertaintyError("broad manifest lacks selected sequence geometry")
    if declared_count <= 0 or declared_length <= 0:
        raise InputUncertaintyError("config lacks broad sequence geometry")
    if manifest_count * manifest_length != declared_count * declared_length:
        raise InputUncertaintyError(
            "broad manifest and config disagree on total token positions"
        )
    if (manifest_count, manifest_length) != (declared_count, declared_length):
        raise InputUncertaintyError(
            "broad identity units are "
            f"{manifest_count}x{manifest_length}, but the config declares contexts of "
            f"{declared_count}x{declared_length}; no context-assembly rule is defined"
        )
    selected = token_manifest.get("selected_candidate_indices")
    if not isinstance(selected, list) or len(selected) != manifest_count:
        raise InputUncertaintyError(
            "selected_candidate_indices do not map one-to-one to broad sequences"
        )
    return manifest_count, manifest_length


def validate_split_records(records: Iterable[SequenceRecord]) -> dict[str, int]:
    """Reject ambiguous identities and content duplicated across partitions."""
    identity_content: dict[str, str] = {}
    content_partitions: dict[str, set[str]] = {}
    content_occurrences: dict[str, int] = {}
    counts = {"fit": 0, "validation": 0}
    for record in records:
        if record.partition not in counts:
            raise InputUncertaintyError(f"unknown split partition {record.partition!r}")
        if record.partition != identity_partition(record.identity):
            raise InputUncertaintyError("stored partition differs from the hash rule")
        previous = identity_content.get(record.identity)
        if previous is not None:
            if previous != record.content_sha256:
                raise InputUncertaintyError(
                    "one sequence identity maps to multiple token contents"
                )
            raise InputUncertaintyError("sequence identity is not unique")
        identity_content[record.identity] = record.content_sha256
        content_partitions.setdefault(record.content_sha256, set()).add(record.partition)
        content_occurrences[record.content_sha256] = (
            content_occurrences.get(record.content_sha256, 0) + 1
        )
        counts[record.partition] += 1
    leaking = sorted(
        digest for digest, partitions in content_partitions.items() if len(partitions) > 1
    )
    if leaking:
        raise SplitLeakageError(
            f"{len(leaking)} token sequences occur in both fit and validation"
        )
    if not counts["fit"] or not counts["validation"]:
        raise InputUncertaintyError("identity split produced an empty partition")
    counts["total"] = counts["fit"] + counts["validation"]
    counts["duplicate_content_within_partition"] = sum(
        occurrences - 1
        for digest, occurrences in content_occurrences.items()
        if len(content_partitions[digest]) == 1
    )
    return counts


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InputUncertaintyError(f"{path} must contain a JSON object")
    return value


def _verify_file(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise InputUncertaintyError(
            f"{role} SHA-256 {actual} != {expected_sha256}"
        )
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def _source_for_candidate(
    candidate_index: int, sources: list[dict[str, Any]]
) -> dict[str, Any]:
    matches = [
        source
        for source in sources
        if int(source["candidate_start"]) <= candidate_index < int(source["candidate_end"])
    ]
    if len(matches) != 1:
        raise InputUncertaintyError(
            f"candidate index {candidate_index} maps to {len(matches)} broad sources"
        )
    return matches[0]


def _sequence_digest(row: np.ndarray) -> str:
    canonical = np.ascontiguousarray(row, dtype="<u4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def broad_sequence_records(
    declared_input: Mapping[str, Any],
    manifest: Mapping[str, Any],
    token_file: Path,
) -> tuple[list[SequenceRecord], np.ndarray]:
    count, length = validate_natural_identity_geometry(declared_input, manifest)
    selected = [int(value) for value in manifest["selected_candidate_indices"]]
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise InputUncertaintyError("broad manifest lacks source ranges")
    expected_bytes = count * length * np.dtype("<u4").itemsize
    if token_file.stat().st_size != expected_bytes:
        raise InputUncertaintyError("broad token-file bytes do not match identity geometry")
    tokens = np.memmap(token_file, mode="r", dtype="<u4", shape=(count, length))
    records: list[SequenceRecord] = []
    for ordinal, candidate_index in enumerate(selected):
        source = _source_for_candidate(candidate_index, sources)
        identity = canonical_json_bytes(
            [
                source["dataset"],
                source["revision"],
                source.get("config"),
                candidate_index,
            ]
        ).decode("utf-8")
        records.append(
            SequenceRecord(
                collection="broad_natural",
                ordinal=ordinal,
                identity=identity,
                content_sha256=_sequence_digest(tokens[ordinal]),
                partition=identity_partition(identity),
                token_count=length,
            )
        )
    return records, tokens


def self_generated_sequence_records(
    stream_id: str,
    declared_stream: Mapping[str, Any],
    token_file: Path,
) -> tuple[list[SequenceRecord], np.ndarray]:
    count = int(declared_stream["sequence_count"])
    length = int(declared_stream["sequence_length"])
    expected_bytes = count * length * np.dtype("<u4").itemsize
    if token_file.stat().st_size != expected_bytes:
        raise InputUncertaintyError(
            f"self-generated stream {stream_id} byte geometry differs"
        )
    tokens = np.memmap(token_file, mode="r", dtype="<u4", shape=(count, length))
    records: list[SequenceRecord] = []
    for ordinal in range(count):
        content_sha256 = _sequence_digest(tokens[ordinal])
        identity = canonical_json_bytes([stream_id, ordinal, content_sha256]).decode("utf-8")
        records.append(
            SequenceRecord(
                collection=f"self_generated_{stream_id}",
                ordinal=ordinal,
                identity=identity,
                content_sha256=content_sha256,
                partition=identity_partition(identity),
                token_count=length,
            )
        )
    return records, tokens


def _validate_config(config: Mapping[str, Any]) -> None:
    model = config["model"]
    split = config["split"]
    if (
        float(split["validation_fraction"]) != 0.2
        or float(split["fit_fraction"]) != 0.8
        or split["hash"] != "sha256"
        or split["validation_rule"]
        != "unsigned_big_endian_sha256(sequence_identity_utf8) mod 5 == 0"
    ):
        raise InputUncertaintyError("identity split rule differs")
    controls = [int(value) for value in config["fit_inputs"]["control_token_ids"]]
    if len(controls) != 24 or len(set(controls)) != 24:
        raise InputUncertaintyError("the token-count config requires exactly 24 unique control IDs")
    if any(value < 0 or value >= int(model["vocab_size"]) for value in controls):
        raise InputUncertaintyError("control token ID is outside the vocabulary")


def _tokenizer_identity(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot = _resolve(config["model"]["tokenizer_snapshot"])
    files = []
    for name in TOKENIZER_IDENTITY_FILES:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"missing tokenizer identity file: {path}")
        files.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    identity = {
        "model_identifier": config["model"]["identifier"],
        "model_revision": config["model"]["revision"],
        "files": [{key: item[key] for key in ("name", "sha256", "bytes")} for item in files],
    }
    identity["sha256"] = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return identity, files


def _validate_control_ids(
    controls: list[int], stream_manifests: Iterable[Mapping[str, Any]]
) -> None:
    for manifest in stream_manifests:
        declared = manifest.get("tokenizer_special_ids")
        if declared != controls:
            raise InputUncertaintyError(
                "checkpoint-self-generated tokenizer control IDs differ from the config"
            )
        if manifest.get("external_text_or_benchmark_examples_used") is not False:
            raise InputUncertaintyError(
                "self-generated stream does not exclude external/benchmark examples"
            )
        if manifest.get("quality_evaluation_allowed") is not False:
            raise InputUncertaintyError("self-generated stream is marked quality-evaluable")


def _validate_complete_rows(path: Path, vocab_size: int) -> None:
    expected_bytes = vocab_size * np.dtype("<u4").itemsize
    if path.stat().st_size != expected_bytes:
        raise InputUncertaintyError("complete tokenizer row file has the wrong byte count")
    rows = np.memmap(path, mode="r", dtype="<u4", shape=(vocab_size,))
    if not bool(np.array_equal(rows, np.arange(vocab_size, dtype="<u4"))):
        raise InputUncertaintyError("complete tokenizer rows are not canonical arange(V)")






def _write_bytes_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_bytes_new(
        temporary,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    os.replace(temporary, path)


def _output_receipt(staged_path: Path, final_path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(final_path.resolve()),
        "sha256": sha256_file(staged_path),
        "bytes": staged_path.stat().st_size,
    }


def build_token_counts(config_path: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    config_sha256 = sha256_file(config_path)
    fit_inputs = config["fit_inputs"]
    input_receipts: list[dict[str, Any]] = []

    broad = fit_inputs["broad_natural"]
    broad_manifest_path = _resolve(broad["token_manifest"])
    broad_token_path = _resolve(broad["token_file"])
    input_receipts.extend(
        [
            _verify_file(_resolve(broad["fit_config"]), broad["fit_config_sha256"], "broad_fit_config"),
            _verify_file(broad_manifest_path, broad["token_manifest_sha256"], "broad_token_manifest"),
            _verify_file(broad_token_path, broad["token_file_sha256"], "broad_token_file"),
            _verify_file(_resolve(broad["gate_file"]), broad["gate_file_sha256"], "broad_gate_file"),
            _verify_file(_resolve(broad["gate_manifest"]), broad["gate_manifest_sha256"], "broad_gate_manifest"),
        ]
    )
    broad_manifest = _load_json(broad_manifest_path)
    if (
        broad_manifest.get("selected_token_file_sha256") != broad["token_file_sha256"]
        or Path(str(broad_manifest.get("selected_token_file", ""))).resolve()
        != broad_token_path.resolve()
    ):
        raise InputUncertaintyError("broad manifest token binding differs from the config")

    selfgen = fit_inputs["checkpoint_self_generated"]
    stream_inputs: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for stream in selfgen["streams"]:
        token_path = _resolve(stream["token_file"])
        manifest_path = _resolve(stream["token_manifest"])
        input_receipts.append(
            _verify_file(token_path, stream["token_file_sha256"], f"selfgen_{stream['stream_id']}_tokens")
        )
        input_receipts.append(
            _verify_file(manifest_path, stream["token_manifest_sha256"], f"selfgen_{stream['stream_id']}_manifest")
        )
        stream_manifest = _load_json(manifest_path)
        if (
            int(stream_manifest.get("sequence_count", -1)) != int(stream["sequence_count"])
            or int(stream_manifest.get("sequence_length", -1))
            != int(stream["sequence_length"])
            or stream_manifest.get("token_file_sha256") != stream["token_file_sha256"]
            or Path(str(stream_manifest.get("token_file", ""))).resolve()
            != token_path.resolve()
        ):
            raise InputUncertaintyError(
                f"self-generated stream {stream['stream_id']} manifest binding differs"
            )
        stream_inputs.append((stream, token_path, stream_manifest))

    complete = fit_inputs["complete_tokenizer_rows"]
    complete_path = _resolve(complete["path"])
    input_receipts.append(
        _verify_file(complete_path, complete["sha256"], "complete_tokenizer_rows")
    )
    model_path = _resolve(config["model"]["weights"])
    input_receipts.append(
        _verify_file(model_path, config["model"]["weights_sha256"], "model_weights")
    )
    tokenizer_identity, tokenizer_files = _tokenizer_identity(config)
    input_receipts.extend(
        {
            "role": f"tokenizer_identity_{item['name']}",
            "path": item["path"],
            "sha256": item["sha256"],
            "bytes": item["bytes"],
        }
        for item in tokenizer_files
    )

    controls = [int(value) for value in fit_inputs["control_token_ids"]]
    _validate_control_ids(controls, [manifest for _, _, manifest in stream_inputs])
    _validate_complete_rows(complete_path, int(config["model"]["vocab_size"]))

    broad_records, broad_tokens = broad_sequence_records(
        broad, broad_manifest, broad_token_path
    )
    collections: list[tuple[str, list[SequenceRecord], np.ndarray]] = [
        ("broad_natural", broad_records, broad_tokens)
    ]
    for stream, token_path, _manifest in stream_inputs:
        records, tokens = self_generated_sequence_records(
            str(stream["stream_id"]), stream, token_path
        )
        collections.append((f"self_generated_{stream['stream_id']}", records, tokens))
    all_records = [record for _, records, _ in collections for record in records]
    split_counts = validate_split_records(all_records)

    vocab_size = int(config["model"]["vocab_size"])
    fit_frequency = np.zeros(vocab_size, dtype="<u8")
    partition_indices: dict[str, dict[str, np.ndarray]] = {}
    for name, records, tokens in collections:
        indices: dict[str, list[int]] = {"fit": [], "validation": []}
        for record in records:
            indices[record.partition].append(record.ordinal)
        if int(tokens.max()) >= vocab_size:
            raise InputUncertaintyError(f"{name} contains an out-of-vocabulary token ID")
        fit_indices = np.asarray(indices["fit"], dtype=np.int64)
        fit_tokens = np.asarray(tokens[fit_indices], dtype=np.int64).reshape(-1)
        fit_frequency += np.bincount(
            fit_tokens, minlength=vocab_size
        ).astype("<u8", copy=False)
        partition_indices[name] = {
            partition: np.asarray(values, dtype="<u4")
            for partition, values in indices.items()
        }

    output_root = _resolve(config["split"]["split_manifest_output"]).parent
    fixed_binary_output_bytes = int(fit_frequency.nbytes) + sum(
        int(value.nbytes)
        for partitions in partition_indices.values()
        for value in partitions.values()
    )
    preflight = {
        "schema_version": "frequency-tiered-token-count-preflight-v1",
        "status": "ready_to_build" if not preflight_only else "preflight_only",
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "benchmark_inputs_used": False,
        "quality_evaluation_allowed": False,
        "input_bytes": sum(int(item["bytes"]) for item in input_receipts),
        "fixed_binary_output_bytes": fixed_binary_output_bytes,
        "sequence_split_counts": split_counts,
        "tokenizer_identity": tokenizer_identity,
        "control_token_ids": controls,
        "output_root": str(output_root.resolve()),
    }
    if preflight_only:
        return preflight
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite token-count output: {output_root}")
    staging = output_root.with_name(output_root.name + f".partial-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to reuse partial token-count output: {staging}")
    staging.mkdir(parents=True)
    outputs: list[dict[str, Any]] = []

    frequency_staged = staging / "fit_token_frequency_le_u64.bin"
    _write_bytes_new(frequency_staged, fit_frequency.tobytes(order="C"))
    outputs.append(
        _output_receipt(
            frequency_staged,
            output_root / frequency_staged.name,
            "fit_only_token_frequency_uint64",
        )
    )
    outputs[-1].update(
        {
            "dtype": "uint64_little_endian",
            "shape": [vocab_size],
            "partition": "fit",
            "validation_sequences_used": False,
        }
    )

    for collection, partitions in sorted(partition_indices.items()):
        for partition, indices in sorted(partitions.items()):
            name = f"{collection}.{partition}_indices_le_u32.bin"
            staged = staging / name
            _write_bytes_new(staged, indices.tobytes(order="C"))
            outputs.append(
                _output_receipt(
                    staged,
                    output_root / name,
                    f"{collection}_{partition}_sequence_indices",
                )
            )
            outputs[-1].update(
                {"dtype": "uint32_little_endian", "shape": [int(indices.size)]}
            )

    split_path = staging / "sequence_split.jsonl"
    split_payload = b"".join(
        canonical_json_bytes(asdict(record)) + b"\n"
        for record in sorted(all_records, key=lambda item: item.identity.encode("utf-8"))
    )
    _write_bytes_new(split_path, split_payload)
    outputs.append(
        _output_receipt(
            split_path,
            output_root / split_path.name,
            "complete_sequence_identity_split",
        )
    )

    manifest = {
        **preflight,
        "schema_version": "frequency-tiered-token-count-manifest-v1",
        "status": "complete",
        "input_assets": sorted(input_receipts, key=lambda item: item["role"]),
        "split": {
            "hash": "sha256",
            "modulus": SPLIT_MODULUS,
            "validation_remainder": VALIDATION_REMAINDER,
            "identity_encoding": "canonical compact JSON UTF-8",
            "counts": split_counts,
            "content_cross_partition_duplicates": 0,
        },
        "fit_frequency": {
            "positions": int(fit_frequency.sum()),
            "nonzero_token_ids": int(np.count_nonzero(fit_frequency)),
            "validation_sequences_used": False,
        },
        "outputs": sorted(outputs, key=lambda item: item["role"]),
        "actual_output_bytes_excluding_manifest": sum(
            int(item["bytes"]) for item in outputs
        ),
    }
    manifest_staged = staging / "token_count_manifest.json"
    _write_json_atomic(manifest_staged, manifest)
    os.replace(staging, output_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = build_token_counts(args.config, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
