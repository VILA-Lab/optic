#!/usr/bin/env python3
"""Build a hash-only 13-gram set from the L1/L2 evaluation inputs."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/calibration/broad.json"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def ngram_hash(tokens: list[str]) -> int:
    payload = "\x1f".join(tokens).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def text_ngram_hashes(text: str, width: int) -> Iterable[int]:
    tokens = normalized_tokens(text)
    for start in range(max(0, len(tokens) - width + 1)):
        yield ngram_hash(tokens[start : start + width])


def recursive_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from recursive_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from recursive_strings(child)


def direct_file_strings(path: Path, kind: str) -> Iterable[str]:
    if kind == "text":
        yield path.read_text(encoding="utf-8")
        return
    if kind != "jsonl":
        raise ValueError(f"unsupported direct-file kind: {kind}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield from recursive_strings(json.loads(line))


def arrow_strings(path: Path, columns: list[str]) -> Iterable[str]:
    import pyarrow.ipc as ipc

    with path.open("rb") as handle:
        try:
            reader = ipc.open_stream(handle)
        except Exception:
            handle.seek(0)
            reader = ipc.open_file(handle)
        missing = sorted(set(columns) - set(reader.schema.names))
        if missing:
            raise RuntimeError(f"missing Arrow columns in {path}: {missing}")
        indices = [reader.schema.get_field_index(name) for name in columns]
        for batch in reader:
            for index in indices:
                for value in batch.column(index).to_pylist():
                    yield from recursive_strings(value)


def add_strings(
    values: Iterable[str], hashes: set[int], width: int, content_digest: Any
) -> tuple[int, int]:
    strings = 0
    windows = 0
    for value in values:
        encoded = value.encode("utf-8")
        content_digest.update(len(encoded).to_bytes(8, "little"))
        content_digest.update(encoded)
        strings += 1
        for value_hash in text_ngram_hashes(value, width):
            hashes.add(value_hash)
            windows += 1
    return strings, windows


def build_signatures(config: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    decontam = config["decontamination"]
    width = int(decontam["ngram_tokens"])
    hashes: set[int] = set()
    records: list[dict[str, Any]] = []
    for source in decontam["direct_files"]:
        path = Path(source["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        strings, windows = add_strings(
            direct_file_strings(path, source["kind"]), hashes, width, digest
        )
        records.append({
            "tag": source["tag"],
            "kind": source["kind"],
            "path": str(path),
            "file_sha256": sha256_file(path),
            "extracted_string_count": strings,
            "ngram_window_count_before_deduplication": windows,
            "extracted_text_stream_sha256": digest.hexdigest(),
        })
    for source in decontam["arrow_sources"]:
        paths = [Path(value) for value in sorted(glob.glob(source["glob"]))]
        if not paths:
            raise FileNotFoundError(source["glob"])
        digest = hashlib.sha256()
        total_strings = 0
        total_windows = 0
        for path in paths:
            strings, windows = add_strings(
                arrow_strings(path, source["columns"]), hashes, width, digest
            )
            total_strings += strings
            total_windows += windows
        records.append({
            "tag": source["tag"],
            "kind": "arrow_text_columns",
            "glob": source["glob"],
            "paths": [str(path) for path in paths],
            "columns": source["columns"],
            "extracted_string_count": total_strings,
            "ngram_window_count_before_deduplication": total_windows,
            "extracted_text_stream_sha256": digest.hexdigest(),
        })
    return sorted(hashes), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["storage"]["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = output_dir / "eval_13gram_hashes_le_u64.bin"
    manifest_path = output_dir / "eval_13gram_manifest.json"
    if signature_path.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite an existing contamination-signature asset")

    hashes, records = build_signatures(config)
    temporary = signature_path.with_suffix(signature_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for value in hashes:
            handle.write(value.to_bytes(8, "little"))
    temporary.replace(signature_path)
    manifest = {
        "schema_version": "ple-eval-ngram-signatures-v1",
        "protocol_id": config["protocol_id"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "normalization": config["decontamination"]["normalization"],
        "ngram_tokens": config["decontamination"]["ngram_tokens"],
        "unique_signature_count": len(hashes),
        "signature_file": str(signature_path.resolve()),
        "signature_file_sha256": sha256_file(signature_path),
        "sources": records,
        "contains_raw_evaluation_text": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
