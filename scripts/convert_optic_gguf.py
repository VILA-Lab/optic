#!/usr/bin/env python3
"""Replace a Gemma 4 GGUF PLE table with online OPTIC tensors or a dense control.

Install numpy and llama.cpp's gguf-py (or put it on PYTHONPATH). Torch is needed
only for the dense-reconstructed control. All non-PLE tensor payloads are copied
without dequantization; the input can therefore use a quantized backbone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

try:
    import gguf
except ImportError as exc:
    raise SystemExit("Install llama.cpp/gguf-py or add it to PYTHONPATH") from exc

DENSE_NAME = "per_layer_token_embd.weight"
PREFIX = "per_layer_token_embd.optic_"
TENSOR_NAMES = {name: PREFIX + name for name in ("codes", "codebooks", "decoder", "bias")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifact(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("OPTIC NPZ is missing metadata")
        metadata = json.loads(str(archive["metadata"].item()))
        required = ("token_codes_packed_uint8", "codebooks_fp16", "decoder_weight_fp16", "decoder_bias_fp16")
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"OPTIC NPZ is missing arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in required}
    if metadata.get("latent_mode") != "discrete" or metadata.get("kind") != "optic":
        raise ValueError("expected a discrete OPTIC artifact")
    fields = ("d_ple", "latent_dim", "n_groups", "codebook_size", "group_dim", "code_bits")
    if any(not isinstance(metadata.get(key), int) or isinstance(metadata[key], bool) or metadata[key] <= 0 for key in fields):
        raise ValueError("OPTIC geometry fields must be positive integers")
    shape = metadata.get("ple_shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("expected ple_shape=[vocabulary, layers * d_ple]")
    vocab, width = shape
    d, latent, groups, k, group_dim, bits = (metadata[key] for key in fields)
    if width % d or latent != groups * group_dim or bits != 8 or k != 256:
        raise ValueError("inconsistent geometry or unsupported code format (requires 8-bit K=256)")
    layers = width // d
    for key, value in (("num_layers", layers), ("vocab_size", vocab)):
        if key in metadata and metadata[key] != value:
            raise ValueError(f"OPTIC {key} disagrees with ple_shape")
    expected = {
        "token_codes_packed_uint8": ((vocab * groups,), np.dtype("uint8")),
        "codebooks_fp16": ((groups, k, group_dim), np.dtype("float16")),
        "decoder_weight_fp16": ((layers, d, latent), np.dtype("float16")),
        "decoder_bias_fp16": ((layers, d), np.dtype("float16")),
    }
    for name, (array_shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != array_shape or value.dtype != dtype:
            raise ValueError(f"{name}: expected {array_shape} {dtype}, got {value.shape} {value.dtype}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    # I8 is a storage container: the custom ggml operation reads bytes unsigned.
    tensors = {
        TENSOR_NAMES["codes"]: arrays["token_codes_packed_uint8"].reshape(vocab, groups).view(np.int8),
        TENSOR_NAMES["codebooks"]: arrays["codebooks_fp16"],
        TENSOR_NAMES["decoder"]: arrays["decoder_weight_fp16"].reshape(width, latent),
        TENSOR_NAMES["bias"]: arrays["decoder_bias_fp16"].reshape(width),
    }
    return metadata, tensors


def validate_base(reader: Any, metadata: dict[str, Any]) -> Any:
    arch = reader.get_field("general.architecture")
    if arch is None or arch.contents() != "gemma4":
        raise ValueError("input must be a Gemma 4 GGUF")
    if reader.get_field("split.count") is not None and reader.get_field("split.count").contents() != 1:
        raise ValueError("merge split GGUFs with llama-gguf-split --merge before converting")
    if any(t.name.startswith(PREFIX) for t in reader.tensors) or reader.get_field("gemma4.optic.version"):
        raise ValueError("input already contains online OPTIC tensors")
    matching = [t for t in reader.tensors if t.name == DENSE_NAME]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one {DENSE_NAME}")
    dense = matching[0]
    if tuple(int(x) for x in dense.shape[::-1]) != tuple(metadata["ple_shape"]):
        raise ValueError("base PLE table shape disagrees with the artifact")
    for key, expected in (("gemma4.block_count", metadata["ple_shape"][1] // metadata["d_ple"]),
                          ("gemma4.embedding_length_per_layer_input", metadata["d_ple"])):
        field = reader.get_field(key)
        if field is not None and field.contents() != expected:
            raise ValueError(f"base {key} disagrees with artifact")
    return dense


def copy_metadata(reader: Any, writer: Any) -> None:
    # Constructor supplies architecture; replace its KV map so original key order,
    # explicit numeric types, string values and array element types are preserved.
    writer.kv_data[0].clear()
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        subtype = field.types[1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), field.types[0], sub_type=subtype)
    writer.data_alignment = int(reader.alignment)


def add_optic_metadata(writer: Any, metadata: dict[str, Any], tensors: dict[str, np.ndarray],
                       artifact_hash: str, artifact_revision: str, mode: str) -> None:
    writer.add_string("gemma4.optic.source_artifact_sha256", artifact_hash)
    if metadata.get("model_id"):
        writer.add_string("gemma4.optic.source_model_id", metadata["model_id"])
    if metadata.get("model_revision"):
        writer.add_string("gemma4.optic.source_model_revision", metadata["model_revision"])
    if artifact_revision:
        writer.add_string("gemma4.optic.source_artifact_revision", artifact_revision)
    writer.add_string("gemma4.optic.decode_precision", "fp32-affine-bf16-output")
    writer.add_string("gemma4.optic.float_tensor_storage", "f16")
    writer.add_string("gemma4.optic.embedding_scale", "runtime-sqrt-d-ple-once")
    if mode == "dense-reconstructed":
        writer.add_string("gemma4.optic.control", mode)
        return
    geometry = {
        "version": 1,
        "vocabulary_size": metadata["ple_shape"][0],
        "layer_count": metadata["ple_shape"][1] // metadata["d_ple"],
        "embedding_length": metadata["d_ple"],
        "latent_dimension": metadata["latent_dim"],
        "group_count": metadata["n_groups"],
        "codebook_size": metadata["codebook_size"],
        "codeword_dimension": metadata["group_dim"],
        "code_bits": metadata["code_bits"],
    }
    for key, value in geometry.items():
        writer.add_uint32(f"gemma4.optic.{key}", value)
    writer.add_uint64("gemma4.optic.payload_bytes", sum(t.nbytes for t in tensors.values()))


def reconstruct_dense(tensors: dict[str, np.ndarray], output: Path, chunk_tokens: int,
                      threads: int) -> np.memmap:
    import torch

    torch.set_num_threads(threads)
    codes = tensors[TENSOR_NAMES["codes"]].view(np.uint8)
    codebooks = torch.from_numpy(tensors[TENSOR_NAMES["codebooks"]]).float()
    decoder = torch.from_numpy(tensors[TENSOR_NAMES["decoder"]]).float()
    bias = torch.from_numpy(tensors[TENSOR_NAMES["bias"]]).float()
    vocab, groups = codes.shape
    width, latent_dim = decoder.shape
    dense = np.memmap(output, mode="w+", dtype=np.uint16, shape=(vocab, width))
    offsets = torch.arange(groups).reshape(1, -1) * 256
    with torch.inference_mode():
        for start in range(0, vocab, chunk_tokens):
            ids = torch.from_numpy(codes[start:start + chunk_tokens].astype(np.int64)) + offsets
            latent = torch.nn.functional.embedding(ids, codebooks.reshape(-1, codebooks.shape[-1])).reshape(-1, latent_dim)
            # Match CompactOptic on CPU: FP32 affine, one BF16 rounding, no sqrt.
            rows = (latent @ decoder.T + bias).to(torch.bfloat16)
            dense[start:start + len(ids)] = rows.view(torch.uint16).numpy()
    dense.flush()
    return dense


def verify_tensor_copy(base: Any, output: Any, mode: str, tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    originals = {t.name: t for t in base.tensors if t.name != DENSE_NAME}
    converted = {t.name: t for t in output.tensors}
    replacements = set(tensors) if mode == "compact" else {DENSE_NAME}
    if set(converted) != set(originals) | replacements:
        raise ValueError("converted tensor names do not match expected names")
    compared_bytes = 0
    for name, original in originals.items():
        actual = converted[name]
        if original.tensor_type != actual.tensor_type or not np.array_equal(original.shape, actual.shape):
            raise ValueError(f"non-PLE tensor type/shape changed: {name}")
        a, b = original.data.view(np.uint8).reshape(-1), actual.data.view(np.uint8).reshape(-1)
        if len(a) != len(b):
            raise ValueError(f"non-PLE tensor byte count changed: {name}")
        for start in range(0, len(a), 8 * 1024 * 1024):
            if not np.array_equal(a[start:start + 8 * 1024 * 1024], b[start:start + 8 * 1024 * 1024]):
                raise ValueError(f"non-PLE tensor payload changed: {name}")
        compared_bytes += len(a)
    for name, field in base.fields.items():
        if name.startswith("GGUF."):
            continue
        actual = output.get_field(name)
        if actual is None or actual.types != field.types or actual.contents() != field.contents():
            raise ValueError(f"base metadata changed: {name}")
    if mode == "compact":
        for name, value in tensors.items():
            actual = converted[name]
            if actual.data.shape != value.shape or not np.array_equal(actual.data.view(np.uint8).reshape(-1), value.view(np.uint8).reshape(-1)):
                raise ValueError(f"OPTIC tensor payload changed: {name}")
        if DENSE_NAME in converted:
            raise ValueError("compact output contains dense PLE")
    return {"non_ple_tensor_count": len(originals), "non_ple_bytes_compared": compared_bytes,
            "metadata_preserved": True, "non_ple_payloads_identical": True}



def paths_alias(first: Path, second: Path) -> bool:
    """Compare normalized paths and existing inodes, including hard links."""
    if first.resolve() == second.resolve():
        return True
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False


def validate_destination_paths(base: Path, artifact: Path, output: Path, *,
                               manifest: Path | None = None, force: bool = False) -> None:
    """Reject destination aliases before opening or creating any result files."""
    if paths_alias(output, base) or paths_alias(output, artifact):
        raise ValueError("output must differ from both inputs (including filesystem aliases)")
    if manifest is not None:
        for label, protected in (("base", base), ("artifact", artifact), ("output", output)):
            if paths_alias(manifest, protected):
                raise ValueError(f"manifest must differ from {label} (including filesystem aliases)")
    for path in (output, manifest):
        if path is None:
            continue
        if path.is_dir():
            raise IsADirectoryError(path)
        if path.exists() and not force:
            raise FileExistsError(path)


def write_manifest_atomic(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="optic-manifest-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def convert(base: Path, artifact: Path, output: Path, *, mode: str = "compact", artifact_revision: str = "",
            expected_artifact_sha256: str | None = None, chunk_tokens: int = 1024,
            threads: int = 8, force: bool = False) -> dict[str, Any]:
    if mode not in ("compact", "dense-reconstructed"):
        raise ValueError(f"unknown conversion mode: {mode}")
    if chunk_tokens <= 0 or threads <= 0:
        raise ValueError("chunk-tokens and threads must be positive")
    validate_destination_paths(base, artifact, output, force=force)
    start = time.monotonic()
    artifact_hash = sha256_file(artifact)
    if expected_artifact_sha256 and artifact_hash != expected_artifact_sha256:
        raise ValueError(f"artifact SHA-256 mismatch: {artifact_hash}")
    metadata, tensors = load_artifact(artifact)
    reader = gguf.GGUFReader(base)
    dense_original = validate_base(reader, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="optic-gguf-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        temporary_output = temporary / output.name
        writer = gguf.GGUFWriter(temporary_output, "gemma4", endianess=reader.endianess)
        copy_metadata(reader, writer)
        add_optic_metadata(writer, metadata, tensors, artifact_hash, artifact_revision, mode)
        for tensor in reader.tensors:
            if tensor.name != DENSE_NAME:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
        if mode == "compact":
            for name, value in tensors.items():
                writer.add_tensor(name, value)
            ple_bytes = sum(value.nbytes for value in tensors.values())
        else:
            print("Reconstructing dense BF16 OPTIC control in bounded chunks", file=sys.stderr, flush=True)
            dense = reconstruct_dense(tensors, temporary / "dense-ple.bf16", chunk_tokens, threads)
            # BF16's gguf-py representation is raw bytes; raw_dtype restores logical shape.
            writer.add_tensor(DENSE_NAME, dense.view(np.uint8).reshape(dense.shape[0], -1), raw_dtype=gguf.GGMLQuantizationType.BF16)
            ple_bytes = dense.nbytes
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=False)
        finally:
            writer.close()
        converted = gguf.GGUFReader(temporary_output)
        checks = verify_tensor_copy(reader, converted, mode, tensors)
        output_hash = sha256_file(temporary_output)
        os.replace(temporary_output, output)
    return {
        "schema_version": 1, "mode": mode, "base": str(base.resolve()),
        "base_sha256": sha256_file(base), "artifact": str(artifact.resolve()),
        "artifact_sha256": artifact_hash, "artifact_revision": artifact_revision,
        "artifact_metadata": metadata, "output": str(output.resolve()), "output_sha256": output_hash,
        "output_bytes": output.stat().st_size, "source_ple_bytes": int(dense_original.n_bytes),
        "gguf_ple_payload_bytes": int(ple_bytes), "compact_payload_bytes": sum(t.nbytes for t in tensors.values()),
        "fp32_cached_ple_payload_bytes": int(tensors[TENSOR_NAMES["codes"]].nbytes + 2 * sum(t.nbytes for n, t in tensors.items() if n != TENSOR_NAMES["codes"])),
        "compression_vs_source_ple": int(dense_original.n_bytes) / int(ple_bytes),
        "compression_vs_bf16_ple": 2 * int(np.prod(metadata["ple_shape"])) / int(ple_bytes),
        "source_ple_ggml_type": dense_original.tensor_type.name,
        "codes_storage": "I8 container; interpreted as uint8, no index expansion",
        "decode": "F16 file tensors; runtime FP32 affine -> BF16; sqrt(d_ple) applied once at runtime",
        "dense_control_threads": threads if mode == "dense-reconstructed" else None,
        "chunk_tokens": chunk_tokens if mode == "dense-reconstructed" else None,
        "elapsed_seconds": time.monotonic() - start, "verification": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="unmodified single-file Gemma 4 GGUF")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("compact", "dense-reconstructed"), default="compact")
    parser.add_argument("--artifact-revision", default="")
    parser.add_argument("--expected-artifact-sha256")
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for dense reconstruction only")
    parser.add_argument("--manifest", type=Path, help="default: OUTPUT.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest or Path(str(args.output) + ".json")
    validate_destination_paths(args.base, args.artifact, args.output, manifest=manifest_path, force=args.force)
    result = convert(args.base, args.artifact, args.output, mode=args.mode, artifact_revision=args.artifact_revision,
                     expected_artifact_sha256=args.expected_artifact_sha256, chunk_tokens=args.chunk_tokens,
                     threads=args.threads, force=args.force)
    result["command"] = [sys.executable, *sys.argv]
    write_manifest_atomic(manifest_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
