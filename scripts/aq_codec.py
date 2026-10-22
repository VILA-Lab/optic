#!/usr/bin/env python3
"""Decode Additive Quantization embeddings by summing selected codewords."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


FORMAT_ID = "full_additive_quantization_ple_npz_v1"
FINAL_TRAINING_STATUSES = {"converged", "completed_fixed_schedule"}
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
RECON_KEY = "embed_tokens_per_layer.weight"
PLE_CHECKPOINT_SUFFIX = "embed_tokens_per_layer.weight"


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packed_stage_bytes(rows: int, bits: int) -> int:
    if rows <= 0 or not 1 <= bits <= 16:
        raise ValueError(f"invalid packed-code geometry: rows={rows}, bits={bits}")
    return math.ceil(rows * bits / 8)


def expected_resident_bytes(rows: int, d_ple: int, m_count: int, clusters: int, bits: int) -> int:
    """Persistent online bytes: packed codes plus FP16 full-d codebooks."""
    if d_ple <= 0 or m_count <= 0 or clusters <= 1:
        raise ValueError("invalid full-AQ geometry")
    if clusters > (1 << bits):
        raise ValueError("code width cannot address all clusters")
    return m_count * packed_stage_bytes(rows, bits) + m_count * clusters * d_ple * 2


def pack_nbit(values: Any, bits: int) -> Any:
    """Pack unsigned integer codes in little-endian bit order."""
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("codes must be one-dimensional")
    if not 1 <= bits <= 16:
        raise ValueError(f"unsupported packed-code width: {bits}")
    if array.size and (int(array.min()) < 0 or int(array.max()) >= (1 << bits)):
        raise ValueError("code is outside the configured bit width")
    shifts = np.arange(bits, dtype=np.uint32)
    bit_matrix = ((array.astype(np.uint32)[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little")


def unpack_nbit(values: Any, bits: int, count: int) -> Any:
    import numpy as np

    packed = np.asarray(values, dtype=np.uint8)
    if packed.ndim != 1 or packed.nbytes != packed_stage_bytes(count, bits):
        raise ValueError("packed-code byte count does not match geometry")
    unpacked = np.unpackbits(packed, bitorder="little", count=count * bits).reshape(count, bits)
    weights = (1 << np.arange(bits, dtype=np.uint32))[None, :]
    return (unpacked.astype(np.uint32) * weights).sum(axis=1, dtype=np.uint32)


def _metadata_value(value: Any) -> dict[str, Any]:
    raw = value.item() if value.shape == () else value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("full-AQ metadata must be a JSON string")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("full-AQ metadata must decode to an object")
    return result


def read_full_aq_npz(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    artifact = Path(path)
    with np.load(artifact, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError(f"{artifact} does not contain metadata")
        metadata = _metadata_value(archive["metadata"])
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
    return metadata, arrays


def _validate(metadata: Mapping[str, Any], arrays: Mapping[str, Any], allow_inconclusive: bool) -> dict[str, int]:
    import numpy as np

    if metadata.get("compressed_ple_format_id") != FORMAT_ID:
        raise ValueError("artifact is not the full Additive Quantization PLE format")
    if metadata.get("method_family") != "full_additive_quantization":
        raise ValueError("artifact method family is not full Additive Quantization")
    status = metadata.get("training_status")
    if status not in FINAL_TRAINING_STATUSES and not allow_inconclusive:
        raise ValueError(
            f"full-AQ training status is {status!r}; decoding requires 'converged' "
            "or 'completed_fixed_schedule'"
        )
    shape = metadata.get("ple_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("ple_shape must be [vocab_size, layers*d_ple]")
    vocab, width = int(shape[0]), int(shape[1])
    layers = int(metadata.get("num_layers", 0))
    d_ple = int(metadata.get("d_ple", 0))
    m_count = int(metadata.get("M", 0))
    clusters = int(metadata.get("K", 0))
    bits = int(metadata.get("bits", 0))
    if min(vocab, layers, d_ple, m_count, clusters, bits) <= 0 or width != layers * d_ple:
        raise ValueError("full-AQ PLE geometry is inconsistent")
    if clusters > (1 << bits):
        raise ValueError("code width cannot address all codewords")
    rows = vocab * layers
    expected_keys = {"codebooks_fp16"} | {f"codes_m{stage:02d}" for stage in range(m_count)}
    if set(arrays) != expected_keys:
        raise ValueError(f"resident array set differs: {sorted(arrays)} != {sorted(expected_keys)}")
    codebooks = arrays["codebooks_fp16"]
    if codebooks.dtype != np.float16 or tuple(codebooks.shape) != (m_count, clusters, d_ple):
        raise ValueError("full-AQ codebook geometry/dtype differs")
    for stage in range(m_count):
        codes = arrays[f"codes_m{stage:02d}"]
        if codes.dtype != np.uint8 or codes.ndim != 1 or codes.nbytes != packed_stage_bytes(rows, bits):
            raise ValueError(f"codes_m{stage:02d} packed geometry/dtype differs")
    actual_bytes = sum(int(value.nbytes) for value in arrays.values())
    expected_bytes = expected_resident_bytes(rows, d_ple, m_count, clusters, bits)
    if actual_bytes != expected_bytes or int(metadata.get("storage_bytes", -1)) != expected_bytes:
        raise ValueError(f"resident byte contract differs: arrays={actual_bytes}, expected={expected_bytes}")
    return {
        "vocab": vocab,
        "layers": layers,
        "d_ple": d_ple,
        "M": m_count,
        "K": clusters,
        "bits": bits,
        "rows": rows,
        "resident_bytes": expected_bytes,
    }


def _decode_selected(torch: Any, packed: Any, row_ids: Any, bits: int) -> Any:
    """Decode selected values without adding persistent padding bytes."""
    bit_offsets = row_ids.to(dtype=torch.long) * bits
    byte_offsets = torch.div(bit_offsets, 8, rounding_mode="floor")
    shifts = bit_offsets.remainder(8)
    last = packed.numel() - 1

    def byte_at(delta: int) -> Any:
        offsets = byte_offsets + delta
        valid = offsets <= last
        safe = offsets.clamp(max=last)
        return torch.where(valid, packed.index_select(0, safe).to(torch.int64), 0)

    combined = byte_at(0) | (byte_at(1) << 8) | (byte_at(2) << 16)
    return ((combined >> shifts) & ((1 << bits) - 1)).to(torch.long)


def _make_module_class(torch: Any) -> type[Any]:
    class FullAQCompactPLE(torch.nn.Module):
        """Resident compact AQ table with requested-token row decoding."""

        def __init__(self, metadata: dict[str, Any], arrays: dict[str, Any], device: Any, *, allow_inconclusive: bool):
            super().__init__()
            geometry = _validate(metadata, arrays, allow_inconclusive)
            self.metadata = dict(metadata)
            self.format_id = FORMAT_ID
            self.vocab_size = geometry["vocab"]
            self.layers = geometry["layers"]
            self.d_ple = geometry["d_ple"]
            self.num_codebooks = geometry["M"]
            self.clusters = geometry["K"]
            self.bits = geometry["bits"]
            self._contract_resident_bytes = geometry["resident_bytes"]
            self.register_buffer("codebooks", torch.from_numpy(arrays["codebooks_fp16"]).to(device))
            for stage in range(self.num_codebooks):
                self.register_buffer(
                    f"codes_m{stage:02d}",
                    torch.from_numpy(arrays[f"codes_m{stage:02d}"]).to(device=device, dtype=torch.uint8),
                )
            self.register_buffer(
                "layer_offsets",
                torch.arange(self.layers, device=device, dtype=torch.long) * self.vocab_size,
                persistent=False,
            )

        @property
        def ple_shape(self) -> tuple[int, int]:
            return self.vocab_size, self.layers * self.d_ple

        def persistent_resident_bytes(self) -> int:
            actual = self.codebooks.numel() * self.codebooks.element_size()
            actual += sum(
                getattr(self, f"codes_m{stage:02d}").numel()
                * getattr(self, f"codes_m{stage:02d}").element_size()
                for stage in range(self.num_codebooks)
            )
            if actual != self._contract_resident_bytes:
                raise RuntimeError("loaded compact tensors violate the full-AQ resident-byte contract")
            return int(actual)

        def decode_flat_rows(self, row_ids: Any) -> Any:
            rows = row_ids.reshape(-1).to(device=self.codebooks.device, dtype=torch.long)
            if bool(((rows < 0) | (rows >= self.vocab_size * self.layers)).any()):
                raise IndexError("full-AQ row id is outside the PLE table")
            decoded = torch.zeros((rows.numel(), self.d_ple), device=self.codebooks.device, dtype=torch.float32)
            for stage in range(self.num_codebooks):
                codes = _decode_selected(torch, getattr(self, f"codes_m{stage:02d}"), rows, self.bits)
                decoded.add_(self.codebooks[stage].index_select(0, codes).to(torch.float32))
            return decoded.to(torch.bfloat16)

        def forward(self, token_ids: Any) -> Any:
            original_shape = token_ids.shape
            flat = token_ids.reshape(-1).to(device=self.codebooks.device, dtype=torch.long)
            if bool(((flat < 0) | (flat >= self.vocab_size)).any()):
                raise IndexError("token id is outside the PLE vocabulary")
            rows = (flat[:, None] + self.layer_offsets[None, :]).reshape(-1)
            decoded = self.decode_flat_rows(rows)
            return decoded.reshape(*original_shape, self.layers * self.d_ple)

    return FullAQCompactPLE


def load_full_aq_ple(
    path: str | Path,
    *,
    device: str | Any = "cpu",
    allow_inconclusive: bool = False,
) -> Any:
    import torch

    metadata, arrays = read_full_aq_npz(path)
    cls = _make_module_class(torch)
    return cls(metadata, arrays, torch.device(device), allow_inconclusive=allow_inconclusive)


def compact_resident_bytes(module: Any) -> int:
    if not hasattr(module, "persistent_resident_bytes"):
        raise TypeError("module does not expose the full-AQ byte contract")
    return int(module.persistent_resident_bytes())


def install_full_aq_ple(
    model: Any,
    artifact_path: str | Path,
    *,
    device: str | Any,
    allow_inconclusive: bool = False,
) -> dict[str, Any]:
    """Install the dedicated compact module without touching common adapters."""
    import torch

    raw = load_full_aq_ple(artifact_path, device=device, allow_inconclusive=allow_inconclusive)

    class ScaledFullAQPLE(torch.nn.Module):
        def __init__(self, raw_module: Any):
            super().__init__()
            self.raw_module = raw_module
            self.register_buffer(
                "embed_scale",
                torch.tensor(math.sqrt(raw_module.d_ple), device=device),
                persistent=False,
            )

        def forward(self, ids: Any) -> Any:
            rows = self.raw_module(ids)
            return rows * self.embed_scale.to(device=rows.device, dtype=rows.dtype)

        def persistent_resident_bytes(self) -> int:
            return self.raw_module.persistent_resident_bytes()

    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        model,
    ]
    language_model = next(
        (candidate for candidate in candidates if candidate is not None and hasattr(candidate, "embed_tokens_per_layer")),
        None,
    )
    if language_model is None:
        raise AttributeError("model does not expose a Gemma-4 embed_tokens_per_layer module")
    language_model.embed_tokens_per_layer = ScaledFullAQPLE(raw)
    return {
        "format_id": FORMAT_ID,
        "deployment_mode": "online_requested_token_row_decode",
        "persistent_resident_bytes": raw.persistent_resident_bytes(),
        "artifact_sha256": artifact_sha256(artifact_path),
        "ple_shape": list(raw.ple_shape),
        "scale": math.sqrt(raw.d_ple),
        "dense_ple_parameter_present": any(
            name.endswith(PLE_CHECKPOINT_SUFFIX) for name, _ in model.named_parameters()
        ),
    }


def audit_requested_rows(
    artifact_path: str | Path,
    reconstruction_path: str | Path,
    *,
    token_ids: list[int],
    device: str | Any = "cpu",
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    module = load_full_aq_ple(artifact_path, device=device)
    ids = torch.tensor(token_ids, device=module.codebooks.device, dtype=torch.long)
    compact = module(ids).cpu()
    with safe_open(str(reconstruction_path), framework="pt", device="cpu") as handle:
        dense = handle.get_tensor(RECON_KEY).index_select(0, ids.cpu())
    exact = bool(torch.equal(compact, dense))
    max_abs = float((compact.float() - dense.float()).abs().max().item()) if compact.numel() else 0.0
    if not exact:
        raise ValueError(f"requested-row compact decode differs from capability reconstruction; max_abs={max_abs}")
    return {
        "schema_version": "ple-compact-requested-row-roundtrip-v1",
        "status": "passed",
        "token_ids": [int(value) for value in token_ids],
        "exact_bfloat16": exact,
        "max_abs_error": max_abs,
        "max_abs_diff_after_bfloat16_materialization": max_abs,
        "persistent_resident_bytes": module.persistent_resident_bytes(),
        "artifact_sha256": artifact_sha256(artifact_path),
        "reconstruction_sha256": artifact_sha256(reconstruction_path),
    }


__all__ = [
    "FORMAT_ID",
    "PLE_KEY",
    "RECON_KEY",
    "artifact_sha256",
    "audit_requested_rows",
    "compact_resident_bytes",
    "expected_resident_bytes",
    "install_full_aq_ple",
    "load_full_aq_ple",
    "pack_nbit",
    "packed_stage_bytes",
    "read_full_aq_npz",
    "unpack_nbit",
]
