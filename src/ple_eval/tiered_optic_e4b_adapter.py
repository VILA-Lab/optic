"""Decode frequency-tiered OPTIC embeddings for BF16 E4B."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import compact_adapter
from .tiered_optic_e4b_codec import (
    FORMAT_ID,
    make_codec_class,
)


def load_compact_ple(
    artifact_path: str | Path,
    *,
    device: str | Any = "cpu",
    source_overrides: Mapping[str, str | Path] | None = None,
) -> Any:
    import torch

    if source_overrides:
        raise ValueError("this self-contained artifact does not accept source overrides")
    artifact = Path(artifact_path).resolve()
    metadata, arrays = compact_adapter.read_compressed_npz(artifact)
    if str(metadata.get("compressed_ple_format_id") or "") != FORMAT_ID:
        raise ValueError("artifact is not the frequency-adaptive variable-rate format")
    module = make_codec_class(torch)(metadata, arrays, torch.device(device))
    module._artifact_array_bytes = sum(int(value.nbytes) for value in arrays.values())
    module.eval()
    module.artifact_path = str(artifact)
    module.artifact_metadata = metadata
    module.artifact_sha256 = compact_adapter.artifact_sha256(artifact)
    return module


def compact_resident_bytes(module: Any) -> int:
    return compact_adapter.compact_resident_bytes(module)


def install_compact_ple(
    model: Any,
    artifact_path: str | Path,
    *,
    device: str | Any,
    source_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    import torch

    raw = load_compact_ple(artifact_path, device=device, source_overrides=source_overrides)

    class ScaledCompactPLE(torch.nn.Module):
        def __init__(self, raw_module: Any):
            super().__init__()
            self.raw_module = raw_module

        def forward(self, ids: Any) -> Any:
            return self.raw_module(ids) * self.raw_module.embedding_scale

    language_model = compact_adapter.find_language_model(model)
    language_model.embed_tokens_per_layer = ScaledCompactPLE(raw)
    return {
        "format_id": raw.format_id,
        "artifact_path": raw.artifact_path,
        "artifact_sha256": raw.artifact_sha256,
        "ple_shape": list(raw.ple_shape),
        "persistent_resident_bytes": compact_resident_bytes(raw),
        "dense_ple_parameter_present": any(
            name.endswith(compact_adapter.PLE_CHECKPOINT_SUFFIX) for name, _ in model.named_parameters()
        ),
        "scale": float(raw.embedding_scale),
        "base_group_count": raw.base_group_count,
        "maximum_group_count": raw.maximum_group_count,
        "high_rate_token_count": int(raw.high_rate_token_ids_uint32.numel()),
        "exact_special_row_count": len(raw.special_token_ids),
        "persistent_special_token_id_tensor_present": False,
        "adapter_version": (
            "frequency_adaptive_variable_rate_cross_layer_latent_vector_quantization_"
            "with_int8_decoding_and_exact_tokenizer_special_rows_adapter_v1"
        ),
    }


__all__ = ["compact_resident_bytes", "install_compact_ple", "load_compact_ple"]
