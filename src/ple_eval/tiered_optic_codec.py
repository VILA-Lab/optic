"""Compressed frequency-tiered OPTIC format with configurable geometry."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .tiered_optic_geometry import (
    CODEBOOK_SIZE,
    GROUP_DIM,
    RateGeometry,
    SPECIAL_TOKEN_IDS,
    VOCAB_SIZE,
    geometry_from_mapping,
)


METHOD_NAME = (
    "frequency-adaptive variable-rate cross-layer latent vector quantization "
    "with int8 decoding and exact tokenizer-special rows"
)
FORMAT_ID = (
    "frequency_adaptive_variable_rate_cross_layer_latent_vector_quantization_"
    "with_int8_decoding_and_exact_tokenizer_special_rows_multiratio_npz"
)
SCHEMA_VERSION = (
    "frequency-adaptive-variable-rate-cross-layer-latent-vector-quantization-"
    "with-int8-decoding-and-exact-tokenizer-special-rows-multiratio-v1"
)


def artifact_schema(geometry: RateGeometry) -> dict[str, Any]:
    return {
        "fixed_metadata": {
            "method_name": METHOD_NAME,
            "seed": 0,
            "ple_shape": [VOCAB_SIZE, geometry.n_layers * geometry.d_ple],
            "vocab_size": VOCAB_SIZE,
            "num_layers": geometry.n_layers,
            "d_ple": geometry.d_ple,
            **geometry.as_dict(),
            "group_dim": GROUP_DIM,
            "codebook_size": CODEBOOK_SIZE,
            "code_bits": 8,
            "high_rate_selection": (
                "descending_fit_frequency_then_ascending_token_id"
            ),
            "high_rate_token_id_storage": "sorted_uint32",
            "low_rate_extra_group_code": 0,
            "embedding_scale": math.sqrt(geometry.d_ple),
            "embedding_scale_storage": "immutable_metadata_python_float",
            "decoder_quantization": "symmetric_int8_per_layer_output_channel",
            "decoder_scale_dtype": "float16",
            "special_token_ids": list(SPECIAL_TOKEN_IDS),
            "special_token_id_storage": "immutable_metadata_python_tuple_no_persistent_tensor",
            "special_row_count": len(SPECIAL_TOKEN_IDS),
            "special_row_storage_dtype": "bfloat16_bit_patterns_as_uint16",
            "special_row_replacement": "exact_checkpoint_row_before_embedding_scale",
            "dense_reference_bytes": geometry.dense_bytes,
        },
        "arrays": {
            "base_token_codes_uint8": {
                "dtype": "uint8",
                "shape": [VOCAB_SIZE, geometry.base_groups],
            },
            "high_rate_token_ids_uint32": {
                "dtype": "uint32",
                "shape": [geometry.high_rate_count],
            },
            "high_rate_extra_codes_uint8": {
                "dtype": "uint8",
                "shape": [geometry.high_rate_count, geometry.extra_groups],
            },
            "codebooks_fp16": {
                "dtype": "float16",
                "shape": [geometry.maximum_groups, CODEBOOK_SIZE, GROUP_DIM],
            },
            "decoder_weight_int8": {
                "dtype": "int8",
                "shape": [geometry.n_layers, geometry.d_ple, geometry.latent_dim],
            },
            "decoder_scale_per_output_fp16": {
                "dtype": "float16",
                "shape": [geometry.n_layers, geometry.d_ple],
            },
            "decoder_bias_fp16": {
                "dtype": "float16",
                "shape": [geometry.n_layers, geometry.d_ple],
            },
            "special_rows_bf16_bits_uint16": {
                "dtype": "uint16",
                "shape": [
                    len(SPECIAL_TOKEN_IDS),
                    geometry.n_layers * geometry.d_ple,
                ],
            },
        },
        "resident_byte_census": geometry.resident_census(),
    }


def geometry_from_metadata(metadata: Mapping[str, Any]) -> RateGeometry:
    return geometry_from_mapping(metadata)


def validate_artifact(
    metadata: Mapping[str, Any], arrays: Mapping[str, Any]
) -> tuple[RateGeometry, dict[str, int]]:
    import numpy as np

    geometry = geometry_from_metadata(metadata)
    schema = artifact_schema(geometry)
    expected = {
        "compressed_ple_format_id": FORMAT_ID,
        "schema_version": SCHEMA_VERSION,
        **schema["fixed_metadata"],
        "resident_byte_census": schema["resident_byte_census"],
    }
    drift = {
        key: metadata.get(key)
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if drift:
        raise ValueError(f"multiratio artifact metadata mismatch: {drift}")
    if set(arrays) != set(schema["arrays"]):
        raise ValueError("multiratio artifact array set differs")
    census: dict[str, int] = {}
    for name, spec in schema["arrays"].items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(spec["dtype"])
            or tuple(value.shape) != tuple(spec["shape"])
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} dtype/shape/contiguity differs")
        census[name] = int(value.nbytes)
    census["total"] = sum(census.values())
    if census != geometry.resident_census():
        raise ValueError("multiratio resident byte census differs")
    high_ids = arrays["high_rate_token_ids_uint32"]
    if not bool((high_ids[1:] > high_ids[:-1]).all()):
        raise ValueError("high-rate token IDs must be strictly increasing")
    if bool(np.isin(high_ids, np.asarray(SPECIAL_TOKEN_IDS, dtype=np.uint32)).any()):
        raise ValueError("high-rate token IDs include a tokenizer-special ID")
    for name in ("codebooks_fp16", "decoder_scale_per_output_fp16", "decoder_bias_fp16"):
        if not bool(np.isfinite(arrays[name]).all()):
            raise ValueError(f"{name} contains a non-finite value")
    if not bool((arrays["decoder_scale_per_output_fp16"] > 0).all()):
        raise ValueError("decoder scales must be strictly positive")
    return geometry, census


def make_codec_class(torch: Any) -> type[Any]:
    class CompactFrequencyAdaptiveVariableRateLatentVQ(torch.nn.Module):
        def __init__(self, metadata: dict[str, Any], arrays: dict[str, Any], device: Any):
            super().__init__()
            geometry, _ = validate_artifact(metadata, arrays)
            self.format_id = FORMAT_ID
            self.geometry = geometry
            self.vocab_size = VOCAB_SIZE
            self.layers = geometry.n_layers
            self.d_ple = geometry.d_ple
            self.latent_dim = geometry.latent_dim
            self.base_group_count = geometry.base_groups
            self.maximum_group_count = geometry.maximum_groups
            self.extra_group_count = geometry.extra_groups
            self.high_rate_count = geometry.high_rate_count
            self.group_dim = GROUP_DIM
            self.embedding_scale = float(metadata["embedding_scale"])
            self.special_token_ids = SPECIAL_TOKEN_IDS
            for name in artifact_schema(geometry)["arrays"]:
                self.register_buffer(name, torch.from_numpy(arrays[name]).to(device))

        @property
        def ple_shape(self) -> tuple[int, int]:
            return self.vocab_size, self.layers * self.d_ple

        def persistent_resident_bytes(self) -> int:
            return sum(
                int(value.numel() * value.element_size()) for value in self.buffers()
            )

        def forward(self, ids: Any) -> Any:
            original_shape = ids.shape
            flat_ids = ids.reshape(-1).to(
                device=self.base_token_codes_uint8.device, dtype=torch.long
            )
            if flat_ids.numel() >= self.vocab_size:
                raise ValueError("online decoder refuses full-table materialization")
            if flat_ids.numel() and bool(
                ((flat_ids < 0) | (flat_ids >= self.vocab_size)).any().item()
            ):
                raise IndexError("requested token ID is outside the vocabulary")
            base_codes = self.base_token_codes_uint8.index_select(0, flat_ids).to(torch.long)
            high_ids = self.high_rate_token_ids_uint32.to(torch.long)
            high_positions = torch.searchsorted(high_ids, flat_ids)
            bounded = high_positions.clamp_max(self.high_rate_count - 1)
            is_high = (high_positions < self.high_rate_count) & (
                high_ids.index_select(0, bounded) == flat_ids
            )
            extra_codes = torch.zeros(
                (flat_ids.numel(), self.extra_group_count),
                dtype=torch.long,
                device=flat_ids.device,
            )
            if bool(is_high.any().item()):
                extra_codes[is_high] = self.high_rate_extra_codes_uint8.index_select(
                    0, bounded[is_high]
                ).to(torch.long)
            code_ids = torch.cat((base_codes, extra_codes), dim=1)
            code_ids += (
                torch.arange(
                    self.maximum_group_count, device=flat_ids.device, dtype=torch.long
                )
                * CODEBOOK_SIZE
            )
            latent = torch.nn.functional.embedding(
                code_ids, self.codebooks_fp16.reshape(-1, GROUP_DIM)
            ).reshape(flat_ids.numel(), self.latent_dim).to(torch.float32)
            decoded = []
            for layer in range(self.layers):
                weight = self.decoder_weight_int8[layer].to(torch.float32)
                weight *= self.decoder_scale_per_output_fp16[layer].to(torch.float32)[:, None]
                decoded.append(
                    latent @ weight.T + self.decoder_bias_fp16[layer].to(torch.float32)
                )
            rows = torch.cat(decoded, dim=1).to(torch.bfloat16)
            exact = self.special_rows_bf16_bits_uint16.view(torch.bfloat16)
            for special_index, token_id in enumerate(self.special_token_ids):
                mask = flat_ids == token_id
                if bool(mask.any().item()):
                    rows[mask] = exact[special_index]
            return rows.reshape(*original_shape, self.layers * self.d_ple)

    return CompactFrequencyAdaptiveVariableRateLatentVQ


def verify_requested_row_roundtrip(
    module: Any,
    token_ids: Sequence[int] | Any,
    expected_rows: Any,
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    import torch

    ids = torch.as_tensor(token_ids, dtype=torch.long, device=next(module.buffers()).device)
    if ids.numel() == 0 or ids.numel() >= module.vocab_size:
        raise ValueError("round-trip requires a nonempty requested-row subset")
    with torch.inference_mode():
        actual = module(ids).cpu().to(torch.bfloat16)
    expected = torch.as_tensor(expected_rows).cpu().to(torch.bfloat16)
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError("round-trip shape differs")
    maximum = float((actual.float() - expected.float()).abs().max().item())
    if not math.isfinite(maximum) or maximum > atol:
        raise ValueError(f"requested-row max_abs_diff {maximum} exceeds {atol}")
    return {
        "status": "passed",
        "requested_rows": int(ids.numel()),
        "max_abs_diff_after_bfloat16_materialization": maximum,
        "tolerance": float(atol),
        "persistent_resident_bytes": int(module.persistent_resident_bytes()),
        "full_table_materialized": False,
    }


__all__ = [
    "FORMAT_ID",
    "METHOD_NAME",
    "SCHEMA_VERSION",
    "artifact_schema",
    "geometry_from_metadata",
    "make_codec_class",
    "validate_artifact",
    "verify_requested_row_roundtrip",
]
