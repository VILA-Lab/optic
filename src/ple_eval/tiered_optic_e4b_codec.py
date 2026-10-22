"""Compressed frequency-tiered OPTIC format for BF16 E4B."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence


METHOD_NAME = (
    "frequency-adaptive variable-rate cross-layer latent vector quantization "
    "with int8 decoding and exact tokenizer-special rows"
)
FORMAT_ID = (
    "frequency_adaptive_variable_rate_cross_layer_latent_vector_quantization_"
    "with_int8_decoding_and_exact_tokenizer_special_rows_npz"
)
SCHEMA_VERSION = (
    "frequency-adaptive-variable-rate-cross-layer-latent-vector-quantization-"
    "with-int8-decoding-and-exact-tokenizer-special-rows-v1"
)
VOCAB_SIZE = 262_144
LAYERS = 42
D_PLE = 256
BASE_GROUPS = 160
MAX_GROUPS = 200
EXTRA_GROUPS = MAX_GROUPS - BASE_GROUPS
GROUP_DIM = 4
CODEBOOK_SIZE = 256
LATENT_DIM = MAX_GROUPS * GROUP_DIM
HIGH_RATE_COUNT = 110_411
DENSE_BYTES = VOCAB_SIZE * LAYERS * D_PLE * 2
STRICT_100X_MAX_BYTES = DENSE_BYTES // 100
SPECIAL_TOKEN_IDS = (
    2, 1, 3, 0, 4, 258881, 256000, 255999, 258883, 101, 258882, 106,
    52, 49, 47, 51, 258880, 100, 105, 48, 46, 50, 98, 258884,
)


def resident_census() -> dict[str, int]:
    values = {
        "base_token_codes_uint8": VOCAB_SIZE * BASE_GROUPS,
        "high_rate_token_ids_uint32": HIGH_RATE_COUNT * 4,
        "high_rate_extra_codes_uint8": HIGH_RATE_COUNT * EXTRA_GROUPS,
        "codebooks_fp16": MAX_GROUPS * CODEBOOK_SIZE * GROUP_DIM * 2,
        "decoder_weight_int8": LAYERS * D_PLE * LATENT_DIM,
        "decoder_scale_per_output_fp16": LAYERS * D_PLE * 2,
        "decoder_bias_fp16": LAYERS * D_PLE * 2,
        "special_rows_bf16_bits_uint16": len(SPECIAL_TOKEN_IDS) * LAYERS * D_PLE * 2,
    }
    values["total"] = sum(values.values())
    return values


EXPECTED_RESIDENT_BYTES = resident_census()["total"]


def artifact_schema() -> dict[str, Any]:
    return {
        "fixed_metadata": {
            "method_name": METHOD_NAME,
            "seed": 0,
            "ple_shape": [VOCAB_SIZE, LAYERS * D_PLE],
            "vocab_size": VOCAB_SIZE,
            "num_layers": LAYERS,
            "d_ple": D_PLE,
            "latent_mode": "frequency_adaptive_variable_rate_discrete_with_exact_tokenizer_special_rows",
            "base_group_count": BASE_GROUPS,
            "maximum_group_count": MAX_GROUPS,
            "extra_group_count": EXTRA_GROUPS,
            "group_dim": GROUP_DIM,
            "maximum_latent_dim": LATENT_DIM,
            "codebook_size": CODEBOOK_SIZE,
            "code_bits": 8,
            "high_rate_ordinary_token_count": HIGH_RATE_COUNT,
            "high_rate_selection": "descending_fit_frequency_then_ascending_token_id",
            "high_rate_token_id_storage": "sorted_uint32",
            "low_rate_extra_group_code": 0,
            "embedding_scale": math.sqrt(D_PLE),
            "embedding_scale_storage": "immutable_metadata_python_float",
            "decoder_quantization": "symmetric_int8_per_layer_output_channel",
            "decoder_scale_dtype": "float16",
            "special_token_ids": list(SPECIAL_TOKEN_IDS),
            "special_token_id_source": "tokenizer_all_special_ids",
            "special_token_id_storage": "immutable_metadata_python_tuple_no_persistent_tensor",
            "special_row_count": len(SPECIAL_TOKEN_IDS),
            "special_row_storage_dtype": "bfloat16_bit_patterns_as_uint16",
            "special_row_replacement": "exact_checkpoint_row_before_embedding_scale",
            "storage_bytes": EXPECTED_RESIDENT_BYTES,
            "dense_reference_bytes": DENSE_BYTES,
            "strict_100x_max_resident_bytes": STRICT_100X_MAX_BYTES,
        },
        "arrays": {
            "base_token_codes_uint8": {
                "dtype": "uint8", "shape": [VOCAB_SIZE, BASE_GROUPS],
            },
            "high_rate_token_ids_uint32": {
                "dtype": "uint32", "shape": [HIGH_RATE_COUNT],
            },
            "high_rate_extra_codes_uint8": {
                "dtype": "uint8", "shape": [HIGH_RATE_COUNT, EXTRA_GROUPS],
            },
            "codebooks_fp16": {
                "dtype": "float16", "shape": [MAX_GROUPS, CODEBOOK_SIZE, GROUP_DIM],
            },
            "decoder_weight_int8": {
                "dtype": "int8", "shape": [LAYERS, D_PLE, LATENT_DIM],
            },
            "decoder_scale_per_output_fp16": {
                "dtype": "float16", "shape": [LAYERS, D_PLE],
            },
            "decoder_bias_fp16": {"dtype": "float16", "shape": [LAYERS, D_PLE]},
            "special_rows_bf16_bits_uint16": {
                "dtype": "uint16", "shape": [len(SPECIAL_TOKEN_IDS), LAYERS * D_PLE],
            },
        },
        "resident_byte_census": resident_census(),
    }


def validate_artifact(metadata: Mapping[str, Any], arrays: Mapping[str, Any]) -> dict[str, int]:
    import numpy as np

    schema = artifact_schema()
    expected = {
        "compressed_ple_format_id": FORMAT_ID,
        "schema_version": SCHEMA_VERSION,
        **schema["fixed_metadata"],
        "resident_byte_census": schema["resident_byte_census"],
    }
    drift = {key: metadata.get(key) for key, value in expected.items() if metadata.get(key) != value}
    if drift:
        raise ValueError(f"frequency-adaptive artifact metadata mismatch: {drift}")
    if set(arrays) != set(schema["arrays"]):
        raise ValueError("frequency-adaptive artifact array set differs")
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
    if census != resident_census() or census["total"] > STRICT_100X_MAX_BYTES:
        raise ValueError("frequency-adaptive resident byte census differs")
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
    provenance = metadata.get("provenance_sha256")
    required = {
        "finalizer", "finalization_launch", "fit_artifact", "fit_receipt",
        "fit_frequency", "checkpoint_weights", "tokenizer_snapshot_identity",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != required:
        raise ValueError("frequency-adaptive provenance keys differ")
    for name in required:
        digest = str(provenance[name])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"provenance_sha256.{name} is not lowercase SHA-256")
    return census


def make_codec_class(torch: Any) -> type[Any]:
    class CompactFrequencyAdaptiveVariableRateLatentVQ(torch.nn.Module):
        def __init__(self, metadata: dict[str, Any], arrays: dict[str, Any], device: Any):
            super().__init__()
            validate_artifact(metadata, arrays)
            self.format_id = FORMAT_ID
            self.vocab_size = VOCAB_SIZE
            self.layers = LAYERS
            self.d_ple = D_PLE
            self.latent_dim = LATENT_DIM
            self.base_group_count = BASE_GROUPS
            self.maximum_group_count = MAX_GROUPS
            self.group_dim = GROUP_DIM
            self.embedding_scale = float(metadata["embedding_scale"])
            self.special_token_ids = SPECIAL_TOKEN_IDS
            for name in artifact_schema()["arrays"]:
                self.register_buffer(name, torch.from_numpy(arrays[name]).to(device))

        @property
        def ple_shape(self) -> tuple[int, int]:
            return self.vocab_size, self.layers * self.d_ple

        def persistent_resident_bytes(self) -> int:
            return sum(int(value.numel() * value.element_size()) for value in self.buffers())

        def forward(self, ids: Any) -> Any:
            original_shape = ids.shape
            flat_ids = ids.reshape(-1).to(device=self.base_token_codes_uint8.device, dtype=torch.long)
            if flat_ids.numel() >= self.vocab_size:
                raise ValueError("online decoder refuses full-table materialization")
            if flat_ids.numel() and bool(((flat_ids < 0) | (flat_ids >= self.vocab_size)).any().item()):
                raise IndexError("requested token ID is outside the vocabulary")
            base_codes = self.base_token_codes_uint8.index_select(0, flat_ids).to(torch.long)
            high_ids = self.high_rate_token_ids_uint32.to(torch.long)
            high_positions = torch.searchsorted(high_ids, flat_ids)
            bounded = high_positions.clamp_max(HIGH_RATE_COUNT - 1)
            is_high = (high_positions < HIGH_RATE_COUNT) & (high_ids.index_select(0, bounded) == flat_ids)
            extra_codes = torch.zeros(
                (flat_ids.numel(), EXTRA_GROUPS), dtype=torch.long, device=flat_ids.device,
            )
            if bool(is_high.any().item()):
                extra_codes[is_high] = self.high_rate_extra_codes_uint8.index_select(
                    0, bounded[is_high]
                ).to(torch.long)
            code_ids = torch.cat((base_codes, extra_codes), dim=1)
            code_ids += torch.arange(MAX_GROUPS, device=flat_ids.device, dtype=torch.long) * CODEBOOK_SIZE
            latent = torch.nn.functional.embedding(
                code_ids, self.codebooks_fp16.reshape(-1, GROUP_DIM)
            ).reshape(flat_ids.numel(), LATENT_DIM).to(torch.float32)
            decoded = []
            for layer in range(LAYERS):
                weight = self.decoder_weight_int8[layer].to(torch.float32)
                weight *= self.decoder_scale_per_output_fp16[layer].to(torch.float32)[:, None]
                decoded.append(latent @ weight.T + self.decoder_bias_fp16[layer].to(torch.float32))
            rows = torch.cat(decoded, dim=1).to(torch.bfloat16)
            exact = self.special_rows_bf16_bits_uint16.view(torch.bfloat16)
            for special_index, token_id in enumerate(self.special_token_ids):
                mask = flat_ids == token_id
                if bool(mask.any().item()):
                    rows[mask] = exact[special_index]
            return rows.reshape(*original_shape, LAYERS * D_PLE)

    return CompactFrequencyAdaptiveVariableRateLatentVQ


def verify_requested_row_roundtrip(
    module: Any, token_ids: Sequence[int] | Any, expected_rows: Any, *, atol: float = 0.0,
) -> dict[str, Any]:
    import torch

    ids = torch.as_tensor(token_ids, dtype=torch.long, device=next(module.buffers()).device)
    if ids.numel() == 0 or ids.numel() >= VOCAB_SIZE:
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
        "requested_token_ids_sha256": hashlib.sha256(
            ids.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "requested_special_token_rows": sum(
            int(value) in SPECIAL_TOKEN_IDS for value in ids.detach().cpu().tolist()
        ),
        "max_abs_diff_after_bfloat16_materialization": maximum,
        "tolerance": float(atol),
        "persistent_resident_bytes": int(module.persistent_resident_bytes()),
        "persistent_special_token_id_tensor_present": False,
        "full_table_materialized": False,
    }


__all__ = [
    "BASE_GROUPS", "CODEBOOK_SIZE", "DENSE_BYTES", "D_PLE", "EXPECTED_RESIDENT_BYTES",
    "EXTRA_GROUPS", "FORMAT_ID", "GROUP_DIM", "HIGH_RATE_COUNT", "LATENT_DIM", "LAYERS",
    "MAX_GROUPS", "METHOD_NAME", "SCHEMA_VERSION", "SPECIAL_TOKEN_IDS",
    "STRICT_100X_MAX_BYTES", "VOCAB_SIZE", "artifact_schema", "make_codec_class",
    "resident_census", "validate_artifact", "verify_requested_row_roundtrip",
]
