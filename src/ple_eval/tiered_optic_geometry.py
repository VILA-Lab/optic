"""Tensor shapes and storage sizes for frequency-tiered OPTIC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


VOCAB_SIZE = 262_144
LAYERS = 42
D_PLE = 256
GROUP_DIM = 4
CODEBOOK_SIZE = 256
HIGH_RATE_COUNT = 110_411
SPECIAL_TOKEN_IDS = (
    2, 1, 3, 0, 4, 258881, 256000, 255999, 258883, 101, 258882, 106,
    52, 49, 47, 51, 258880, 100, 105, 48, 46, 50, 98, 258884,
)
DENSE_BYTES = VOCAB_SIZE * LAYERS * D_PLE * 2


@dataclass(frozen=True)
class RateGeometry:
    """Integer group allocation and its closed-form resident-byte accounting."""

    target_ratio: int
    base_groups: int
    extra_groups: int
    high_rate_count: int = HIGH_RATE_COUNT
    n_layers: int = LAYERS
    d_ple: int = D_PLE

    def __post_init__(self) -> None:
        if self.target_ratio <= 0:
            raise ValueError("target_ratio must be positive")
        if self.n_layers <= 0 or self.d_ple <= 0:
            raise ValueError("PLE layer count and width must be positive")
        if self.extra_groups <= 0 or self.base_groups != 4 * self.extra_groups:
            raise ValueError("rate geometry requires positive base:extra groups of exactly 4:1")
        ordinary_count = VOCAB_SIZE - len(SPECIAL_TOKEN_IDS)
        if not 0 < self.high_rate_count <= ordinary_count:
            raise ValueError("high_rate_count is outside the ordinary-token population")
        if self.resident_bytes > self.maximum_resident_bytes:
            raise ValueError("rate geometry exceeds its resident-byte budget")

    @property
    def maximum_groups(self) -> int:
        return self.base_groups + self.extra_groups

    @property
    def latent_dim(self) -> int:
        return self.maximum_groups * GROUP_DIM

    @property
    def maximum_resident_bytes(self) -> int:
        return self.dense_bytes // self.target_ratio

    @property
    def dense_bytes(self) -> int:
        return VOCAB_SIZE * self.n_layers * self.d_ple * 2

    def resident_census(self) -> dict[str, int]:
        values = {
            "base_token_codes_uint8": VOCAB_SIZE * self.base_groups,
            "high_rate_token_ids_uint32": self.high_rate_count * 4,
            "high_rate_extra_codes_uint8": self.high_rate_count * self.extra_groups,
            "codebooks_fp16": self.maximum_groups * CODEBOOK_SIZE * GROUP_DIM * 2,
            "decoder_weight_int8": self.n_layers * self.d_ple * self.latent_dim,
            "decoder_scale_per_output_fp16": self.n_layers * self.d_ple * 2,
            "decoder_bias_fp16": self.n_layers * self.d_ple * 2,
            "special_rows_bf16_bits_uint16": (
                len(SPECIAL_TOKEN_IDS) * self.n_layers * self.d_ple * 2
            ),
        }
        values["total"] = sum(values.values())
        return values

    @property
    def resident_bytes(self) -> int:
        return self.resident_census()["total"]

    @property
    def compression_ratio(self) -> float:
        return self.dense_bytes / self.resident_bytes

    @property
    def raw_fit_resident_bytes(self) -> int:
        """Full-code uint8 fit artifact with an FP16 decoder and bias."""

        return (
            VOCAB_SIZE * self.maximum_groups
            + self.maximum_groups * CODEBOOK_SIZE * GROUP_DIM * 2
            + self.n_layers * self.d_ple * self.latent_dim * 2
            + self.n_layers * self.d_ple * 2
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "target_ratio": self.target_ratio,
            "base_group_count": self.base_groups,
            "extra_group_count": self.extra_groups,
            "maximum_group_count": self.maximum_groups,
            "maximum_latent_dim": self.latent_dim,
            "high_rate_ordinary_token_count": self.high_rate_count,
            "maximum_resident_bytes": self.maximum_resident_bytes,
            "expected_resident_bytes": self.resident_bytes,
            "actual_compression_ratio": self.compression_ratio,
            "raw_full_code_fit_resident_bytes": self.raw_fit_resident_bytes,
        }


def solve_rate_geometry(
    target_ratio: int,
    *,
    n_layers: int = LAYERS,
    d_ple: int = D_PLE,
    high_rate_count: int = HIGH_RATE_COUNT,
) -> RateGeometry:
    """Use the largest 4:1 group bundle that fits the requested ratio."""

    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    if n_layers <= 0 or d_ple <= 0:
        raise ValueError("PLE layer count and width must be positive")
    fixed_bytes = (
        high_rate_count * 4
        + n_layers * d_ple * 2
        + n_layers * d_ple * 2
        + len(SPECIAL_TOKEN_IDS) * n_layers * d_ple * 2
    )
    bytes_per_extra_group_bundle = (
        VOCAB_SIZE * 4
        + high_rate_count
        + 5 * CODEBOOK_SIZE * GROUP_DIM * 2
        + 5 * n_layers * d_ple * GROUP_DIM
    )
    dense_bytes = VOCAB_SIZE * n_layers * d_ple * 2
    budget = dense_bytes // target_ratio
    extra_groups = (budget - fixed_bytes) // bytes_per_extra_group_bundle
    if extra_groups <= 0:
        raise ValueError("target ratio leaves no positive 4:1 group allocation")
    geometry = RateGeometry(
        target_ratio=target_ratio,
        base_groups=4 * extra_groups,
        extra_groups=extra_groups,
        high_rate_count=high_rate_count,
        n_layers=n_layers,
        d_ple=d_ple,
    )
    next_geometry_bytes = fixed_bytes + (extra_groups + 1) * bytes_per_extra_group_bundle
    if next_geometry_bytes <= budget:
        raise AssertionError("byte solver did not select the largest fitting group bundle")
    return geometry


def geometry_from_mapping(value: Mapping[str, Any]) -> RateGeometry:
    """Load a declared geometry and prove it equals the deterministic solver."""

    target_ratio = int(value["target_ratio"])
    declared = RateGeometry(
        target_ratio=target_ratio,
        base_groups=int(value["base_group_count"]),
        extra_groups=int(value["extra_group_count"]),
        high_rate_count=int(value.get("high_rate_ordinary_token_count", HIGH_RATE_COUNT)),
        n_layers=int(value.get("num_layers", LAYERS)),
        d_ple=int(value.get("d_ple", D_PLE)),
    )
    solved = solve_rate_geometry(
        target_ratio,
        n_layers=declared.n_layers,
        d_ple=declared.d_ple,
        high_rate_count=declared.high_rate_count,
    )
    if declared != solved:
        raise ValueError(f"declared geometry differs from byte solver: {declared} != {solved}")
    optional_exact = {
        "maximum_group_count": declared.maximum_groups,
        "maximum_latent_dim": declared.latent_dim,
        "maximum_resident_bytes": declared.maximum_resident_bytes,
        "expected_resident_bytes": declared.resident_bytes,
        "raw_full_code_fit_resident_bytes": declared.raw_fit_resident_bytes,
    }
    drift = {
        key: value.get(key)
        for key, expected in optional_exact.items()
        if key in value and int(value[key]) != expected
    }
    if drift:
        raise ValueError(f"declared derived geometry differs: {drift}")
    return declared


__all__ = [
    "CODEBOOK_SIZE",
    "DENSE_BYTES",
    "D_PLE",
    "GROUP_DIM",
    "HIGH_RATE_COUNT",
    "LAYERS",
    "RateGeometry",
    "SPECIAL_TOKEN_IDS",
    "VOCAB_SIZE",
    "geometry_from_mapping",
    "solve_rate_geometry",
]
