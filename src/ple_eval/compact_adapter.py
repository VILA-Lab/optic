"""Online adapters for compact Gemma-4 Per-Layer Embedding artifacts.

The modules in this file keep the compressed arrays resident and decode only
the token rows requested by a forward call.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


PLE_CHECKPOINT_SUFFIX = "embed_tokens_per_layer.weight"


def _metadata_value(value: Any) -> dict[str, Any]:
    raw = value.item() if value.shape == () else value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("compressed PLE metadata must be a JSON string")
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("compressed PLE metadata must decode to an object")
    return metadata


def read_compressed_npz(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read compressed embeddings and metadata from an NPZ file."""
    import numpy as np

    artifact = Path(path)
    with np.load(artifact, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError(f"{artifact} does not contain metadata")
        metadata = _metadata_value(archive["metadata"])
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
    return metadata, arrays


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry(metadata: Mapping[str, Any], d_ple: int) -> tuple[int, int, int]:
    shape = metadata.get("ple_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("compressed PLE metadata must contain ple_shape=[V,L*d]")
    vocab, width = (int(shape[0]), int(shape[1]))
    if vocab <= 0 or width <= 0 or d_ple <= 0 or width % d_ple:
        raise ValueError(f"invalid compressed PLE geometry: shape={shape}, d_ple={d_ple}")
    return vocab, width // d_ple, d_ple


def _tensor_nbytes(value: Any) -> int:
    return int(value.numel() * value.element_size())


class _CompactPLEBase:
    """Mixin shared by compact modules (the concrete classes also inherit nn.Module)."""

    vocab_size: int
    layers: int
    d_ple: int
    format_id: str

    @property
    def ple_shape(self) -> tuple[int, int]:
        return self.vocab_size, self.layers * self.d_ple

    def persistent_resident_bytes(self) -> int:
        return compact_resident_bytes(self)


def _decode_little_endian_codes(torch: Any, packed: Any, row_ids: Any, bits: int) -> Any:
    """Decode values packed by baseline_residual_vq.pack_nbit at selected rows."""
    if bits <= 0 or bits > 16:
        raise ValueError(f"unsupported packed-code width: {bits}")
    bit_offsets = row_ids.to(torch.long) * bits
    byte_offsets = torch.div(bit_offsets, 8, rounding_mode="floor")
    shifts = bit_offsets.remainder(8)

    # Pad the three-byte read window transiently at the end of the packed array.
    guard = torch.zeros(2, device=packed.device, dtype=packed.dtype)
    packed_size = packed.numel()

    def byte_at(offsets: Any) -> Any:
        in_artifact = (offsets >= 0) & (offsets < packed_size)
        artifact_offsets = offsets.clamp(min=0, max=packed_size - 1)
        guard_offsets = (offsets - packed_size).clamp(min=0, max=1)
        return torch.where(in_artifact, packed[artifact_offsets], guard[guard_offsets]).to(torch.int64)

    b0 = byte_at(byte_offsets)
    b1 = byte_at(byte_offsets + 1)
    b2 = byte_at(byte_offsets + 2)
    combined = b0 | (b1 << 8) | (b2 << 16)
    return ((combined >> shifts) & ((1 << bits) - 1)).to(torch.long)


def _resident_packed(torch: Any, value: Any, device: Any) -> Any:
    """Load exactly the serialized packed bytes into persistent state."""
    return torch.from_numpy(value).to(device=device, dtype=torch.uint8)


def _matmul_fp32(torch: Any, left: Any, right_t: Any) -> Any:
    if left.device.type == "cuda":
        return torch.mm(left, right_t, out_dtype=torch.float32)
    return left.to(torch.float32) @ right_t.to(torch.float32)


def _make_classes(torch: Any) -> dict[str, type[Any]]:
    class CompactOptic(torch.nn.Module, _CompactPLEBase):
        def __init__(self, metadata: dict[str, Any], arrays: dict[str, Any], device: Any):
            super().__init__()
            self.format_id = "optic_npz"
            if metadata.get("latent_mode") != "discrete":
                raise ValueError("the online OPTIC adapter requires a discrete artifact")
            self.d_ple = int(metadata["d_ple"])
            self.vocab_size, self.layers, _ = _geometry(metadata, self.d_ple)
            self.latent_dim = int(metadata["latent_dim"])
            self.n_groups = int(metadata["n_groups"])
            self.codebook_size = int(metadata["codebook_size"])
            self.group_dim = int(metadata["group_dim"])
            self.code_bits = int(metadata["code_bits"])
            if self.latent_dim != self.n_groups * self.group_dim:
                raise ValueError("OPTIC latent geometry is inconsistent")
            if self.code_bits != 8 or self.codebook_size != 256:
                raise ValueError("the online OPTIC path requires 8-bit K=256 codes")
            codes = arrays["token_codes_packed_uint8"]
            codebooks = arrays["codebooks_fp16"]
            decoder_weight = arrays["decoder_weight_fp16"]
            decoder_bias = arrays["decoder_bias_fp16"]
            expected = {
                "codes": (self.vocab_size * self.n_groups,),
                "codebooks": (self.n_groups, self.codebook_size, self.group_dim),
                "decoder_weight": (self.layers, self.d_ple, self.latent_dim),
                "decoder_bias": (self.layers, self.d_ple),
            }
            actual = {
                "codes": tuple(codes.shape),
                "codebooks": tuple(codebooks.shape),
                "decoder_weight": tuple(decoder_weight.shape),
                "decoder_bias": tuple(decoder_bias.shape),
            }
            if actual != expected:
                raise ValueError(f"OPTIC resident geometry mismatch: {actual} != {expected}")
            self.register_buffer(
                "token_codes", torch.from_numpy(codes).reshape(self.vocab_size, self.n_groups).to(device)
            )
            self.register_buffer("codebooks", torch.from_numpy(codebooks).to(device))
            self.register_buffer(
                "decoder_weight",
                torch.from_numpy(decoder_weight).reshape(self.layers * self.d_ple, self.latent_dim).to(device),
            )
            self.register_buffer("decoder_bias", torch.from_numpy(decoder_bias).reshape(-1).to(device))
            self.register_buffer(
                "group_offsets",
                (torch.arange(self.n_groups, device=device, dtype=torch.long) * self.codebook_size).reshape(1, -1),
                persistent=False,
            )

        def forward(self, ids: Any) -> Any:
            original_shape = ids.shape
            flat_ids = ids.reshape(-1).to(torch.long)
            code_ids = self.token_codes.index_select(0, flat_ids).to(torch.long) + self.group_offsets
            latent = torch.nn.functional.embedding(
                code_ids, self.codebooks.reshape(-1, self.group_dim)
            ).reshape(flat_ids.numel(), self.latent_dim)
            # Stack the layer decoders for one matrix multiplication with FP32 accumulation.
            decoded = _matmul_fp32(torch, latent, self.decoder_weight.T)
            decoded.add_(self.decoder_bias.to(torch.float32))
            return decoded.to(torch.bfloat16).reshape(*original_shape, self.layers * self.d_ple)

    class CompactPackedPQ(torch.nn.Module, _CompactPLEBase):
        def __init__(self, metadata: dict[str, Any], arrays: dict[str, Any], device: Any):
            super().__init__()
            self.format_id = "packed_pq_npz"
            codebooks = arrays["codebooks"]
            self.subspaces = int(metadata["num_subspaces"])
            self.clusters = int(metadata["clusters"])
            self.bits = int(metadata["bits"])
            self.d_ple = int(codebooks.shape[0] * codebooks.shape[2])
            self.vocab_size, self.layers, _ = _geometry(metadata, self.d_ple)
            if tuple(codebooks.shape) != (self.subspaces, self.clusters, self.d_ple // self.subspaces):
                raise ValueError("packed-PQ codebook geometry is inconsistent")
            self.register_buffer("codebooks", torch.from_numpy(codebooks).to(device))
            rows = self.vocab_size * self.layers
            expected_bytes = math.ceil(rows * self.bits / 8)
            for subspace in range(self.subspaces):
                name = f"codes_m{subspace:02d}"
                if int(arrays[name].nbytes) != expected_bytes:
                    raise ValueError(f"{name} byte count does not match packed-PQ geometry")
                self.register_buffer(name, _resident_packed(torch, arrays[name], device))
            self.register_buffer(
                "layer_offsets",
                torch.arange(self.layers, device=device, dtype=torch.long) * self.vocab_size,
                persistent=False,
            )

        def _decode_rows(self, row_ids: Any) -> Any:
            parts = []
            for subspace in range(self.subspaces):
                codes = _decode_little_endian_codes(
                    torch, getattr(self, f"codes_m{subspace:02d}"), row_ids, self.bits
                )
                parts.append(self.codebooks[subspace].index_select(0, codes))
            return torch.cat(parts, dim=-1)

        def forward(self, ids: Any) -> Any:
            original_shape = ids.shape
            flat = ids.reshape(-1).to(torch.long)
            rows = (flat[:, None] + self.layer_offsets[None, :]).reshape(-1)
            decoded = self._decode_rows(rows)
            return decoded.reshape(*original_shape, self.layers * self.d_ple).to(torch.bfloat16)

    class CompactResidualAQ(torch.nn.Module, _CompactPLEBase):
        def __init__(self, metadata: dict[str, Any], arrays: dict[str, Any], device: Any):
            super().__init__()
            self.format_id = str(metadata.get("compressed_ple_format_id") or "")
            if self.format_id not in {"residual_vq_npz", "joint_refined_additive_vq_npz"}:
                raise ValueError(f"unsupported additive-VQ source format {self.format_id!r}")
            codebooks = arrays["codebooks"]
            self.num_codebooks = int(metadata["M"])
            self.clusters = int(metadata["K"])
            self.bits = int(metadata["bits"])
            self.d_ple = int(codebooks.shape[2])
            self.vocab_size, self.layers, _ = _geometry(metadata, self.d_ple)
            if tuple(codebooks.shape) != (self.num_codebooks, self.clusters, self.d_ple):
                raise ValueError("residual-AQ codebook geometry is inconsistent")
            self.register_buffer("codebooks", torch.from_numpy(codebooks).to(device))
            rows = self.vocab_size * self.layers
            expected_bytes = math.ceil(rows * self.bits / 8)
            for codebook in range(self.num_codebooks):
                name = f"codes_m{codebook:02d}"
                if int(arrays[name].nbytes) != expected_bytes:
                    raise ValueError(f"{name} byte count does not match residual-AQ geometry")
                self.register_buffer(name, _resident_packed(torch, arrays[name], device))
            self.uses_coordinate_weights = bool(metadata.get("uses_coordinate_weights"))
            if self.uses_coordinate_weights:
                weights = arrays["coord_weights_fp16"]
                if tuple(weights.shape) != (self.layers, self.d_ple):
                    raise ValueError("residual-AQ coordinate-weight geometry is inconsistent")
                self.register_buffer("coordinate_scales", torch.from_numpy(weights).to(device).float().sqrt())
            self.register_buffer(
                "layer_offsets",
                torch.arange(self.layers, device=device, dtype=torch.long) * self.vocab_size,
                persistent=False,
            )

        def forward(self, ids: Any) -> Any:
            original_shape = ids.shape
            flat = ids.reshape(-1).to(torch.long)
            rows = (flat[:, None] + self.layer_offsets[None, :]).reshape(-1)
            decoded = torch.zeros((rows.numel(), self.d_ple), device=ids.device, dtype=torch.float32)
            for codebook in range(self.num_codebooks):
                codes = _decode_little_endian_codes(
                    torch, getattr(self, f"codes_m{codebook:02d}"), rows, self.bits
                )
                decoded.add_(self.codebooks[codebook].index_select(0, codes).to(torch.float32))
            decoded = decoded.reshape(flat.numel(), self.layers, self.d_ple)
            if self.uses_coordinate_weights:
                decoded.div_(self.coordinate_scales)
            return decoded.to(torch.bfloat16).reshape(*original_shape, self.layers * self.d_ple)

    class CompactQuantizedLowRank(torch.nn.Module, _CompactPLEBase):
        def __init__(self, metadata: dict[str, Any], arrays: dict[str, Any], device: Any):
            super().__init__()
            self.format_id = "quant_lowrank_factor_npz"
            q_f = arrays["q_f_int8"]
            q_g = arrays["q_g_int8"]
            f_scale = arrays["f_scale_fp16"]
            g_scale = arrays["g_scale_fp16"]
            self.rank = int(metadata["rank"])
            self.d_ple = int(q_g.shape[2])
            self.vocab_size, self.layers, _ = _geometry(metadata, self.d_ple)
            expected = {
                "q_f": (self.vocab_size, self.rank),
                "q_g": (self.rank, self.layers, self.d_ple),
                "f_scale": (self.rank,),
                "g_scale": (self.rank, self.layers),
            }
            actual = {
                "q_f": tuple(q_f.shape), "q_g": tuple(q_g.shape),
                "f_scale": tuple(f_scale.shape), "g_scale": tuple(g_scale.shape),
            }
            if actual != expected:
                raise ValueError(f"quantized-low-rank geometry mismatch: {actual} != {expected}")
            self.register_buffer("q_f", torch.from_numpy(q_f).to(device))
            self.register_buffer("f_scale", torch.from_numpy(f_scale).to(device))
            self.register_buffer("q_g", torch.from_numpy(q_g).to(device))
            self.register_buffer("g_scale", torch.from_numpy(g_scale).to(device))

        def forward(self, ids: Any) -> Any:
            original_shape = ids.shape
            flat = ids.reshape(-1).to(torch.long)
            left = self.q_f.index_select(0, flat).to(torch.float32) * self.f_scale.to(torch.float32)
            right = self.q_g.to(torch.float32) * self.g_scale.to(torch.float32).unsqueeze(2)
            decoded = left @ right.reshape(self.rank, self.layers * self.d_ple)
            return decoded.to(torch.bfloat16).reshape(*original_shape, self.layers * self.d_ple)


    class CompactDoutWeightedPcaPQ(torch.nn.Module, _CompactPLEBase):
        def __init__(self, metadata: dict[str, Any], arrays: dict[str, Any], device: Any):
            super().__init__()
            self.format_id = str(metadata.get("compressed_ple_format_id") or "")
            if self.format_id not in {
                "dout_weighted_pca_rotated_pq_scaled_exceptions_npz",
                "parametric_opq_npz",
            }:
                raise ValueError(f"unsupported PCA-PQ source format {self.format_id!r}")
            self.d_ple = int(metadata.get("d_ple") or arrays["mean_fp16"].shape[0])
            self.vocab_size, self.layers, _ = _geometry(metadata, self.d_ple)
            self.subspaces = int(metadata.get("subspaces") or metadata.get("num_subspaces"))
            self.clusters = int(metadata["clusters"])
            self.bits = int(metadata.get("bits_per_code") or metadata.get("bits"))
            codebooks = arrays["codebooks_fp16"]
            if int(codebooks.shape[0]) != self.subspaces or int(codebooks.shape[1]) != self.clusters:
                raise ValueError("output-weighted PCA-PQ codebook geometry is inconsistent")
            self.register_buffer("codebooks", torch.from_numpy(codebooks).to(device))
            self.register_buffer("rotation", torch.from_numpy(arrays["rotation_fp16"]).to(device))
            self.register_buffer("mean", torch.from_numpy(arrays["mean_fp16"]).to(device))
            self.register_buffer("coord_scales", torch.from_numpy(arrays["coord_scales_fp16"]).to(device))
            self.register_buffer("exception_token_ids", torch.from_numpy(arrays["exception_token_ids_uint32"]).to(device))
            self.register_buffer("exception_residuals", torch.from_numpy(arrays["exception_scaled_residuals_fp16"]).to(device))
            rows = self.vocab_size * self.layers
            expected_bytes = math.ceil(rows * self.bits / 8)
            for subspace in range(self.subspaces):
                name = f"codes_m{subspace:02d}"
                if int(arrays[name].nbytes) != expected_bytes:
                    raise ValueError(f"{name} byte count does not match output-weighted PCA-PQ geometry")
                self.register_buffer(name, _resident_packed(torch, arrays[name], device))
            self.register_buffer(
                "layer_offsets", torch.arange(self.layers, device=device, dtype=torch.long) * self.vocab_size,
                persistent=False,
            )

        def _exceptions(self, token_ids: Any) -> Any:
            shape = (token_ids.numel(), self.layers, self.d_ple)
            if self.exception_token_ids.numel() == 0:
                return torch.zeros(shape, device=token_ids.device, dtype=torch.float32)
            exception_token_ids = self.exception_token_ids.to(torch.long)
            matches = token_ids[:, None] == exception_token_ids[None, :]
            has_match = matches.any(dim=1).reshape(-1, 1, 1)
            positions = matches.to(torch.int64).argmax(dim=1)
            selected = self.exception_residuals.index_select(0, positions).to(torch.float32)
            return torch.where(has_match, selected, torch.zeros_like(selected))

        def decode_layer(self, token_ids: Any, layer: int) -> Any:
            """Decode one full layer with the builder's GEMM geometry."""
            rows = token_ids.to(torch.long) + self.layer_offsets[layer]
            parts = []
            for subspace in range(self.subspaces):
                codes = _decode_little_endian_codes(
                    torch, getattr(self, f"codes_m{subspace:02d}"), rows, self.bits
                )
                parts.append(self.codebooks[subspace].index_select(0, codes).to(torch.float32))
            rotation_t = self.rotation.to(torch.float32).T.contiguous()
            decoded = torch.cat(parts, dim=1) @ rotation_t + self.mean.to(torch.float32)
            decoded.div_(self.coord_scales[layer].to(torch.float32))
            if self.exception_token_ids.numel():
                matches = token_ids[:, None] == self.exception_token_ids.to(torch.long)[None, :]
                has_match = matches.any(dim=1)
                if bool(has_match.any().item()):
                    positions = matches[has_match].to(torch.int64).argmax(dim=1)
                    decoded[has_match] += self.exception_residuals[positions, layer, :].to(torch.float32)
            return decoded.to(torch.bfloat16)

        def forward(self, ids: Any) -> Any:
            original_shape = ids.shape
            flat = ids.reshape(-1).to(torch.long)
            rows = (flat[:, None] + self.layer_offsets[None, :]).reshape(-1)
            parts = []
            for subspace in range(self.subspaces):
                codes = _decode_little_endian_codes(
                    torch, getattr(self, f"codes_m{subspace:02d}"), rows, self.bits
                )
                parts.append(self.codebooks[subspace].index_select(0, codes).to(torch.float32))
            rotation_t = self.rotation.to(torch.float32).T.contiguous()
            decoded = torch.cat(parts, dim=1) @ rotation_t + self.mean.to(torch.float32)
            decoded = decoded.reshape(flat.numel(), self.layers, self.d_ple) / self.coord_scales.to(torch.float32)
            decoded.add_(self._exceptions(flat))
            return decoded.to(torch.bfloat16).reshape(*original_shape, self.layers * self.d_ple)



    class CompactScaleCalibration(torch.nn.Module, _CompactPLEBase):
        def __init__(self, source: Any, metadata: dict[str, Any], arrays: dict[str, Any], device: Any):
            super().__init__()
            self.format_id = str(metadata["compressed_ple_format_id"])
            self.source = source
            self.vocab_size, self.layers, self.d_ple = source.vocab_size, source.layers, source.d_ple
            scale = arrays["layer_coordinate_scale_fp16"]
            if tuple(scale.shape) != (self.layers, self.d_ple):
                raise ValueError("scale-calibration geometry is inconsistent")
            self.register_buffer("layer_coordinate_scale", torch.from_numpy(scale).to(device).float())
            bias = arrays.get("layer_coordinate_bias_fp16")
            self.has_bias = bias is not None
            if bias is not None:
                if tuple(bias.shape) != (self.layers, self.d_ple):
                    raise ValueError("affine-calibration bias geometry is inconsistent")
                self.register_buffer("layer_coordinate_bias", torch.from_numpy(bias).to(device).float())

        def forward(self, ids: Any) -> Any:
            raw = self.source(ids).to(torch.float32).reshape(*ids.shape, self.layers, self.d_ple)
            calibrated = raw * self.layer_coordinate_scale
            if self.has_bias:
                calibrated = calibrated + self.layer_coordinate_bias
            return calibrated.to(torch.bfloat16).reshape(
                *ids.shape, self.layers * self.d_ple
            )

    return {
        "optic_npz": CompactOptic,
        "packed_pq_npz": CompactPackedPQ,
        "residual_vq_npz": CompactResidualAQ,
        "joint_refined_additive_vq_npz": CompactResidualAQ,
        "quant_lowrank_factor_npz": CompactQuantizedLowRank,
        "dout_weighted_pca_rotated_pq_scaled_exceptions_npz": CompactDoutWeightedPcaPQ,
        "parametric_opq_npz": CompactDoutWeightedPcaPQ,
        "scale_calibrated_dout_weighted_pca_rotated_pq_scaled_exceptions_npz": CompactScaleCalibration,
        "affine_calibrated_pca_rotated_pq_scaled_exceptions_npz": CompactScaleCalibration,
    }


def _resolve_nested_source(
    artifact: Path,
    metadata: Mapping[str, Any],
    source_overrides: Mapping[str, str | Path] | None,
) -> Path:
    format_id = str(metadata["compressed_ple_format_id"])
    if source_overrides and format_id in source_overrides:
        return Path(source_overrides[format_id]).resolve()
    value = metadata.get("source_compressed_ple_npz")
    if not isinstance(value, str) or not value:
        raise ValueError(f"nested artifact {artifact} does not declare source_compressed_ple_npz")
    declared = Path(value)
    candidates = [declared, artifact.parent / declared.name, artifact.parent.parent / "source" / declared.name]
    if not declared.is_absolute():
        candidates.insert(0, artifact.parent / declared)
    for candidate in candidates:
        if candidate.resolve() != artifact.resolve() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not resolve nested compact PLE source {value!r} for {artifact}")


def load_compact_ple(
    artifact_path: str | Path,
    *,
    device: str | Any = "cpu",
    source_overrides: Mapping[str, str | Path] | None = None,
) -> Any:
    """Build an online raw (unscaled) PLE decoder from a compressed NPZ."""
    import torch

    artifact = Path(artifact_path).resolve()
    metadata, arrays = read_compressed_npz(artifact)
    format_id = str(metadata.get("compressed_ple_format_id") or "")
    if format_id == "frequency_adaptive_variable_rate_cross_layer_latent_vector_quantization_with_int8_decoding_and_exact_tokenizer_special_rows_npz":
        from .tiered_optic_e4b_adapter import load_compact_ple as load_tiered
        return load_tiered(artifact, device=device, source_overrides=source_overrides)
    if format_id == "frequency_adaptive_variable_rate_cross_layer_latent_vector_quantization_with_int8_decoding_and_exact_tokenizer_special_rows_multiratio_npz":
        from .tiered_optic_adapter import load_compact_ple as load_tiered
        return load_tiered(artifact, device=device, source_overrides=source_overrides)
    if format_id == "full_additive_quantization_ple_npz_v1":
        import importlib.util
        codec_path = Path(__file__).resolve().parents[2] / "scripts/aq_codec.py"
        spec = importlib.util.spec_from_file_location("aq_codec", codec_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load AQ codec: {codec_path}")
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        return codec.load_full_aq_ple(artifact, device=device)
    classes = _make_classes(torch)
    if format_id not in classes:
        raise ValueError(f"unsupported compressed PLE format: {format_id!r}")
    device_value = torch.device(device)
    nested_formats = {
        "scale_calibrated_dout_weighted_pca_rotated_pq_scaled_exceptions_npz",
        "affine_calibrated_pca_rotated_pq_scaled_exceptions_npz",
    }
    if format_id in nested_formats:
        source_path = _resolve_nested_source(artifact, metadata, source_overrides)
        expected_sha = str(metadata.get("source_compressed_ple_sha256") or "")
        if expected_sha and artifact_sha256(source_path) != expected_sha:
            raise ValueError("nested compressed PLE source sha256 mismatch")
        source = load_compact_ple(source_path, device=device_value, source_overrides=source_overrides)
        module = classes[format_id](source, metadata, arrays, device_value)
    else:
        module = classes[format_id](metadata, arrays, device_value)
    module._artifact_array_bytes = sum(int(value.nbytes) for value in arrays.values())
    module.eval()
    module.artifact_path = str(artifact)
    module.artifact_metadata = metadata
    module.artifact_sha256 = artifact_sha256(artifact)
    return module


def compact_resident_bytes(module: Any) -> int:
    """Count persistent tensor storage in the loaded module tree.
    """
    non_persistent = getattr(module, "_non_persistent_buffers_set", set())
    own = sum(
        _tensor_nbytes(value)
        for name, value in module.named_buffers(recurse=False)
        if value is not None and name not in non_persistent
    )
    return own + sum(compact_resident_bytes(child) for child in module.children())


def find_language_model(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        model,
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "embed_tokens_per_layer"):
            return candidate
    raise AttributeError("model does not expose a Gemma-4 embed_tokens_per_layer module")


def install_compact_ple(
    model: Any,
    artifact_path: str | Path,
    *,
    device: str | Any,
    source_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Replace Gemma-4's dense scaled PLE embedding with an online decoder."""
    import torch

    raw = load_compact_ple(artifact_path, device=device, source_overrides=source_overrides)

    class ScaledCompactPLE(torch.nn.Module):
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

    language_model = find_language_model(model)
    language_model.embed_tokens_per_layer = ScaledCompactPLE(raw)
    return {
        "format_id": raw.format_id,
        "artifact_path": raw.artifact_path,
        "artifact_sha256": raw.artifact_sha256,
        "ple_shape": list(raw.ple_shape),
        "persistent_resident_bytes": compact_resident_bytes(raw),
        "dense_ple_parameter_present": any(
            name.endswith(PLE_CHECKPOINT_SUFFIX) for name, _ in model.named_parameters()
        ),
        "scale": math.sqrt(raw.d_ple),
    }
