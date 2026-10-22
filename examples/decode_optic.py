"""Decode a small synthetic OPTIC embedding table on CPU."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from ple_eval.compact_adapter import compact_resident_bytes, load_compact_ple


def main() -> None:
    generator = np.random.default_rng(0)
    vocabulary, layers, embedding_width = 32, 3, 16
    groups, group_width, codebook_size = 2, 4, 256
    latent_width = groups * group_width
    metadata = {
        "compressed_ple_format_id": "optic_npz",
        "ple_shape": [vocabulary, layers * embedding_width],
        "latent_mode": "discrete",
        "d_ple": embedding_width,
        "latent_dim": latent_width,
        "n_groups": groups,
        "group_dim": group_width,
        "codebook_size": codebook_size,
        "code_bits": 8,
    }
    arrays = {
        "token_codes_packed_uint8": generator.integers(
            codebook_size, size=vocabulary * groups, dtype=np.uint8
        ),
        "codebooks_fp16": generator.standard_normal(
            (groups, codebook_size, group_width)
        ).astype(np.float16),
        "decoder_weight_fp16": generator.standard_normal(
            (layers, embedding_width, latent_width)
        ).astype(np.float16),
        "decoder_bias_fp16": np.zeros((layers, embedding_width), dtype=np.float16),
    }
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "optic.npz"
        np.savez(artifact, metadata=np.asarray(json.dumps(metadata)), **arrays)
        decoder = load_compact_ple(artifact, device="cpu")
        token_ids = torch.tensor([3, 7, 3])
        embeddings = decoder(token_ids).reshape(-1, layers, embedding_width)
        print(f"Token IDs: {token_ids.tolist()}")
        print(f"Decoded shape [tokens, layers, width]: {list(embeddings.shape)}")
        print(f"Output dtype: {embeddings.dtype}")
        print(f"Compressed tensor storage: {compact_resident_bytes(decoder):,} bytes")


if __name__ == "__main__":
    main()
