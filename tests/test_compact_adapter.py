import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ple_eval.compact_adapter import compact_resident_bytes, load_compact_ple
from ple_eval.artifact_audit import load_builder
from safetensors import safe_open
from safetensors.torch import save_file


class CompactOpticTests(unittest.TestCase):
    def test_selected_tokens_match_codebook_and_affine_decoding(self):
        generator = np.random.default_rng(17)
        metadata = {
            "compressed_ple_format_id": "optic_npz",
            "ple_shape": [7, 12],
            "method": "optic",
            "kind": "optic",
            "latent_mode": "discrete",
            "d_ple": 4,
            "latent_dim": 8,
            "n_groups": 2,
            "group_dim": 4,
            "codebook_size": 256,
            "code_bits": 8,
        }
        codes = generator.integers(256, size=(7, 2), dtype=np.uint8)
        codebooks = generator.standard_normal((2, 256, 4)).astype(np.float16)
        weights = generator.standard_normal((3, 4, 8)).astype(np.float16)
        bias = generator.standard_normal((3, 4)).astype(np.float16)
        arrays = {
            "token_codes_packed_uint8": codes.flatten(),
            "codebooks_fp16": codebooks,
            "decoder_weight_fp16": weights,
            "decoder_bias_fp16": bias,
        }
        token_ids = np.array([[6, 1], [1, 0]])
        selected = codes[token_ids]
        latent = np.concatenate(
            [codebooks[group, selected[..., group]] for group in range(2)], axis=-1
        )
        expected = latent.astype(np.float32) @ weights.reshape(12, 8).astype(np.float32).T
        expected += bias.reshape(12).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "optic.npz"
            resident_bytes = sum(array.nbytes for array in arrays.values())
            builder = load_builder(Path(__file__).resolve().parents[1] / "builders/methods/optic.py")
            all_latents = np.concatenate([codebooks[group, codes[:, group]] for group in range(2)], axis=-1)
            full = all_latents.astype(np.float32) @ weights.reshape(12, 8).astype(np.float32).T
            full += bias.reshape(12).astype(np.float32)
            reconstruction = Path(directory) / "reconstruction.safetensors"
            save_file({"embed_tokens_per_layer.weight": torch.from_numpy(full).to(torch.bfloat16)}, reconstruction)
            np.savez(artifact, metadata=np.asarray(json.dumps(metadata)), **arrays)
            decoder = load_compact_ple(artifact)
            actual = decoder(torch.from_numpy(token_ids))
            torch.testing.assert_close(actual, torch.from_numpy(expected).to(torch.bfloat16), rtol=0, atol=0)
            self.assertEqual(compact_resident_bytes(decoder), resident_bytes)
            self.assertEqual(tuple(actual.shape), (2, 2, 12))
            difference = builder.verify_resident_roundtrip(
                torch, np, safe_open, artifact, reconstruction, 7, 3, 4,
                "discrete", 8, 2, 256, resident_bytes, torch.device("cpu"), 4, 0.0,
            )
            self.assertEqual(difference, 0.0)
            metadata["compressed_ple_format_id"] = "unknown_npz"
            np.savez(artifact, metadata=np.asarray(json.dumps(metadata)), **arrays)
            with self.assertRaisesRegex(ValueError, "unsupported compressed PLE format"):
                load_compact_ple(artifact)
            with self.assertRaisesRegex(SystemExit, "compressed_ple_format_id"):
                builder.load_resident_payload(np, artifact, 7, 3, 4, "discrete", 8, 2, 256, resident_bytes)


if __name__ == "__main__":
    unittest.main()
