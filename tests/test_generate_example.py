import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4VisionConfig

from ple_eval.compact_adapter import load_compact_ple


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples/generate.py"
spec = importlib.util.spec_from_file_location("optic_generate_example", EXAMPLE_PATH)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class GenerateExampleTests(unittest.TestCase):
    def test_streamed_compact_model_matches_text_and_image_without_reading_dense_ple(self):
        generator = np.random.default_rng(17)
        metadata = {
            "compressed_ple_format_id": "optic_npz", "ple_shape": [32, 8],
            "latent_mode": "discrete", "d_ple": 4, "latent_dim": 8,
            "n_groups": 2, "group_dim": 4, "codebook_size": 256, "code_bits": 8,
        }
        text = Gemma4TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8,
            hidden_size_per_layer_input=4, vocab_size_per_layer_input=32,
            layer_types=["sliding_attention", "full_attention"], sliding_window=16,
            max_position_embeddings=64, rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        )
        vision = Gemma4VisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            patch_size=2, pooling_kernel_size=2, position_embedding_size=16,
        )
        config = Gemma4Config(
            text_config=text, vision_config=vision, image_token_id=29,
            boi_token_id=30, eoi_token_id=31, audio_token_id=27, video_token_id=28,
        )
        config._attn_implementation = "sdpa"
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            artifact = snapshot / "optic.npz"
            np.savez(
                artifact, metadata=np.asarray(json.dumps(metadata)),
                token_codes_packed_uint8=generator.integers(256, size=64, dtype=np.uint8),
                codebooks_fp16=generator.standard_normal((2, 256, 4)).astype(np.float16),
                decoder_weight_fp16=generator.standard_normal((2, 4, 8)).astype(np.float16),
                decoder_bias_fp16=np.zeros((2, 4), dtype=np.float16),
            )
            dense = Gemma4ForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()
            decoder = load_compact_ple(artifact)
            with torch.no_grad():
                dense.model.language_model.embed_tokens_per_layer.weight.copy_(decoder(torch.arange(32)))
            dense.save_pretrained(snapshot)
            dense = Gemma4ForConditionalGeneration.from_pretrained(
                snapshot, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
            ).eval()
            read_keys = []
            original_open = example.safe_open

            class TrackedCheckpoint:
                def __init__(self, *args, **kwargs):
                    self.handle = original_open(*args, **kwargs)

                def __enter__(self):
                    self.handle.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.handle.__exit__(*args)

                def keys(self):
                    return self.handle.keys()

                def get_tensor(self, name):
                    read_keys.append(name)
                    return self.handle.get_tensor(name)

            with patch.object(example, "safe_open", TrackedCheckpoint):
                compact, receipt = example.load_model(snapshot, artifact, "cpu")
            self.assertNotIn(example.PLE_WEIGHT, read_keys)
            self.assertFalse(receipt["dense_ple_parameter_present"])
            self.assertIs(compact.lm_head.weight, compact.model.language_model.embed_tokens.weight)
            inputs = [
                {"input_ids": torch.tensor([[2, 3, 4]])},
                {"input_ids": torch.tensor([[2, 30, 29, 31, 4]]),
                 "pixel_values": torch.randn((1, 4, 12), dtype=torch.bfloat16),
                 "image_position_ids": torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])},
            ]
            with torch.inference_mode():
                for batch in inputs:
                    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
                    torch.testing.assert_close(compact(**batch).logits, dense(**batch).logits, rtol=0, atol=0)
                    generated = compact.generate(**batch, max_new_tokens=2, do_sample=False)
                    self.assertGreater(generated.shape[1], batch["input_ids"].shape[1])


if __name__ == "__main__":
    unittest.main()
