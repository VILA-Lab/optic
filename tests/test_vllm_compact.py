import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file


class VllmCompactTests(unittest.TestCase):
    def test_checkpoint_iterator_skips_dense_ple(self):
        from vllm.model_executor.model_loader import default_loader
        from vllm.model_executor.models import gemma4

        from ple_eval import vllm_compact

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(default_loader, "safetensors_weights_iterator"), \
                patch.object(gemma4, "VocabParallelEmbedding"):
            artifact = Path(directory) / "compressed_ple.npz"
            artifact.write_bytes(b"")
            checkpoint = Path(directory) / "model.safetensors"
            save_file({"model.language_model.embed_tokens_per_layer.weight": torch.zeros(2, 3),
                       "model.language_model.norm.weight": torch.ones(3)}, str(checkpoint))
            vllm_compact.install(artifact)
            names = [name for name, _ in default_loader.safetensors_weights_iterator([str(checkpoint)], False)]
        self.assertEqual(names, ["model.language_model.norm.weight"])


if __name__ == "__main__":
    unittest.main()
