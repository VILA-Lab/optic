"""Serve Gemma 4 in vLLM with compact OPTIC per-layer embeddings.

The vLLM plugin calls ``install`` when ``PLE_OPTIC_ARTIFACT`` names an OPTIC
artifact. Gemma 4 then builds the compact decoder in place of the dense PLE
table, and the dense PLE tensor is skipped before it is read from the
checkpoint. Each forward pass decodes only the requested token rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

PLE_SUFFIX = "embed_tokens_per_layer.weight"


def install(artifact_path: str | Path) -> None:
    """Replace the dense Gemma 4 PLE in vLLM with the compact OPTIC decoder."""
    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader import default_loader
    from vllm.model_executor.model_loader.weight_utils import should_skip_weight
    from vllm.model_executor.models import gemma4

    from .compact_adapter import load_compact_ple

    artifact = Path(artifact_path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"OPTIC artifact is missing: {artifact}")
    dense_embedding = gemma4.VocabParallelEmbedding

    def embedding(num_embeddings: int, embedding_dim: int, *args: Any, **kwargs: Any) -> Any:
        if not str(kwargs.get("prefix", "")).endswith("embed_tokens_per_layer"):
            return dense_embedding(num_embeddings, embedding_dim, *args, **kwargs)
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("compact OPTIC PLE requires tensor parallel size 1")
        module = load_compact_ple(artifact, device=torch.empty(0).device)
        shape = (int(module.vocab_size), int(module.layers) * int(module.d_ple))
        if shape != (num_embeddings, embedding_dim):
            raise ValueError(f"OPTIC artifact PLE shape {shape} != model PLE shape "
                             f"{(num_embeddings, embedding_dim)}")
        return module

    def weights_without_dense_ple(
        hf_weights_files: list[str],
        use_tqdm_on_load: bool,
        safetensors_load_strategy: str | None = None,
        local_expert_ids: set[int] | None = None,
    ) -> Iterator[tuple[str, Any]]:
        for filename in hf_weights_files:
            with safe_open(filename, framework="pt") as handle:
                for name in handle.keys():
                    if should_skip_weight(name, local_expert_ids) or name.endswith(PLE_SUFFIX):
                        continue
                    yield name, handle.get_tensor(name)

    gemma4.VocabParallelEmbedding = embedding
    default_loader.safetensors_weights_iterator = weights_without_dense_ple
