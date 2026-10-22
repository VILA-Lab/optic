"""Generate text or answer an image question with compact OPTIC embeddings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration, GenerationConfig

from ple_eval.compact_adapter import install_compact_ple
from ple_eval.evaluate import resolve_artifact, resolve_model_snapshot


PLE_WEIGHT = "model.language_model.embed_tokens_per_layer.weight"


def load_model(snapshot: Path, artifact: Path, device: str):
    """Stream model weights to the device, skipping the dense PLE tensor."""
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config._attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=False):
        model = Gemma4ForConditionalGeneration(config)
    receipt = install_compact_ple(model, artifact, device=device)
    text_model = model.model.language_model
    shared_weights = {
        f"model.language_model.{name}.weight"
        for name in text_model._keys_to_ignore_on_load_unexpected
    }
    shapes = {name: value.shape for name, value in model.state_dict().items()}
    with safe_open(str(snapshot / "model.safetensors"), framework="pt", device=device) as checkpoint:
        for name in checkpoint.keys():
            if name == PLE_WEIGHT or name in shared_weights:
                continue
            tensor = checkpoint.get_tensor(name)
            if name not in shapes or tensor.shape != shapes[name]:
                raise ValueError(f"Checkpoint tensor does not match the model: {name}")
            module_name, leaf = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
            if leaf in module._parameters:
                module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
            else:
                module._buffers[leaf] = tensor
    model.lm_head.weight = text_model.embed_tokens.weight
    missing = [name for name, value in model.named_parameters() if value.is_meta]
    if missing:
        raise ValueError(f"Checkpoint is missing model weights: {missing}")
    generation_config = snapshot / "generation_config.json"
    if generation_config.is_file():
        model.generation_config = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
    return model.to(device).eval(), receipt


def prepare_inputs(processor, prompt: str, image: Path | None, device: str):
    content = []
    if image is not None:
        from PIL import Image

        with Image.open(image) as source:
            content.append({"type": "image", "image": source.convert("RGB")})
    content.append({"type": "text", "text": prompt})
    batch = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return batch.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    snapshot = resolve_model_snapshot(args.model_dir)
    artifact = args.artifact.expanduser() if args.artifact else resolve_artifact()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    model, receipt = load_model(snapshot, artifact, str(device))
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loaded = time.perf_counter()
    inputs = prepare_inputs(processor, args.prompt, args.image, str(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    finished = time.perf_counter()
    answer = generated[:, inputs["input_ids"].shape[1]:]
    print(processor.batch_decode(answer, skip_special_tokens=True)[0])
    stats = {
        "load_elapsed_seconds": round(loaded - started, 3),
        "generate_elapsed_seconds": round(finished - generation_started, 3),
        "generated_tokens": answer.shape[1],
        "ple_resident_bytes": receipt["persistent_resident_bytes"],
    }
    if device.type == "cuda":
        stats["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
