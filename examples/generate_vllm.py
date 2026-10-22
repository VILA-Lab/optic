"""Generate text with vLLM and compact OPTIC per-layer embeddings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ple_eval.evaluate import resolve_artifact, resolve_model_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    snapshot = resolve_model_snapshot(args.model_dir)
    artifact = args.artifact.expanduser() if args.artifact else resolve_artifact()
    # The ple_eval vLLM plugin reads this variable in every vLLM process.
    os.environ["PLE_OPTIC_ARTIFACT"] = str(artifact)

    from vllm import LLM, SamplingParams

    llm = LLM(model=str(snapshot), max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    outputs = llm.chat([{"role": "user", "content": args.prompt}], sampling,
                       chat_template_kwargs={"enable_thinking": False})
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
