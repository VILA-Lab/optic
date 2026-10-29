#!/usr/bin/env python3
"""Generate text from an OPTIC GGUF using the CPU-only llama.cpp build."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bin", type=Path, required=True,
                        help="path to the OPTIC-patched build-cpu/bin/llama")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpus", help="optional Linux taskset CPU list, e.g. 36-71")
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    parser.add_argument("--n-predict", type=int, default=64)
    args = parser.parse_args()
    if args.threads < 1 or args.n_predict < 1:
        parser.error("threads and n-predict must be positive")
    command = [
        str(args.llama_bin.resolve()), "cli", "-m", str(args.model.resolve()),
        "-ngl", "0", "-t", str(args.threads), "-tb", str(args.threads),
        "--load-mode", "auto", "--lazy-mode", "on",
        "-c", "512", "-b", "512", "-ub", "512", "-n", str(args.n_predict),
        "--cache-type-k", "f16", "--cache-type-v", "f16", "--flash-attn", "on",
        "--single-turn", "--simple-io", "--reasoning", "off",
        "--temp", "0", "--seed", "0", "--warmup", "-p", args.prompt,
    ]
    if args.cpus:
        command = ["taskset", "-c", args.cpus, *command]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_DYNAMIC="FALSE")
    raise SystemExit(subprocess.run(command, env=environment).returncode)


if __name__ == "__main__":
    main()
