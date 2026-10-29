# Gemma 4 E4B GGUF: CPU inference and benchmarking

The integration combines Google's official QAT Q4_0 backbone with the released
OPTIC 100× PLE. The converter preserves all 665 non-PLE tensor payloads and
replaces the PLE table with codes, codebooks and affine decoders. llama.cpp
reconstructs the requested token rows during inference.

## Quick CPU deployment

Requirements: Linux, Git, CMake, a C++17 compiler, curl and sha256sum. From this
repository, run:

```bash
bash examples/run_gguf_cpu.sh --threads 8 -p "Explain why the sky is blue."
```

The script builds the pinned, patched llama.cpp CPU runtime and downloads the
2.90 GB GGUF. Subsequent runs reuse both. `-n 128` changes the output limit;
`OPTIC_GGUF_CACHE` selects a cache directory. The matching runtime performs
online PLE decoding directly from the compressed codes.

To deploy directly from Hugging Face, install its CLI with
`python -m pip install huggingface_hub`, then run:

```bash
hf download zcahjl3/gemma-4-E4B-it-OPTIC-100x \
  --include "gguf/*" --local-dir optic-100x
bash optic-100x/gguf/run_cpu.sh --threads 8 -p "Explain why the sky is blue."
```

For manual builds or converting another copy of the official GGUF, use the
steps below.

## Build

Use a C++ compiler, CMake, Python 3.12 and Git. Run from the OPTIC repository:

```bash
export OPTIC_ROOT="$PWD"
export LLAMA_CPP="$OPTIC_ROOT/.external/llama.cpp"
mkdir -p .external
git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP"
git -C "$LLAMA_CPP" checkout f7c01c5226cac1de1052c17920ff38cdefea0ae6
git -C "$LLAMA_CPP" apply "$OPTIC_ROOT/patches/llama.cpp-optic.patch"

cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build-cpu" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
  -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build "$LLAMA_CPP/build-cpu" -j 8

python3.12 -m venv .venv-gguf
source .venv-gguf/bin/activate
python -m pip install numpy==2.2.6 huggingface_hub -e "$LLAMA_CPP/gguf-py"
```

The supplied patch adds OPTIC tensor loading and online PLE reconstruction.
The CPU build runs without CUDA. This pinned llama.cpp version uses the unified
`llama cli` and `llama serve` commands.

## Download and convert

```bash
hf download google/gemma-4-E4B-it-qat-q4_0-gguf \
  gemma-4-E4B_q4_0-it.gguf \
  --revision 4b4a2c1d584be7264f87aac328a1bc739ce81b6c \
  --local-dir models/official-gguf
hf download zcahjl3/gemma-4-E4B-it-OPTIC-100x compressed_ple.npz \
  --revision 41e64d1d89b1a4dff4aa9dd7f5dfa631aa4acac9 \
  --local-dir models/optic-ple

export BASE_GGUF="$OPTIC_ROOT/models/official-gguf/gemma-4-E4B_q4_0-it.gguf"
export OPTIC_GGUF="$OPTIC_ROOT/models/gemma-4-E4B-QAT-Q4_0-OPTIC-100x.gguf"
python scripts/convert_optic_gguf.py \
  --base "$BASE_GGUF" --artifact models/optic-ple/compressed_ple.npz \
  --output "$OPTIC_GGUF" \
  --artifact-revision 41e64d1d89b1a4dff4aa9dd7f5dfa631aa4acac9 \
  --expected-artifact-sha256 ce7073fe05e890a9c2fcb4e156567952cc5a3e2aaf846ad40425c668f450cacd
```

Each conversion also writes `OUTPUT.gguf.json` with input/output hashes,
PLE storage, compression ratios and the backbone byte-comparison result.
The artifact is the published BF16 E4B fit; this conversion reuses its parameters.

| Storage | Official GGUF | OPTIC GGUF |
| --- | ---: | ---: |
| Whole model, bytes | 5,154,941,280 | 2,899,237,408 |
| PLE payload, bytes | 2,312,110,080 (Q6_K) | 56,404,992 |

The compact PLE is 99.94× smaller than the original BF16 PLE and 40.99× smaller
than this official GGUF's Q6_K PLE. Codes use one byte each; codebooks, decoder
weights and biases use F16 storage. The affine operation accumulates in FP32,
rounds raw PLE to BF16, and applies the existing Gemma PLE scale once.

## CPU generation

```bash
python examples/generate_gguf_cpu.py \
  --llama-bin "$LLAMA_CPP/build-cpu/bin/llama" \
  --model "$OPTIC_GGUF" --threads 8 \
  --prompt "Explain why the sky is blue in two sentences."
```

The example sets CPU offload to zero, uses F16 KV, disables thinking and samples
greedily. `--cpus` optionally pins the process to a Linux CPU list; `--n-predict`
sets the generation limit. The model stays compressed and decodes PLE rows on demand.

## Matched CPU benchmark

The comparison uses an Intel Xeon Platinum 8352V, with 36 physical
cores on one socket (Linux CPU IDs 36–71). Both models use 36 decode and
36 prefill threads, context/batch/ubatch 512, F16 KV, flash attention enabled,
greedy seed 0, and the same 249-token templated prompt. Each process generates
64 tokens. Two rounds of four randomized blocks give eight measured runs per
model and loading mode, plus untimed warmups. Runs execute serially on a shared
host; the script retains system load observations.

The primary comparison enables PLE lazy loading for both models. Model files
have warm caches and were placed on the same NUMA node as inference. Process
wall time includes loading, initialization, the CLI warmup, generation and exit.
Generation speed follows the native slot convention: 64 output tokens contain
63 decode steps, since the first output token comes from prefill logits.

| Metric, median of 8 runs | Official GGUF | OPTIC GGUF |
| --- | ---: | ---: |
| Generation, tokens/s | 24.8087 | 24.4857 |
| Process wall time, seconds | 6.2077 | 6.2892 |

Use CPU IDs appropriate for your machine. On the measured host, the commands
below create separate file copies and warm them from the inference node.
NUMA snapshots are saved with each process's logs.

```bash
mkdir -p models/gguf-cpu-local
taskset -c 36 cp --reflink=auto "$BASE_GGUF" models/gguf-cpu-local/official.gguf
taskset -c 36 cp --reflink=auto "$OPTIC_GGUF" models/gguf-cpu-local/optic.gguf

taskset -c 36 python scripts/benchmark_gguf_cpu.py \
  --binary "$LLAMA_CPP/build-cpu/bin/llama" \
  --official models/gguf-cpu-local/official.gguf \
  --optic models/gguf-cpu-local/optic.gguf \
  --threads 36 --cpus 36-71 --blocks 4 --seed 202610291435 \
  --output results/gguf-cpu-round1
taskset -c 36 python scripts/benchmark_gguf_cpu.py \
  --binary "$LLAMA_CPP/build-cpu/bin/llama" \
  --official models/gguf-cpu-local/official.gguf \
  --optic models/gguf-cpu-local/optic.gguf \
  --threads 36 --cpus 36-71 --blocks 4 --seed 202610291452 \
  --output results/gguf-cpu-round2
```

The script measures both `--lazy-mode on` and the default `auto` setting,
always matching the setting between models. `results.json` contains each run,
medians and paired time ratios; stdout, stderr and NUMA snapshots are saved
alongside it. Actual output counts are checked against the requested 64 tokens.
The default prompt is [gguf-cpu.txt](../examples/prompts/gguf-cpu.txt);
`--prompt-file` selects another single-line prompt.

Measurements, model hashes and version information are in
[gguf-results.json](benchmarks/gguf-results.json). The measured deployment is
Google's QAT Q4_0 GGUF with its original Q6_K PLE, compared with the same
backbone and OPTIC PLE.

## CPU workload matrix

The native `llama bench` measurements cover prefill lengths 64/512/2048,
128-step decoding at KV depths 0/256/2048, 512-step decoding, and 4/8/16/36
threads. It runs synthetic token sequences through the full model without
sampling. Both models use the same mmap/lazy-on loading, F16 KV, flash attention,
batch/ubatch 512 and pinned physical cores. Each condition has four paired
process blocks with balanced model order and two native repeats per process.
The table reports medians of the four process medians for each model.

| Workload | Threads | Official tokens/s | OPTIC tokens/s | OPTIC change |
| --- | ---: | ---: | ---: | ---: |
| Prefill 64 | 36 | 257.29 | 255.47 | -0.71% |
| Prefill 512 | 36 | 266.20 | 263.35 | -1.07% |
| Prefill 2048 | 36 | 239.86 | 238.52 | -0.56% |
| Decode 128, KV depth 0 | 36 | 27.05 | 27.08 | +0.14% |
| Decode 128, KV depth 256 | 36 | 26.35 | 26.28 | -0.25% |
| Decode 128, KV depth 2048 | 36 | 25.11 | 24.71 | -1.56% |
| Decode 512, KV depth 256 | 36 | 26.23 | 26.15 | -0.30% |
| Decode 128, KV depth 256 | 4 | 10.75 | 10.63 | -1.12% |
| Decode 128, KV depth 256 | 8 | 16.95 | 16.50 | -2.65% |
| Decode 128, KV depth 256 | 16 | 23.33 | 22.87 | -1.95% |

The change column compares the two model medians. All four paired ratios,
individual samples, process RSS and observed CPU overlap are saved in
[gguf-cpu-matrix-results.json](benchmarks/gguf-cpu-matrix-results.json).
The measurements ran on a shared host with concurrent CPU, GPU and storage
activity. The selected physical cores and their SMT siblings were not exclusive.
Model file caches were warmed from the inference NUMA node. The native tool sets each condition's context capacity
from `prompt + generation + depth`; depth preparation occurs before decode timing.

Reproduce the matrix on the measured host with:

```bash
python scripts/benchmark_gguf_cpu_matrix.py \
  --binary "$LLAMA_CPP/build-cpu/bin/llama" \
  --official models/gguf-cpu-local/official.gguf \
  --optic models/gguf-cpu-local/optic.gguf \
  --plan docs/benchmarks/gguf-cpu-matrix-plan.json \
  --first-cpu 36 --monitor-cpu 0 --blocks 4 --repetitions 2 \
  --seed 202610291645 --output results/gguf-cpu-matrix
```

Use physical-core IDs and thread counts suitable for your machine in the plan.

## Longer real-prompt generation

The same 249-token library-planning prompt was also tested with 128 and 256
actual output tokens, alongside the 64-token run. Both models use
36 threads on CPU36–71, context 512, F16 KV, lazy-on, greedy sampling and warm
file caches. Native decode rate excludes the first output token, which comes
from prefill. Each row keeps its own original repetitions and host window.

| Output tokens | Measured runs per model | Official tokens/s | OPTIC tokens/s | Official process seconds | OPTIC process seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 8 | 24.81 | 24.49 | 6.208 | 6.289 |
| 128 | 4 | 24.59 | 24.73 | 8.848 | 8.861 |
| 256 | 4 | 25.34 | 25.24 | 13.758 | 13.867 |

The new 128/256-token runs use four interleaved process pairs per length plus
untimed warmups. Actual token counts, identical within-model responses,
individual native timings, process walls and RSS are retained in
[gguf-cpu-cli-lengths.json](benchmarks/gguf-cpu-cli-lengths.json).
Reproduce either length with the existing benchmark script:

```bash
taskset -c 36 python scripts/benchmark_gguf_cpu.py \
  --binary "$LLAMA_CPP/build-cpu/bin/llama" \
  --official models/gguf-cpu-local/official.gguf \
  --optic models/gguf-cpu-local/optic.gguf \
  --threads 36 --cpus 36-71 --blocks 4 --predict 256 --lazy-modes on \
  --seed 202610291714 --output results/gguf-cpu-cli256
```

For 128 output tokens, use `--predict 128 --seed 202610291712` and a new output
directory. These measurements ran on a shared host.

## GSM8K (provisional)

The paired evaluation covers all 1,319 GSM8K test items. It uses the project's
8-shot prompt, thinking disabled, greedy seed 0, a 10,240-token context per
slot, an 8,192-token generation limit and F16 KV. Each model ran on one
RTX 4090 with four request slots and `-ngl 99`.

| EvalScope 1.8.1 score | Official GGUF | OPTIC GGUF |
| --- | ---: | ---: |
| Original project scoring | 1227/1319 (93.03%) | 1228/1319 (93.10%) |
| Output-limit completions counted incorrect | 1227/1319 (93.03%) | 1227/1319 (93.03%) |

One OPTIC answer reaches the generation limit after computing the correct
number; the second row counts it as incorrect. The scores are recorded as
provisional results for this GGUF integration. The evaluation runner is
[evaluate_gguf_gsm8k.py](../scripts/evaluate_gguf_gsm8k.py); its `prepare`, `run`
and `compare` subcommands save prompts, raw answers and paired scores.

For a new run, install the [OPTIC evaluation dependencies](evaluation.md#install),
launch a server for each model, and reuse one prepared prompt file for both:

```bash
CUDA_VISIBLE_DEVICES=0 "$LLAMA_CPP/build-cuda/bin/llama" serve \
  -m "$BASE_GGUF" --host 127.0.0.1 --port 8091 --jinja \
  -ngl 99 --parallel 4 -c 40960 -b 2048 -ub 512 \
  --cache-type-k f16 --cache-type-v f16
```

Start the OPTIC server on a second available GPU and port 8092 with the same
options and `-m "$OPTIC_GGUF"`. Build `build-cuda` with `GGML_CUDA=ON` and
`CMAKE_CUDA_ARCHITECTURES=89` for RTX 4090. Then run:

```bash
python scripts/evaluate_gguf_gsm8k.py prepare \
  --source-dir data/request-sets --url http://127.0.0.1:8091 \
  --output results/gguf-gsm8k/prepared
python scripts/evaluate_gguf_gsm8k.py run \
  --prepared results/gguf-gsm8k/prepared --model-path "$BASE_GGUF" \
  --label official-qat-q4_0 --url http://127.0.0.1:8091 --concurrency 4 \
  --output results/gguf-gsm8k/official
python scripts/evaluate_gguf_gsm8k.py run \
  --prepared results/gguf-gsm8k/prepared --model-path "$OPTIC_GGUF" \
  --label optic-100x --url http://127.0.0.1:8092 --concurrency 4 \
  --output results/gguf-gsm8k/optic
python scripts/evaluate_gguf_gsm8k.py compare \
  --baseline results/gguf-gsm8k/official --candidate results/gguf-gsm8k/optic \
  --output results/gguf-gsm8k/comparison.json
```
