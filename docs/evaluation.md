# Evaluate on the benchmarks

The released artifact is the seed-0 OPTIC fit for BF16 Gemma 4 E4B at 99.94× PLE
compression, whose result is the 97.8% entry of the
[results table](../README.md#paper-results). `ple-eval evaluate` runs the seven
benchmarks with this artifact and reports the task scores and retention.

## Install

Use Python 3.12 and an NVIDIA GPU. The evaluation uses vLLM 0.19.1,
Transformers 5.7.0, and EvalScope 1.8.1.

Run these commands from the repository root:

```bash
git clone https://github.com/VILA-Lab/optic.git
cd optic
python -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -r requirements-eval.txt
python -m pip install -e .
mkdir -p .external
git clone https://github.com/allenai/IFBench.git .external/IFBench
git -C .external/IFBench checkout db69a6f05689830b0068b8f1529ebcfd2f3b164c
python -m pip install -r .external/IFBench/requirements.txt
```

Request access to
[`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa), then authenticate:

```bash
hf auth login
```

## Prepare inputs

```bash
ple-eval evaluate --prepare-only --gpus 0 --run-id optic-e4b
```

This downloads the pinned checkpoint and the
[seed-0 artifact](https://huggingface.co/zcahjl3/gemma-4-E4B-it-OPTIC-100x),
checks their hashes, decodes the compressed PLE to a BF16 reconstruction for
capability evaluation, and checks the reconstruction against the compact decoder. The checkpoint revision is
`ee0ef6023621cff504d758262d4e04895a5af4a2`. The artifact location and SHA-256 are
recorded in [`configs/artifact.json`](../configs/artifact.json).

To use files already on disk:

```bash
ple-eval evaluate --prepare-only --run-id optic-e4b \
  --model-dir /path/to/gemma-4-E4B-it/snapshot \
  --artifact /path/to/compressed_ple.npz
```

`--model-dir` points to the snapshot containing `config.json`, tokenizer and
processor files, and `model.safetensors`. Run-specific paths are written under
`results/<run-id>/prepared/`.
The prepared reconstruction is reused by subsequent runs.

Dataset caches default to `.cache/`. Set `PLE_EVALSCOPE_DATASET_DIR`,
`MODELSCOPE_CACHE`, and `NLTK_DATA` to reuse existing caches.

## Run

Start with a small GPU check covering text generation, image input, and scoring:

```bash
ple-eval evaluate --smoke --gpus 0 --run-id optic-smoke
```

The check writes `results/optic-smoke/smoke.json`. Run the seven benchmarks:

```bash
ple-eval evaluate --gpus 0 --run-id optic-e4b
```

Supply the same `--model-dir` and `--artifact` options when using custom input
paths. With multiple GPUs, `--gpus 0,1` assigns independent tasks to the available
GPUs. Each task loads one model on one GPU.

The command runs PIQA, GSM8K, GPQA Diamond, AIME 2026, IFBench, InfographicVQA,
and MMLU-Redux. InfographicVQA uses the image processor and image-plus-text
input with the same compressed PLE. [Deployment and image input](deployment.md)
gives a direct Transformers example.

## Read the result

`results/optic-e4b/result.json` contains:

- `result.scores_pct`: the seven measured task scores.
- `result.retention_pct`: the geometric mean of the seven score/reference ratios,
  each capped at one, expressed as a percentage.
- `result.reference`: the matching uncompressed checkpoint scores from
  [`main-table.json`](../docs/figures/main-table.json), checkpoint revision, and
  benchmark protocol identifiers.
- `result.task_directories`: predictions, scoring reports, and logs for each task.

The prompts, generation settings, and scoring rules
are in [`configs/benchmarks/`](../configs/benchmarks).
