# Fit OPTIC

This guide fits OPTIC on BF16 Gemma 4 E4B at 100× PLE compression with the
settings of the released artifact. Run the commands from the repository root.

## Install

Use Python 3.12 and an NVIDIA GPU:

```bash
python -m venv .venv-fit
source .venv-fit/bin/activate
python -m pip install -r requirements-fit.txt
python -m pip install -e .
```

## Set local paths

Set the local roots once, replacing the example paths:

```bash
python scripts/configure_paths.py \
  --cache-dir /path/to/cache \
  --model-dir /path/to/models \
  --data-dir /path/to/data \
  --scratch-dir /path/to/scratch
export PLE_SCRATCH_DIR=/path/to/scratch
export PYTHONPATH=src
```

This fills the path placeholders in the JSON configurations. The model is
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) at revision
`ee0ef6023621cff504d758262d4e04895a5af4a2`; check its weight path in
[`configs/fit/optic_e4b_bf16.json`](../configs/fit/optic_e4b_bf16.json) against the
downloaded files.

## Calibration

Calibration uses FineWeb, FineWeb-Edu, OpenWebMath, and FineWeb-2. Dataset
revisions, language subsets, and sampling settings are in
[`configs/calibration/`](../configs/calibration). Prepare the benchmark datasets at
the paths in `configs/calibration/broad.json`. The first command builds their
13-gram signatures, which remove benchmark text from the calibration data. The
next commands select calibration tokens and collect the BF16 E4B gates:

```bash
python scripts/build_eval_contamination_signatures.py --config configs/calibration/broad.json
python scripts/refresh_file_hashes.py
python scripts/build_coverage_optimized_calibration_tokens.py --config configs/calibration/coverage.json
python scripts/build_broad_gate_cache.py --config configs/calibration/coverage.json --device cuda:0
python scripts/refresh_file_hashes.py
```

`refresh_file_hashes.py` records the SHA-256 of the new files in the
configurations, and the next step checks them.

## Fit

```bash
python scripts/build_optic_e4b_bf16.py \
  --config configs/fit/optic_e4b_bf16.json \
  --input-mode natural_model_rows_special --device cuda:0 \
  --output-dir artifacts/optic-fit/e4b-bf16
```

The output directory contains `compressed_ple.npz`. To generate text with it:

```bash
python examples/generate.py --artifact artifacts/optic-fit/e4b-bf16/compressed_ple.npz
```

[Method settings](baselines.md) list the builders and settings of the other
methods in the results table.
