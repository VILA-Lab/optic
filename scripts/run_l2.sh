#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
PYTHON_BIN_DIR="$(dirname "$PYTHON")"
EVAL_ENV="$(dirname "$PYTHON_BIN_DIR")"
# vLLM imports CUDA before sqlite3; prefer this environment's C++ runtime so
# its ICU build cannot bind the older system libstdc++ first.
export LD_LIBRARY_PATH="${EVAL_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" -m ple_eval.cli run --level L2 "$@"
