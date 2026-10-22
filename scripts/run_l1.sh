#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" -m ple_eval.cli run --level L1 "$@"

