#!/usr/bin/env python3
"""Run EvalScope after applying the benchmark adapter override."""

from protocol_overrides import apply_from_environment

apply_from_environment()

from evalscope.cli.cli import run_cmd  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_cmd())
