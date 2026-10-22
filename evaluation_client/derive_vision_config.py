#!/usr/bin/env python3
"""Derive the Gemma 4 1120-soft-token service config from the pinned model config."""

import json
import pathlib
import sys


def main() -> None:
    source = pathlib.Path(sys.argv[1])
    destination = pathlib.Path(sys.argv[2])
    config = json.loads(source.read_text())
    config["vision_config"]["default_output_length"] = 1120
    config["vision_soft_tokens_per_image"] = 1120
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
