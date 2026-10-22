"""Set local model, data, cache, and scratch paths in the configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(root: Path, paths: dict[str, Path]) -> list[str]:
    replacements = {
        f"<{name.upper()}>": str(path.expanduser().resolve())
        for name, path in paths.items()
    }

    def replace(value):
        if isinstance(value, str):
            for source, destination in replacements.items():
                value = value.replace(source, destination)
            return value
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {replace(key): replace(item) for key, item in value.items()}
        return value

    changed = []
    for directory in (root / "configs", root / "data" / "calibration"):
        for path in sorted(directory.rglob("*.json")):
            original = json.loads(path.read_text())
            updated = replace(original)
            if updated != original:
                path.write_text(json.dumps(updated, indent=2) + "\n")
                changed.append(str(path.relative_to(root)))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "model", "data", "scratch"):
        parser.add_argument(f"--{name}-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps({"updated": configure(ROOT, vars(args))}, indent=2))


if __name__ == "__main__":
    main()
