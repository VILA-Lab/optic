from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "models/e4b_bf16.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_config(path: str | Path | None = None) -> Path:
    """Resolve the selected model configuration to one canonical file path."""
    candidate = DEFAULT_MODEL_CONFIG if path is None else Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"model config is not a file: {resolved}")
    return resolved


def load_model(path: str | Path | None = None) -> dict[str, Any]:
    return read_json(resolve_model_config(path))


def model_config_binding(path: str | Path | None = None) -> dict[str, str]:
    resolved = resolve_model_config(path)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
    }


def load_vlm() -> dict[str, Any]:
    return read_json(ROOT / "configs" / "evaluation/vision.json")


def load_external_dependencies() -> dict[str, Any]:
    return read_json(ROOT / "configs" / "evaluation/external_dependencies.json")


def load_method(name: str) -> dict[str, Any]:
    directory = Path(os.environ.get("PLE_EVAL_METHOD_CONFIG_DIR", ROOT / "configs" / "methods"))
    return read_json(directory / f"{name}.json")


def load_level(name: str) -> dict[str, Any]:
    return read_json(ROOT / "configs" / "levels" / f"{name.lower()}.json")


def load_benchmark(name: str) -> dict[str, Any]:
    return read_json(ROOT / "configs" / "benchmarks" / f"{name}.json")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
