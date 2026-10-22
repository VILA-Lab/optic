"""Load a configured evaluation runtime."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

from .config import file_sha256, read_json


def validate_runtime_binding(binding: dict[str, Any]) -> dict[str, str]:
    if not isinstance(binding, dict) or binding.get("schema_version") != "ple-evaluation-runtime-binding-v1":
        raise ValueError("unsupported evaluation runtime binding")
    if set(binding) != {"schema_version", "backend", "entrypoint", "source_bindings", "configuration"}:
        raise ValueError("evaluation runtime binding fields differ")
    if not isinstance(binding["backend"], str) or not re.fullmatch(r"[a-z][a-z0-9_]+", binding["backend"]):
        raise ValueError("invalid evaluation backend name")
    if not isinstance(binding["configuration"], dict):
        raise ValueError("evaluation runtime configuration must be an object")
    rows = binding["source_bindings"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation runtime requires source bindings")
    sources: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("invalid runtime source binding")
        if not isinstance(row["path"], str) or not Path(row["path"]).is_absolute():
            raise ValueError("runtime source path must be absolute")
        path = Path(row["path"]).resolve(strict=True)
        digest = row["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid runtime source digest")
        if not path.is_file() or str(path) in sources or file_sha256(path) != digest:
            raise ValueError(f"runtime source is missing, duplicated, or changed: {path}")
        sources[str(path)] = digest
    entrypoint = binding["entrypoint"]
    if not isinstance(entrypoint, dict) or set(entrypoint) != {"path", "sha256"}:
        raise ValueError("invalid evaluation runtime entrypoint")
    path = Path(entrypoint["path"])
    if not path.is_absolute() or path.suffix != ".py":
        raise ValueError("runtime entrypoint must be an absolute Python source path")
    if sources.get(str(path.resolve(strict=True))) != entrypoint["sha256"]:
        raise ValueError("runtime entrypoint lacks its exact source binding")
    return sources


def load_bound_runtime(binding: dict[str, Any], **arguments: Any) -> Any:
    sources = validate_runtime_binding(binding)
    entrypoint = binding["entrypoint"]
    # Execute verified bytes, not a possibly stale .pyc or previously imported
    # module. Each load has its own module instance and source identity.
    path = Path(entrypoint["path"]).resolve(strict=True)
    source = path.read_bytes()
    import hashlib
    if hashlib.sha256(source).hexdigest() != entrypoint["sha256"]:
        raise ValueError("runtime entrypoint changed after validation")
    module_name = "ple_bound_evaluation_runtime_" + entrypoint["sha256"]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
        runtime = module.load_runtime(
            configuration=binding["configuration"], source_bindings=sources, **arguments
        )
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    if (getattr(runtime, "runtime_backend", None) != binding["backend"]
            or not callable(getattr(runtime, "generate", None))
            or not callable(getattr(runtime, "close", None))
            or (binding["configuration"].get("chat_request_contract") is not None
                and not callable(getattr(runtime, "complete_chat", None)))):
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
        raise ValueError("loaded runtime does not implement its declared interface")
    return runtime


def execution_backend(model_config: dict[str, Any], *, full_batch: bool = False) -> str:
    binding = model_config.get("evaluation_runtime")
    if binding is None:
        return "vllm_direct_full_batch" if full_batch else "vllm_direct"
    return binding["backend"]


def validate_runtime_method(binding: dict[str, Any], method: dict[str, Any]) -> None:
    """Reject a method label unless the selected runtime explicitly binds it."""
    selected = binding["configuration"].get("method_binding")
    if not isinstance(selected, dict) or set(selected) != {"path", "sha256"}:
        raise ValueError("The evaluation runtime is diagnostic-only without an exact method binding.")
    path = Path(selected["path"])
    if not path.is_absolute():
        raise ValueError("The evaluation method binding must use an absolute path.")
    resolved = str(path.resolve(strict=True))
    sources = {str(Path(row["path"]).resolve(strict=True)): row["sha256"]
               for row in binding["source_bindings"]}
    if (sources.get(resolved) != selected["sha256"]
            or file_sha256(path) != selected["sha256"] or read_json(path) != method):
        raise ValueError("The requested method differs from the runtime's exact method binding.")


def validate_runtime_interface(binding: dict[str, Any], interface: str) -> None:
    if interface == "direct":
        return
    if interface != "chat":
        raise ValueError("Unsupported evaluation runtime interface.")
    contract = binding["configuration"].get("chat_request_contract")
    if not isinstance(contract, dict) or set(contract) != {"path", "sha256"}:
        raise ValueError("The evaluation runtime has no exact chat request contract.")
    path = Path(contract["path"])
    sources = {str(Path(row["path"]).resolve(strict=True)): row["sha256"]
               for row in binding["source_bindings"]}
    if (not path.is_absolute() or sources.get(str(path.resolve(strict=True))) != contract["sha256"]
            or file_sha256(path) != contract["sha256"]):
        raise ValueError("The chat request contract is not exactly source-bound.")
