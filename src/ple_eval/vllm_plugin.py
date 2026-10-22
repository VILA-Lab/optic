"""Load OPTIC per-layer embeddings into vLLM as compact codes or a dense reconstruction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator


PLE_SUFFIX = "embed_tokens_per_layer.weight"
_REGISTERED = False


def _write_audit(payload: dict[str, Any]) -> None:
    audit_dir = os.environ.get("PLE_EVAL_PLUGIN_AUDIT_DIR", "")
    if not audit_dir:
        return
    path = Path(audit_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"plugin-{os.getpid()}.json"
    events: list[dict[str, Any]] = []
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        prior_events = previous.get("load_events", [])
        if not isinstance(prior_events, list) or not all(
            isinstance(event, dict) for event in prior_events
        ):
            raise RuntimeError(f"invalid prior plugin audit: {output}")
        events = list(prior_events)
    event = {**payload, "load_sequence": len(events) + 1}
    events.append(event)
    document = {
        **payload,
        "load_event_count": len(events),
        "load_events": events,
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _replacement() -> tuple[Any, dict[str, Any]]:
    from safetensors import safe_open

    recon_path = Path(os.environ["PLE_EVAL_RECON_PATH"]).resolve()
    with safe_open(str(recon_path), framework="pt", device="cpu") as handle:
        keys = [key for key in handle.keys() if key.endswith(PLE_SUFFIX)]
        if len(keys) != 1:
            raise RuntimeError(f"expected one PLE tensor in {recon_path}, got {keys}")
        tensor = handle.get_tensor(keys[0])
    if tensor.ndim != 2 or any(int(dimension) <= 0 for dimension in tensor.shape):
        raise RuntimeError(f"invalid PLE reconstruction shape: {tuple(tensor.shape)}")
    expected_text = os.environ.get("PLE_EVAL_EXPECTED_PLE_SHAPE", "").strip()
    expected_shape = None
    if expected_text:
        try:
            expected_shape = tuple(int(value) for value in expected_text.split(","))
        except ValueError as exc:
            raise RuntimeError("PLE_EVAL_EXPECTED_PLE_SHAPE must be V,L_TIMES_D") from exc
        if len(expected_shape) != 2 or tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"PLE reconstruction shape {tuple(tensor.shape)} != expected {expected_shape}"
            )
    return tensor, {
        "mode": "dense_reconstruction_injection",
        "reconstruction_path": str(recon_path),
        "reconstruction_sha256": os.environ.get("PLE_EVAL_RECON_SHA256"),
        "source_key": keys[0],
        "shape": list(tensor.shape),
        "expected_shape": list(expected_shape) if expected_shape else None,
        "dtype": str(tensor.dtype),
    }


def register() -> None:
    """Patch the ordinary safetensors iterator before model weights load."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    from .vllm_compat import install_evaluation_process_lifecycle_compat
    install_evaluation_process_lifecycle_compat()

    artifact = os.environ.get("PLE_OPTIC_ARTIFACT", "").strip()
    if artifact:
        from .vllm_compact import install
        install(artifact)
        return

    recon_path = os.environ.get("PLE_EVAL_RECON_PATH", "").strip()
    from vllm.model_executor.model_loader import default_loader

    original = default_loader.safetensors_weights_iterator

    if not recon_path:
        def passthrough(*args: Any, **kwargs: Any) -> Iterator[tuple[str, Any]]:
            ple_tensor_count = 0
            for name, tensor in original(*args, **kwargs):
                if name.endswith(PLE_SUFFIX):
                    ple_tensor_count += 1
                yield name, tensor
            _write_audit({
                "mode": "checkpoint_dense",
                "pid": os.getpid(),
                "replacement_count": 0,
                "observed_ple_tensor_count": ple_tensor_count,
            })

        default_loader.safetensors_weights_iterator = passthrough
        return

    def wrapped(*args: Any, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        replacement = None
        audit = None
        replacement_count = 0
        for name, tensor in original(*args, **kwargs):
            if name.endswith(PLE_SUFFIX):
                if replacement is None:
                    replacement, audit = _replacement()
                tensor = replacement
                replacement_count += 1
            yield name, tensor
        if replacement_count != 1:
            raise RuntimeError(f"expected to replace one PLE tensor, replaced {replacement_count}")
        _write_audit({**(audit or {}), "pid": os.getpid(), "replacement_count": replacement_count})

    default_loader.safetensors_weights_iterator = wrapped
