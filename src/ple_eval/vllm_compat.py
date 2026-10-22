"""Compatibility shims that keep the locked sampling settings on vLLM 0.19.1."""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any, Callable


def install_evaluation_process_lifecycle_compat() -> dict[str, Any]:
    """Remove upstream readiness and shutdown cutoffs for evaluation processes.

    vLLM 0.19.1 maps shutdown(timeout=None) to a five-second forced kill.
    Its process managers capture a module-level shutdown function when they
    are created. Install cooperative termination and unbounded joins before
    creating any manager. Negative ZeroMQ poll timeouts mean indefinite wait.
    """
    if os.environ.get("PLE_EVAL_NO_TIME_LIMITS") != "1":
        return {"status": "not_requested"}
    os.environ["VLLM_ENGINE_READY_TIMEOUT_S"] = "-1"
    import vllm
    if vllm.__version__ != "0.19.1":
        raise RuntimeError("Evaluation process lifecycle compatibility requires the bound vLLM 0.19.1 implementation")
    from vllm.v1 import utils as process_utils
    from vllm import envs
    from vllm.v1.engine import utils as engine_utils

    def cooperative_shutdown(procs: list[Any], timeout: float | None = None) -> None:
        for process in procs:
            if process.is_alive():
                process.terminate()
        for process in procs:
            if process.pid is not None:
                process.join()

    cooperative_shutdown._ple_cooperative_shutdown = True  # type: ignore[attr-defined]
    process_utils.shutdown = cooperative_shutdown
    engine_utils.shutdown = cooperative_shutdown
    envs.VLLM_ENGINE_READY_TIMEOUT_S = -1
    core_client = sys.modules.get("vllm.v1.engine.core_client")
    if core_client is not None:
        core_client.VLLM_ENGINE_READY_TIMEOUT_S = -1
    return {"status": "installed", "engine_readiness_wait": "indefinite",
            "process_shutdown": "cooperative_termination_and_unbounded_join"}


def best_of_one_factory(original: Callable[..., Any]) -> Callable[..., Any]:
    """Accept the removed ``best_of=1`` spelling and reject every other value."""

    def compatible(*args: Any, **kwargs: Any) -> Any:
        best_of = kwargs.pop("best_of", 1)
        if best_of != 1:
            raise ValueError("PLE evaluation permits best_of=1 only")
        return original(*args, **kwargs)

    compatible.__name__ = getattr(original, "__name__", "SamplingParams")
    compatible.__qualname__ = getattr(original, "__qualname__", compatible.__name__)
    compatible.__doc__ = getattr(original, "__doc__", None)
    compatible._ple_best_of_one_compat = True  # type: ignore[attr-defined]
    compatible._ple_original = original  # type: ignore[attr-defined]
    return compatible


def install_best_of_one_compat() -> dict[str, Any]:
    """Patch vLLM only when its current constructor removed ``best_of``."""
    import vllm

    current = vllm.SamplingParams
    if getattr(current, "_ple_best_of_one_compat", False):
        return {"status": "already_installed", "native_best_of": False}
    if "best_of" in inspect.signature(current).parameters:
        return {"status": "native", "native_best_of": True}
    vllm.SamplingParams = best_of_one_factory(current)
    return {
        "status": "installed",
        "native_best_of": False,
        "semantic_boundary": "best_of=1 is ignored; any other value raises",
    }
