from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .config import canonical_json, write_json_atomic
from .vllm_compat import install_evaluation_process_lifecycle_compat, install_best_of_one_compat


# Importing vLLM before transformers/sqlite can make the dynamic loader retain
# the host libstdc++ ahead of the conda copy required by ICU. Defer the shim
# until load_llm imports the full vLLM entrypoint in the normal runner order.
VLLM_BEST_OF_COMPAT: dict[str, Any] = {"status": "deferred"}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    """Write a complete raw sidecar and return its content SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            line = canonical_json(row) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    temporary.replace(path)
    return digest.hexdigest()


def load_llm(
    model_path: str,
    *,
    max_model_len: int,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    prompt_logprobs: bool = False,
    dtype: str = "bfloat16",
    load_format: str = "safetensors",
    quantization: str | None = None,
    evaluation_runtime: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    global VLLM_BEST_OF_COMPAT
    install_evaluation_process_lifecycle_compat()
    if evaluation_runtime is not None:
        from .evaluation_runtime import load_bound_runtime

        started = time.perf_counter()
        runtime = load_bound_runtime(
            evaluation_runtime, model_path=model_path, max_model_len=max_model_len,
            seed=seed, gpu_memory_utilization=gpu_memory_utilization,
            prompt_logprobs=prompt_logprobs, dtype=dtype,
            load_format=load_format, quantization=quantization,
        )
        try:
            VLLM_BEST_OF_COMPAT = install_best_of_one_compat()
        except BaseException:
            runtime.close()
            raise
        return runtime, time.perf_counter() - started
    from vllm import LLM

    VLLM_BEST_OF_COMPAT = install_best_of_one_compat()

    kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": dtype,
        "tensor_parallel_size": 1,
        "seed": seed,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "load_format": load_format,
        "enforce_eager": False,
        "disable_log_stats": True,
    }
    if quantization:
        kwargs["quantization"] = quantization
    if prompt_logprobs:
        # This preserves exact causal log probabilities while bounding the
        # temporary full-vocabulary log-softmax allocation.
        kwargs["max_num_batched_tokens"] = 512
    started = time.perf_counter()
    return LLM(**kwargs), time.perf_counter() - started


def shutdown_llm(model: Any) -> None:
    if getattr(model, "runtime_backend", None) is not None:
        model.close()
        return
    engine = model.llm_engine
    try:
        engine.engine_core.shutdown(timeout=None if os.environ.get("PLE_EVAL_NO_TIME_LIMITS") == "1" else 30)
    finally:
        # vLLM 0.19.1's in-process executor shutdown stops workers but leaves
        # the LLM -> LLMEngine -> EngineCore -> model-executor object graph
        # reachable.  Detach it before collection so a second engine in this
        # process does not inherit the first model and CUDA graph pool.
        model.llm_engine = None
        del engine
        del model
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        try:
            destroy_model_parallel()
        finally:
            try:
                destroy_distributed_environment()
            finally:
                gc.collect()
                import torch

                torch.cuda.empty_cache()


def reclaim_llm_memory_after_reference_drop() -> None:
    """Collect an LLM after its caller has dropped the final strong reference."""
    gc.collect()
    import torch

    torch.cuda.empty_cache()


def cuda_memory_snapshot(label: str) -> dict[str, int | str]:
    """Capture allocator and device memory at a completed engine boundary."""
    import torch

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "label": label,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


def generate_once(
    model: Any,
    prompts: list[str],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[Any], float]:
    """Submit the complete prompt set through its bound runtime."""
    global VLLM_BEST_OF_COMPAT
    VLLM_BEST_OF_COMPAT = install_best_of_one_compat()
    from vllm import SamplingParams

    sampling = SamplingParams(
        n=1,
        best_of=1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    started = time.perf_counter()
    outputs = model.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.perf_counter() - started
    if len(outputs) != len(prompts):
        raise RuntimeError(f"output count differs: {len(outputs)} != {len(prompts)}")
    if any(len(output.outputs) != 1 for output in outputs):
        raise RuntimeError("request_n must equal one for every prompt")
    return outputs, elapsed


def write_run_outputs(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    raw_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    sidecar = output_dir / "raw-completions.jsonl.gz"
    summary["raw_sidecar"] = {
        "path": str(sidecar),
        "rows": len(raw_rows),
        "uncompressed_content_sha256": write_jsonl_gzip(sidecar, raw_rows),
    }
    write_json_atomic(output_dir / "summary.json", summary)
