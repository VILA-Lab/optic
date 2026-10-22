from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    file_sha256,
    load_benchmark,
    load_model,
    model_config_binding,
    object_sha256,
    resolve_model_config,
    write_json_atomic,
)
from .evaluation_runtime import execution_backend
from .direct_common import (
    cuda_memory_snapshot,
    load_llm,
    sha256_text,
    shutdown_llm,
    write_jsonl_gzip,
    write_run_outputs,
)


GSM8K_TEMPLATE = (
    "Here are some examples of how to solve similar problems:\n\n"
    "{fewshot}\n\n{question}\nPlease reason step by step, and put your final "
    "answer within \\boxed{{}}."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def score_prompt_tokens(token_ids: list[int], prompt_logprobs: list[Any], start_index: int) -> float:
    if len(prompt_logprobs) != len(token_ids):
        raise RuntimeError("prompt-logprob length differs")
    score = 0.0
    for index in range(start_index, len(token_ids)):
        candidates = prompt_logprobs[index]
        token_id = token_ids[index]
        if candidates is None or token_id not in candidates:
            raise RuntimeError(f"scored token absent from prompt logprobs at {index}")
        value = float(candidates[token_id].logprob)
        if not math.isfinite(value):
            raise RuntimeError("prompt log probability is not finite")
        score += value
    return score


def build_wikitext(tokenizer: Any, path: Path) -> tuple[list[list[int]], int]:
    content_ids = tokenizer.encode(path.read_text("utf-8"), add_special_tokens=False)
    if tokenizer.bos_token_id is None:
        raise RuntimeError("tokenizer has no BOS token")
    segments = [
        [int(tokenizer.bos_token_id), *content_ids[offset : offset + 2047]]
        for offset in range(0, len(content_ids), 2047)
    ]
    return segments, len(content_ids)


def run_wikitext(model: Any, tokenizer: Any, config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    from vllm import SamplingParams

    segments, full_content_tokens = build_wikitext(tokenizer, Path(config["path"]))
    if full_content_tokens != config["tokens"] or len(segments) != config["segments"]:
        raise RuntimeError("WikiText token or segment identity differs")
    selected = segments[:3] if smoke else segments
    sampling = SamplingParams(
        n=1, best_of=1, temperature=0, top_p=1.0, top_k=0,
        max_tokens=1, prompt_logprobs=1, seed=0,
    )
    started = time.perf_counter()
    outputs = []
    for segment in selected:
        batch = model.generate([segment], sampling, use_tqdm=False)
        if len(batch) != 1:
            raise RuntimeError("WikiText output count differs")
        outputs.extend(batch)
    elapsed = time.perf_counter() - started
    total_nll = 0.0
    scored_tokens = 0
    rows = []
    for index, (segment, output) in enumerate(zip(selected, outputs, strict=True)):
        score = score_prompt_tokens(segment, output.prompt_logprobs, 1)
        token_count = len(segment) - 1
        total_nll -= score
        scored_tokens += token_count
        rows.append({
            "segment_index": index,
            "scored_token_count": token_count,
            "negative_log_likelihood": -score,
            "token_ids_sha256": hashlib.sha256(
                json.dumps(segment, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        })
    return {
        "formal": not smoke,
        "full_content_token_count": full_content_tokens,
        "scored_token_count": scored_tokens,
        "full_segment_count": len(segments),
        "scored_segment_count": len(selected),
        "negative_log_likelihood": total_nll,
        "perplexity": math.exp(total_nll / scored_tokens),
        "elapsed_seconds": elapsed,
        "segments": rows,
    }


def encode_piqa_pair(tokenizer: Any, context: str, choice: str) -> tuple[list[int], int]:
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        choice = context[-trailing_spaces:] + choice
        context = context[:-trailing_spaces]
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    whole_ids = tokenizer.encode(context + choice, add_special_tokens=False)
    if tokenizer.bos_token_id is None:
        raise RuntimeError("tokenizer has no BOS token")
    return [int(tokenizer.bos_token_id), *whole_ids], len(context_ids) + 1


def run_piqa(model: Any, tokenizer: Any, config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    from vllm import SamplingParams

    source_rows = read_jsonl(Path(config["path"]))
    if len(source_rows) != config["n"]:
        raise RuntimeError("PIQA cardinality differs")
    selected = source_rows[:3] if smoke else source_rows
    prompts: list[list[int]] = []
    boundaries: list[int] = []
    for row in selected:
        context = f"Question: {row['goal']}\nAnswer:"
        for choice in (row["sol1"], row["sol2"]):
            token_ids, boundary = encode_piqa_pair(tokenizer, context, str(choice))
            prompts.append(token_ids)
            boundaries.append(boundary)
    sampling = SamplingParams(
        n=1, best_of=1, temperature=0, top_p=1.0, top_k=0,
        max_tokens=1, prompt_logprobs=1, seed=0,
    )
    started = time.perf_counter()
    outputs = []
    for offset in range(0, len(prompts), 32):
        batch_prompts = prompts[offset : offset + 32]
        batch = model.generate(batch_prompts, sampling, use_tqdm=False)
        if len(batch) != len(batch_prompts):
            raise RuntimeError("PIQA output count differs")
        outputs.extend(batch)
    elapsed = time.perf_counter() - started
    choice_scores = [
        score_prompt_tokens(prompt, output.prompt_logprobs, boundary)
        for prompt, output, boundary in zip(prompts, outputs, boundaries, strict=True)
    ]
    correct = 0
    normalized_correct = 0
    rows = []
    for index, row in enumerate(selected):
        scores = choice_scores[index * 2 : index * 2 + 2]
        counts = [
            len(prompts[index * 2 + choice_index]) - boundaries[index * 2 + choice_index]
            for choice_index in range(2)
        ]
        raw_prediction = max(range(2), key=lambda choice_index: scores[choice_index])
        normalized_prediction = max(
            range(2), key=lambda choice_index: scores[choice_index] / counts[choice_index]
        )
        label = int(row["label"])
        correct += int(raw_prediction == label)
        normalized_correct += int(normalized_prediction == label)
        rows.append({
            "item_id": row["item_id"],
            "label": label,
            "choice_log_likelihoods": scores,
            "choice_token_counts": counts,
            "raw_prediction": raw_prediction,
            "normalized_prediction": normalized_prediction,
        })
    return {
        "formal": not smoke,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "normalized_correct": normalized_correct,
        "normalized_accuracy": normalized_correct / len(rows),
        "elapsed_seconds": elapsed,
        "items": rows,
    }


def build_gsm8k(tokenizer: Any, config: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fewshot_rows = read_jsonl(Path(config["fewshot_path"]))
    request_rows = read_jsonl(Path(config["path"]))
    if len(fewshot_rows) != config["few_shot"] or len(request_rows) != config["n"]:
        raise RuntimeError("GSM8K source cardinality differs")
    shots = [
        row["question"]
        + "\n\nReasoning:\n"
        + row["reasoning"]
        + "\n\nANSWER: \\boxed{"
        + str(row["gold_answer"])
        + "}"
        for row in fewshot_rows
    ]
    fewshot = "\n\n".join(shots)
    prompts = [
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": GSM8K_TEMPLATE.format(fewshot=fewshot, question=row["question"]),
            }],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in request_rows
    ]
    return prompts, request_rows


def run_gsm8k(model: Any, tokenizer: Any, config: dict[str, Any], smoke: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from evalscope.metrics.math_parser import extract_answer, math_equal
    from vllm import SamplingParams

    prompts, source_rows = build_gsm8k(tokenizer, config)
    if smoke:
        prompts = prompts[:3]
        source_rows = source_rows[:3]
    prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    model_limit = load_benchmark("l1_quality_bundle")["runtime"]["max_model_len"]
    if max(prompt_lengths) + config["max_new_tokens"] > model_limit:
        raise RuntimeError(
            f"GSM8K zero-truncation contract fails: {max(prompt_lengths)} + "
            f"{config['max_new_tokens']} > {model_limit}"
        )
    sampling = SamplingParams(
        n=1,
        best_of=1,
        temperature=config["temperature"],
        top_p=1.0,
        top_k=0,
        max_tokens=config["max_new_tokens"],
        seed=config["seed"],
    )
    started = time.perf_counter()
    outputs = model.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.perf_counter() - started
    if len(outputs) != len(source_rows) or any(len(output.outputs) != 1 for output in outputs):
        raise RuntimeError("GSM8K output cardinality differs")
    correct = 0
    parse_failures = 0
    max_token_hits = 0
    raw_rows = []
    output_tokens = 0
    for index, (prompt, source, output) in enumerate(zip(prompts, source_rows, outputs, strict=True)):
        completion = output.outputs[0]
        prediction = extract_answer(completion.text)
        ok = prediction is not None and math_equal(str(prediction), str(source["gold_answer"]))
        correct += int(ok)
        parse_failures += int(prediction is None)
        cap_hit = len(completion.token_ids) >= config["max_new_tokens"]
        max_token_hits += int(cap_hit)
        output_tokens += len(completion.token_ids)
        raw_rows.append({
            "attempt_idx": index,
            "item_id": source["item_id"],
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "prompt_token_count": prompt_lengths[index],
            "gold": str(source["gold_answer"]),
            "prediction": None if prediction is None else str(prediction),
            "correct": bool(ok),
            "raw_completion": completion.text,
            "completion_sha256": sha256_text(completion.text),
            "output_token_count": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "max_token_hit": cap_hit,
        })
    return {
        "formal": not smoke,
        "n": len(raw_rows),
        "correct": correct,
        "accuracy": correct / len(raw_rows),
        "parse_failures": parse_failures,
        "max_token_hits": max_token_hits,
        "empty_final": sum(not row["raw_completion"].strip() for row in raw_rows),
        "max_prompt_tokens": max(prompt_lengths),
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
    }, raw_rows


def validate_files(protocol: dict[str, Any]) -> dict[str, str]:
    expected = {
        Path(protocol["wikitext"]["path"]): protocol["wikitext"]["sha256"],
        Path(protocol["piqa"]["path"]): protocol["piqa"]["sha256"],
        Path(protocol["gsm8k"]["path"]): protocol["gsm8k"]["sha256"],
        Path(protocol["gsm8k"]["fewshot_path"]): protocol["gsm8k"]["fewshot_sha256"],
    }
    actual = {str(path): file_sha256(path) for path in expected}
    mismatches = {str(path): value for path, value in expected.items() if actual[str(path)] != value}
    if mismatches:
        raise RuntimeError(f"locked L1 input differs: {mismatches}")
    return actual


def _run_stage_process(
    *,
    stage: str,
    stage_output: Path,
    model_path: str,
    model_config_path: Path,
    smoke: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "ple_eval.l1_runner",
        "--model-path",
        model_path,
        "--model-config",
        str(model_config_path),
        "--output-dir",
        str(stage_output.parent / "unused-final-output"),
        "--engine-stage",
        stage,
        "--stage-output",
        str(stage_output),
    ]
    if smoke:
        command.append("--smoke")
    process = subprocess.Popen(command)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"L1 {stage} engine subprocess {process.pid} failed with {return_code}"
        )
    return {
        "stage": stage,
        "pid": process.pid,
        "exit_code": return_code,
        "stage_output": str(stage_output),
        "stage_output_sha256": file_sha256(stage_output),
    }


def _gpu_memory_after_process_exit(label: str) -> dict[str, int | str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdecimal():
        raise RuntimeError(
            "separate-process L1 requires the verified host ordinal in CUDA_VISIBLE_DEVICES"
        )
    output = subprocess.check_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    selected = None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0] == visible:
            selected = fields
            break
    if selected is None:
        raise RuntimeError("verified host GPU is absent from nvidia-smi memory inventory")
    return {
        "label": label,
        "physical_gpu_uuid": selected[1],
        "device_free_bytes": int(selected[2]) * 1024 * 1024,
        "device_total_bytes": int(selected[3]) * 1024 * 1024,
    }


def _run_prompt_stage(
    args: argparse.Namespace,
    model_config: dict[str, Any],
    tokenizer: Any,
    benchmark: dict[str, Any],
) -> None:
    if args.stage_output is None:
        raise RuntimeError("--stage-output is required for a prompt engine child")
    protocol = benchmark["protocol"]
    model = None
    lifecycle: list[dict[str, int | str]] = []
    try:
        model, load_seconds = load_llm(
            args.model_path,
            max_model_len=benchmark["runtime"]["max_model_len"],
            seed=0,
            gpu_memory_utilization=benchmark["runtime"]["gpu_memory_utilization"],
            prompt_logprobs=True,
            dtype=model_config.get("dtype", "bfloat16"),
            load_format=model_config.get("load_format", "safetensors"),
            quantization=model_config.get("quantization"),
            evaluation_runtime=model_config.get("evaluation_runtime"),
        )
        lifecycle.append(cuda_memory_snapshot("prompt_engine_loaded"))
        wikitext = run_wikitext(model, tokenizer, protocol["wikitext"], args.smoke)
        piqa = run_piqa(model, tokenizer, protocol["piqa"], args.smoke)
        lifecycle.append(cuda_memory_snapshot("prompt_scoring_completed"))
    finally:
        if model is not None:
            shutdown_llm(model)
    write_json_atomic(
        args.stage_output,
        {
            "schema_version": "ple-l1-prompt-stage-result-v1",
            "pid": os.getpid(),
            "engine_load_seconds": load_seconds,
            "runtime_metadata": getattr(model, "execution_metadata", None),
            "engine_lifecycle_memory": lifecycle,
            "wikitext": wikitext,
            "piqa": piqa,
        },
    )


def _run_generation_stage(
    args: argparse.Namespace,
    model_config: dict[str, Any],
    tokenizer: Any,
    benchmark: dict[str, Any],
) -> None:
    if args.stage_output is None:
        raise RuntimeError("--stage-output is required for a generation engine child")
    protocol = benchmark["protocol"]
    model = None
    lifecycle: list[dict[str, int | str]] = []
    try:
        model, load_seconds = load_llm(
            args.model_path,
            max_model_len=benchmark["runtime"]["max_model_len"],
            seed=protocol["gsm8k"]["seed"],
            gpu_memory_utilization=benchmark["runtime"]["gpu_memory_utilization"],
            dtype=model_config.get("dtype", "bfloat16"),
            load_format=model_config.get("load_format", "safetensors"),
            quantization=model_config.get("quantization"),
            evaluation_runtime=model_config.get("evaluation_runtime"),
        )
        lifecycle.append(cuda_memory_snapshot("generation_engine_loaded"))
        gsm8k, raw_rows = run_gsm8k(model, tokenizer, protocol["gsm8k"], args.smoke)
        lifecycle.append(cuda_memory_snapshot("generation_completed"))
    finally:
        if model is not None:
            shutdown_llm(model)
    raw_path = args.stage_output.with_name("generation-raw-completions.jsonl.gz")
    raw_sha256 = write_jsonl_gzip(raw_path, raw_rows)
    write_json_atomic(
        args.stage_output,
        {
            "schema_version": "ple-l1-generation-stage-result-v1",
            "pid": os.getpid(),
            "engine_load_seconds": load_seconds,
            "runtime_metadata": getattr(model, "execution_metadata", None),
            "engine_lifecycle_memory": lifecycle,
            "gsm8k": gsm8k,
            "raw_sidecar": {
                "path": str(raw_path),
                "rows": len(raw_rows),
                "uncompressed_content_sha256": raw_sha256,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--engine-stage",
        choices=("orchestrate", "prompt", "generation"),
        default="orchestrate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--stage-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    model_config_path = resolve_model_config(args.model_config or None)
    model_config = load_model(model_config_path)
    if Path(args.model_path).resolve() != Path(model_config["snapshot"]).resolve():
        raise RuntimeError("--model-path differs from the selected --model-config snapshot")

    benchmark = load_benchmark("l1_quality_bundle")
    protocol = benchmark["protocol"]
    input_hashes = validate_files(protocol)
    if args.engine_stage != "orchestrate":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=True
        )
        if args.engine_stage == "prompt":
            _run_prompt_stage(args, model_config, tokenizer, benchmark)
        else:
            _run_generation_stage(args, model_config, tokenizer, benchmark)
        return

    if args.stage_output is not None:
        raise RuntimeError("--stage-output is valid only for an engine child")
    stage_root = args.output_dir.with_name(args.output_dir.name + ".engine-stages")
    stage_root.mkdir(parents=True, exist_ok=False)
    prompt_path = stage_root / "prompt.json"
    generation_path = stage_root / "generation.json"
    engine_processes = []
    lifecycle_memory: list[dict[str, int | str]] = []

    engine_processes.append(
        _run_stage_process(
            stage="prompt",
            stage_output=prompt_path,
            model_path=args.model_path,
            model_config_path=model_config_path,
            smoke=args.smoke,
        )
    )
    prompt_exit = _gpu_memory_after_process_exit("prompt_engine_process_exited")
    prompt_stage = json.loads(prompt_path.read_text("utf-8"))
    lifecycle_memory.extend(prompt_stage["engine_lifecycle_memory"])
    lifecycle_memory.append(prompt_exit)

    engine_processes.append(
        _run_stage_process(
            stage="generation",
            stage_output=generation_path,
            model_path=args.model_path,
            model_config_path=model_config_path,
            smoke=args.smoke,
        )
    )
    generation_exit = _gpu_memory_after_process_exit("generation_engine_process_exited")
    generation_stage = json.loads(generation_path.read_text("utf-8"))
    lifecycle_memory.extend(generation_stage["engine_lifecycle_memory"])
    lifecycle_memory.append(generation_exit)

    wikitext = prompt_stage["wikitext"]
    piqa = prompt_stage["piqa"]
    gsm8k = generation_stage["gsm8k"]
    raw_stage = Path(generation_stage["raw_sidecar"]["path"])
    with gzip.open(raw_stage, "rt", encoding="utf-8") as handle:
        raw_rows = [json.loads(line) for line in handle if line.strip()]
    if len(raw_rows) != generation_stage["raw_sidecar"]["rows"]:
        raise RuntimeError("generation stage raw sidecar cardinality differs")

    summary = {
        "schema_version": "ple-l1-bundle-result-v1",
        "benchmark_id": "l1_quality_bundle",
        "formal": not args.smoke,
        "protocol": protocol,
        "protocol_sha256": object_sha256(protocol),
        "model": model_config,
        "model_config": model_config_binding(model_config_path),
        "dataset_hashes": input_hashes,
        "execution": {
            "backend": execution_backend(model_config),
            "engine_isolation": "separate_child_processes",
            "engine_processes": engine_processes,
            "runtime_metadata": {"prompt": prompt_stage.get("runtime_metadata"),
                                 "generation": generation_stage.get("runtime_metadata")},
            "request_n": 1,
            "best_of": 1,
            "vllm_enable_v1_multiprocessing": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING", ""
            ),
            "engine_lifecycle_memory": lifecycle_memory,
            "prompt_logprob_engine_load_seconds": prompt_stage["engine_load_seconds"],
            "generation_engine_load_seconds": generation_stage["engine_load_seconds"],
        },
        "wikitext": wikitext,
        "piqa": piqa,
        "gsm8k": gsm8k,
    }
    write_run_outputs(args.output_dir, summary=summary, raw_rows=raw_rows)
    print(json.dumps({
        "wikitext_perplexity": wikitext["perplexity"],
        "piqa_accuracy": piqa["accuracy"],
        "gsm8k_accuracy": gsm8k["accuracy"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
