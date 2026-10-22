from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .config import (
    canonical_json,
    load_benchmark,
    load_model,
    model_config_binding,
    object_sha256,
    resolve_model_config,
)
from .evaluation_runtime import execution_backend
from .direct_common import generate_once, load_llm, sha256_text, shutdown_llm, write_run_outputs


LETTERS = "ABCD"
PROMPT = """Answer the following multiple choice question.
The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}""".strip()
ANSWER_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")
EXPECTED_PERMUTATION_SHA256 = "ca74eb241f1deb38a02c464b3137d07fef952ba2ec5286f5eb4d718c6400cc47"


def clean_choice(value: Any) -> str:
    text = " " if value is None else str(value).strip()
    text = text.replace(" [title]", ". ")
    return text.replace("  ", " ")


def extract_answer(text: str) -> str | None:
    match = ANSWER_PATTERN.search(text)
    return match.group(1).upper() if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    model_config_path = resolve_model_config(args.model_config or None)
    model_config = load_model(model_config_path)
    if Path(args.model_path).resolve() != Path(model_config["snapshot"]).resolve():
        raise RuntimeError("--model-path differs from the selected --model-config snapshot")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    benchmark = load_benchmark("gpqa_diamond")
    protocol = benchmark["protocol"]
    dataset = load_dataset(
        protocol["dataset"],
        protocol["config"],
        split=protocol["split"],
        revision=protocol["revision"],
        download_mode="reuse_dataset_if_exists",
    )
    if len(dataset) != protocol["questions"]:
        raise RuntimeError(f"GPQA cardinality differs: {len(dataset)}")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    base_docs: list[dict[str, Any]] = []
    for row in dataset:
        correct = clean_choice(row["Correct Answer"])
        base_docs.append({
            "record_id": str(row["Record ID"]),
            "question": str(row["Question"]).strip(),
            "choices": [
                correct,
                clean_choice(row["Incorrect Answer 1"]),
                clean_choice(row["Incorrect Answer 2"]),
                clean_choice(row["Incorrect Answer 3"]),
            ],
            "correct": correct,
            "domain": row.get("High-level domain"),
        })
    if len({row["record_id"] for row in base_docs}) != len(base_docs):
        raise RuntimeError("GPQA Record IDs are not unique")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    rng = random.Random(0)
    prompts: list[str] = []
    golds: list[str] = []
    metadata: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    for repeat_idx in range(protocol["option_permutations"]):
        for doc in base_docs:
            permutation = rng.sample(range(4), 4)
            choices = [doc["choices"][index] for index in permutation]
            gold = LETTERS[choices.index(doc["correct"])]
            content = PROMPT.format(
                question=doc["question"], a=choices[0], b=choices[1], c=choices[2], d=choices[3]
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(rendered)
            golds.append(gold)
            item = {
                "record_id": doc["record_id"],
                "repeat_idx": repeat_idx,
                "permutation": permutation,
                "domain": doc["domain"],
            }
            metadata.append(item)
            permutation_rows.append({
                "record_id": doc["record_id"],
                "repeat_idx": repeat_idx,
                "permutation": permutation,
            })

    permutation_sha = hashlib.sha256(
        canonical_json(permutation_rows).encode("utf-8")
    ).hexdigest()
    if not args.limit and permutation_sha != EXPECTED_PERMUTATION_SHA256:
        raise RuntimeError(f"GPQA permutation identity differs: {permutation_sha}")
    expected_attempts = len(base_docs) * protocol["option_permutations"]
    if len(prompts) != expected_attempts:
        raise RuntimeError("GPQA prompt count differs")

    model = None
    try:
        model, load_seconds = load_llm(
            args.model_path,
            max_model_len=benchmark["runtime"]["max_model_len"],
            seed=protocol["seed"],
            dtype=model_config.get("dtype", "bfloat16"),
            load_format=model_config.get("load_format", "safetensors"),
            quantization=model_config.get("quantization"),
            evaluation_runtime=model_config.get("evaluation_runtime"),
        )
        outputs, generation_seconds = generate_once(
            model,
            prompts,
            temperature=protocol["temperature"],
            top_p=protocol["top_p"],
            top_k=protocol["top_k"],
            max_new_tokens=protocol["max_new_tokens"],
            seed=protocol["seed"],
        )
    finally:
        if model is not None:
            shutdown_llm(model)

    correct = 0
    parse_failures = 0
    max_token_hits = 0
    empty_finals = 0
    repeat_counts = [{"correct": 0, "n": 0} for _ in range(protocol["option_permutations"])]
    raw_rows: list[dict[str, Any]] = []
    for index, (output, prompt, gold, item) in enumerate(zip(outputs, prompts, golds, metadata, strict=True)):
        completion = output.outputs[0]
        raw_text = completion.text
        prediction = extract_answer(raw_text)
        ok = prediction == gold
        correct += int(ok)
        parse_failures += int(prediction is None)
        empty_finals += int(not raw_text.strip())
        cap_hit = len(completion.token_ids) >= protocol["max_new_tokens"]
        max_token_hits += int(cap_hit)
        repeat = repeat_counts[item["repeat_idx"]]
        repeat["correct"] += int(ok)
        repeat["n"] += 1
        raw_rows.append({
            "attempt_idx": index,
            **item,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "gold": gold,
            "prediction": prediction,
            "correct": ok,
            "raw_completion": raw_text,
            "completion_sha256": sha256_text(raw_text),
            "output_token_count": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "max_token_hit": cap_hit,
        })

    summary = {
        "schema_version": "ple-direct-result-v1",
        "benchmark_id": "gpqa_diamond",
        "formal": not bool(args.limit),
        "protocol": protocol,
        "protocol_sha256": object_sha256(protocol),
        "model": model_config,
        "model_config": model_config_binding(model_config_path),
        "dataset": {
            "name": protocol["dataset"],
            "config": protocol["config"],
            "split": protocol["split"],
            "revision": protocol["revision"],
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "questions": len(base_docs),
            "attempts": len(raw_rows),
            "permutation_manifest_sha256": permutation_sha,
        },
        "execution": {
            "backend": execution_backend(model_config, full_batch=True),
            "runtime_metadata": getattr(model, "execution_metadata", None),
            "request_n": 1,
            "best_of": 1,
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
        },
        "integrity": {
            "denominator": len(raw_rows),
            "missing": expected_attempts - len(raw_rows),
            "parse_failures": parse_failures,
            "empty_final": empty_finals,
            "max_token_hits": max_token_hits,
        },
        "result": {
            "correct": correct,
            "n": len(raw_rows),
            "accuracy": correct / len(raw_rows),
            "repeat_metrics": [
                {**row, "accuracy": row["correct"] / row["n"]} for row in repeat_counts
            ],
        },
    }
    write_run_outputs(args.output_dir, summary=summary, raw_rows=raw_rows)
    print(json.dumps(summary["result"], sort_keys=True))


if __name__ == "__main__":
    main()
