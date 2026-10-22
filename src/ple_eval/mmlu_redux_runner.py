from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .config import (
    load_benchmark,
    load_model,
    model_config_binding,
    object_sha256,
    resolve_model_config,
)
from .evaluation_runtime import execution_backend
from .direct_common import generate_once, load_llm, sha256_text, shutdown_llm, write_run_outputs


LETTERS = "ABCD"
PROMPT = """For the multiple-choice question related to {category}, which option (A-D) is correct?.

Question:{question}{options}
End the answer with the following:
The best answer is (the_answer_letter) where the (the_answer_letter) is one of 'A', 'B', 'C' or 'D'."""


def extract_answer(text: str) -> str | None:
    hits = re.findall(r"best answer is\s*\(?\s*([A-D])\s*\)?", text, flags=re.IGNORECASE)
    return hits[-1].upper() if hits else None


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

    benchmark = load_benchmark("mmlu_redux_30x100")
    protocol = benchmark["protocol"]
    subjects = protocol["subject_names"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    prompts: list[str] = []
    golds: list[str] = []
    metadata: list[dict[str, Any]] = []
    fingerprints: dict[str, str | None] = {}
    remaining = args.limit if args.limit else None
    for subject in subjects:
        if remaining == 0:
            break
        dataset = load_dataset(
            protocol["dataset"],
            subject,
            split=protocol["split"],
            revision=protocol["revision"],
            download_mode="reuse_dataset_if_exists",
        )
        if len(dataset) != protocol["subject_n"]:
            raise RuntimeError(f"MMLU-Redux {subject} cardinality differs: {len(dataset)}")
        fingerprints[subject] = getattr(dataset, "_fingerprint", None)
        take = len(dataset) if remaining is None else min(remaining, len(dataset))
        for row_idx, row in enumerate(dataset.select(range(take))):
            choices = list(row["choices"])
            if len(choices) != 4:
                raise RuntimeError(f"MMLU-Redux choice count differs: {subject}:{row_idx}")
            answer_index = int(row["answer"])
            if not 0 <= answer_index < 4:
                raise RuntimeError(f"MMLU-Redux answer out of range: {subject}:{row_idx}")
            options = "".join(f"\n{LETTERS[index]}: {str(choice).strip()}" for index, choice in enumerate(choices))
            content = PROMPT.format(
                category=subject.replace("_", " "),
                question=str(row["question"]).strip(),
                options=options,
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(rendered)
            golds.append(LETTERS[answer_index])
            metadata.append({
                "question_id": f"{subject}:{row_idx}",
                "category": subject,
                "source": row.get("source"),
                "error_type": row.get("error_type"),
            })
        if remaining is not None:
            remaining -= take

    expected_n = args.limit if args.limit else protocol["n"]
    if len(prompts) != expected_n:
        raise RuntimeError(f"MMLU-Redux prompt count differs: {len(prompts)} != {expected_n}")

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
    empty_finals = 0
    max_token_hits = 0
    by_subject: dict[str, dict[str, int]] = {}
    raw_rows: list[dict[str, Any]] = []
    for index, (output, prompt, gold, item) in enumerate(zip(outputs, prompts, golds, metadata, strict=True)):
        completion = output.outputs[0]
        text = completion.text
        prediction = extract_answer(text)
        ok = prediction == gold
        correct += int(ok)
        parse_failures += int(prediction is None)
        empty_finals += int(not text.strip())
        cap_hit = len(completion.token_ids) >= protocol["max_new_tokens"]
        max_token_hits += int(cap_hit)
        bucket = by_subject.setdefault(item["category"], {"correct": 0, "n": 0})
        bucket["correct"] += int(ok)
        bucket["n"] += 1
        raw_rows.append({
            "attempt_idx": index,
            **item,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "gold": gold,
            "prediction": prediction,
            "correct": ok,
            "raw_completion": text,
            "completion_sha256": sha256_text(text),
            "output_token_count": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "max_token_hit": cap_hit,
        })

    summary = {
        "schema_version": "ple-direct-result-v1",
        "benchmark_id": "mmlu_redux_30x100",
        "formal": not bool(args.limit),
        "protocol": protocol,
        "protocol_sha256": object_sha256(protocol),
        "model": model_config,
        "model_config": model_config_binding(model_config_path),
        "dataset": {
            "name": protocol["dataset"],
            "revision": protocol["revision"],
            "split": protocol["split"],
            "fingerprints": fingerprints,
            "subjects": list(fingerprints),
            "n": len(raw_rows),
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
            "missing": expected_n - len(raw_rows),
            "parse_failures": parse_failures,
            "empty_final": empty_finals,
            "max_token_hits": max_token_hits,
        },
        "result": {
            "correct": correct,
            "n": len(raw_rows),
            "accuracy": correct / len(raw_rows),
            "subject_metrics": {
                key: {**row, "accuracy": row["correct"] / row["n"]}
                for key, row in sorted(by_subject.items())
            },
        },
    }
    write_run_outputs(args.output_dir, summary=summary, raw_rows=raw_rows)
    print(json.dumps({key: summary["result"][key] for key in ("correct", "n", "accuracy")}, sort_keys=True))


if __name__ == "__main__":
    main()
