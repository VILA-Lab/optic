from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import load_benchmark, write_json_atomic


def jsonl_rows(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def final_text(model_output: dict[str, Any]) -> str:
    choices = model_output.get("choices") or []
    if len(choices) != 1:
        return ""
    content = (choices[0].get("message") or {}).get("content") or []
    if isinstance(content, str):
        return content
    return "".join(str(part.get("text", "")) for part in content if part.get("type") == "text")


def modality_order(messages: list[dict[str, Any]]) -> list[str]:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return [str(part.get("type")) for part in content]
        return ["text"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    protocol = benchmark["protocol"]
    prediction_paths = sorted(args.output_dir.glob("**/predictions/**/*.jsonl"))
    review_paths = sorted(args.output_dir.glob("**/reviews/**/*.jsonl"))
    if not prediction_paths or not review_paths:
        raise RuntimeError("EvalScope predictions or reviews are missing")

    prediction_count = 0
    prediction_keys: set[tuple[str, Any]] = set()
    duplicates = 0
    request_errors = 0
    choice_count_errors = 0
    empty_final = 0
    max_token_hits = 0
    image_order_failures = 0
    prompt_hash = hashlib.sha256()
    for path, _, row in jsonl_rows(prediction_paths):
        prediction_count += 1
        key = (str(path.relative_to(args.output_dir)), row.get("index"))
        duplicates += int(key in prediction_keys)
        prediction_keys.add(key)
        model_output = row.get("model_output") or {}
        request_errors += int(bool(model_output.get("error")))
        choices = model_output.get("choices") or []
        choice_count_errors += int(len(choices) != 1)
        text = final_text(model_output)
        empty_final += int(not text.strip())
        output_tokens = int((model_output.get("usage") or {}).get("output_tokens") or 0)
        stop_reason = choices[0].get("stop_reason") if len(choices) == 1 else None
        max_token_hits += int(stop_reason == "length" or output_tokens >= protocol["max_new_tokens"])
        messages = row.get("messages") or []
        prompt_hash.update(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        if benchmark["runtime"]["kind"] == "vision":
            order = modality_order(messages)
            first_text = order.index("text") if "text" in order else -1
            last_image = max((index for index, value in enumerate(order) if value == "image"), default=-1)
            if not order or first_text < 0 or last_image < 0 or last_image > first_text:
                image_order_failures += 1

    review_count = sum(1 for _ in jsonl_rows(review_paths))
    if args.benchmark == "aime_2026":
        expected = protocol["attempts"]
    else:
        expected = protocol.get("n")
    count_ok = prediction_count > 0 if args.smoke else prediction_count == expected
    review_ok = review_count == prediction_count
    integrity = {
        "schema_version": "ple-evalscope-integrity-v1",
        "benchmark_id": args.benchmark,
        "formal": not args.smoke,
        "status": "passed" if count_ok and review_ok and duplicates == 0 and choice_count_errors == 0 and image_order_failures == 0 else "failed",
        "expected_denominator": None if args.smoke else expected,
        "prediction_rows": prediction_count,
        "review_rows": review_count,
        "duplicates": duplicates,
        "request_errors_retained_in_denominator": request_errors,
        "choice_count_errors": choice_count_errors,
        "empty_final_retained_in_denominator": empty_final,
        "max_token_hits_retained_in_denominator": max_token_hits,
        "image_before_text_failures": image_order_failures,
        "rendered_messages_stream_sha256": prompt_hash.hexdigest(),
        "prediction_files": [str(path) for path in prediction_paths],
        "review_files": [str(path) for path in review_paths],
        "report_files": [str(path) for path in sorted(args.output_dir.glob("**/reports/**/*.json"))],
    }
    write_json_atomic(args.output_dir / "evalscope-integrity.json", integrity)
    print(json.dumps(integrity, sort_keys=True))
    if integrity["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

