#!/usr/bin/env python3
"""Rescore saved IFBench generations with a pinned author checkout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_text(row: dict[str, Any]) -> str:
    content = row["model_output"]["choices"][0]["message"].get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def load_author_evaluator(repo: Path) -> Any:
    sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("evaluation_lib")
    finally:
        sys.path.pop(0)


def author_score(author: Any, doc: dict[str, Any], response: str) -> tuple[Any, Any]:
    # The official runner calls strict before loose on the same InputExample.
    # Strict removes null placeholder kwargs in place; loose then sees that
    # normalized object. Preserve this ordering exactly.
    inp = author.InputExample(
        key=doc["key"],
        instruction_id_list=copy.deepcopy(doc["instruction_id_list"]),
        prompt=doc["prompt"],
        kwargs=copy.deepcopy(doc["kwargs"]),
    )
    prompt_to_response = {doc["prompt"]: response}
    strict = author.test_instruction_following_strict(inp, prompt_to_response)
    loose = author.test_instruction_following_loose(inp, prompt_to_response)
    return strict, loose


def remove_null_kwargs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if value is not None} for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    author = load_author_evaluator(args.official_repo)
    from evalscope.benchmarks.ifbench import evaluation_lib as evalscope_evaluator

    data_path = args.official_repo / "data" / "IFBench_test.jsonl"
    author_data = {
        str(row["key"]): row
        for row in (json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines())
    }
    prediction_files = sorted((args.artifact_dir / "predictions").glob("*/*.jsonl"))
    if not prediction_files:
        raise RuntimeError(f"No prediction files under {args.artifact_dir}")
    rows = [
        json.loads(line)
        for path in prediction_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    seen: Counter[str] = Counter()
    dataset_mismatches: list[str] = []
    scorer_divergences: list[dict[str, Any]] = []
    prompt_strict = 0
    prompt_loose = 0
    instruction_strict = 0
    instruction_loose = 0
    instruction_total = 0
    empty_final = 0
    request_errors = 0
    stop_reasons: Counter[str] = Counter()

    for row in rows:
        doc = copy.deepcopy(row["metadata"])
        key = str(doc["key"])
        seen[key] += 1
        source = author_data.get(key)
        fields_match = source is not None and all(
            doc.get(field) == source.get(field) for field in ("prompt", "instruction_id_list")
        )
        kwargs_match = source is not None and remove_null_kwargs(source["kwargs"]) == remove_null_kwargs(doc["kwargs"])
        if not fields_match or not kwargs_match:
            dataset_mismatches.append(key)

        response = final_text(row)
        empty_final += int(not response.strip())
        request_errors += int(bool(row["model_output"].get("error")))
        choice = row["model_output"]["choices"][0]
        stop_reasons[str(choice.get("stop_reason") or choice.get("finish_reason"))] += 1
        strict, loose = author_score(author, source if source is not None else doc, response)
        prompt_strict += int(strict.follow_all_instructions)
        prompt_loose += int(loose.follow_all_instructions)
        instruction_strict += sum(strict.follow_instruction_list)
        instruction_loose += sum(loose.follow_instruction_list)
        instruction_total += len(strict.follow_instruction_list)

        evalscope_score = evalscope_evaluator.process_results(copy.deepcopy(doc), [response])
        evalscope_prompt = {
            "prompt_level_strict": evalscope_score["prompt_level_strict"],
            "prompt_level_loose": evalscope_score["prompt_level_loose"],
        }
        author_prompt = {
            "prompt_level_strict": float(strict.follow_all_instructions),
            "prompt_level_loose": float(loose.follow_all_instructions),
        }
        if author_prompt != evalscope_prompt:
            scorer_divergences.append(
                {
                    "key": key,
                    "instruction_id_list": doc["instruction_id_list"],
                    "author": author_prompt,
                    "evalscope": evalscope_prompt,
                }
            )

    expected_keys = set(author_data)
    revision = subprocess.check_output(
        ["git", "-C", str(args.official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    scorer_files = [
        args.official_repo / "evaluation_lib.py",
        args.official_repo / "ifbench" / "instructions.py",
        args.official_repo / "ifbench" / "instructions_registry.py",
        args.official_repo / "ifbench" / "instructions_util.py",
    ]
    result = {
        "benchmark": "IFBench",
        "artifact_dir": str(args.artifact_dir),
        "official_repo_revision": revision,
        "official_primary_metric": "prompt_level_loose",
        "samples": len(rows),
        "metrics": {
            "prompt_level_strict": prompt_strict / len(rows),
            "instruction_level_strict": instruction_strict / instruction_total,
            "prompt_level_loose": prompt_loose / len(rows),
            "instruction_level_loose": instruction_loose / instruction_total,
        },
        "counts": {
            "prompt_strict_correct": prompt_strict,
            "prompt_loose_correct": prompt_loose,
            "instruction_strict_correct": instruction_strict,
            "instruction_loose_correct": instruction_loose,
            "instruction_total": instruction_total,
        },
        "audit": {
            "expected_samples": len(expected_keys),
            "missing_keys": sorted(expected_keys - set(seen)),
            "unexpected_keys": sorted(set(seen) - expected_keys),
            "duplicate_keys": sorted(key for key, value in seen.items() if value != 1),
            "dataset_field_mismatches": sorted(set(dataset_mismatches)),
            "empty_final": empty_final,
            "request_errors": request_errors,
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "evalscope_prompt_scorer_divergences": scorer_divergences,
        },
        "prediction_sha256": {str(path): sha256(path) for path in prediction_files},
        "dataset_sha256": sha256(data_path),
        "official_scorer_sha256": {str(path.relative_to(args.official_repo)): sha256(path) for path in scorer_files},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
