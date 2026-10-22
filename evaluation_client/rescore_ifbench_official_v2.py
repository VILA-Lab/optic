#!/usr/bin/env python3
"""Rescore IFBench with the pinned official scorer.

The reported IFBench score comes from the official scorer, and EvalScope is
replayed for comparison.  EvalScope 1.8.1 vendors an older
``AlphabetLoopChecker`` which raises ``IndexError`` when a loose-response
normalization contains no words.  The pinned official revision treats that
case as a failed instruction, and this wrapper applies the same rule to the
EvalScope replay.
"""

from __future__ import annotations

import argparse
import copy
import json
import string
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from rescore_ifbench_official import (
    author_score,
    final_text,
    load_author_evaluator,
    remove_null_kwargs,
    sha256,
)


SCHEMA_VERSION = "ifbench-pinned-author-rescore-v2"
EVALSCOPE_ALPHABET_GUARD = "evalscope-1.8.1-alphabet-loop-author-empty-word-guard"
EVALSCOPE_LAST_FIRST_GUARD = (
    "evalscope-1.8.1-last-word-first-next-author-empty-sentence-guard"
)


def evalscope_diagnostic_score(
    evaluator: Any, doc: dict[str, Any], response: str
) -> tuple[dict[str, Any], int]:
    """Replay EvalScope while changing only its stale alphabet-loop edge case."""

    from evalscope.benchmarks.ifbench import instructions as evalscope_instructions

    checker = evalscope_instructions.AlphabetLoopChecker
    original = checker.check_following
    last_first_checker = evalscope_instructions.LastWordFirstNextChecker
    original_last_first = last_first_checker.check_following
    guard_hits = 0

    def author_guard(instance: Any, value: str) -> bool:
        nonlocal guard_hits
        normalized = value.translate(str.maketrans("", "", string.punctuation))
        words = normalized.strip("".join(string.punctuation) + " ").split()
        if not words:
            guard_hits += 1
            return False
        return bool(original(instance, value))

    def last_first_author_guard(instance: Any, value: str) -> bool:
        """Match the author's failure semantics for an empty split sentence.

        EvalScope's checker indexes the first and last word without first
        checking that both sentence fragments contain a word.  The pinned
        official scorer treats this response as an instruction failure instead
        of raising.
        """

        nonlocal guard_hits
        try:
            return bool(original_last_first(instance, value))
        except IndexError:
            guard_hits += 1
            return False

    checker.check_following = author_guard
    last_first_checker.check_following = last_first_author_guard
    try:
        score = evaluator.process_results(copy.deepcopy(doc), [response])
    finally:
        checker.check_following = original
        last_first_checker.check_following = original_last_first
    return score, guard_hits


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
        for row in (
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
        )
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
    evalscope_diagnostic_corrections: list[dict[str, Any]] = []
    prompt_strict = 0
    prompt_loose = 0
    instruction_strict = 0
    instruction_loose = 0
    instruction_total = 0
    evalscope_prompt_strict = 0
    evalscope_prompt_loose = 0
    empty_final = 0
    request_errors = 0
    stop_reasons: Counter[str] = Counter()

    for row in rows:
        doc = copy.deepcopy(row["metadata"])
        key = str(doc["key"])
        seen[key] += 1
        source = author_data.get(key)
        fields_match = source is not None and all(
            doc.get(field) == source.get(field)
            for field in ("prompt", "instruction_id_list")
        )
        kwargs_match = (
            source is not None
            and remove_null_kwargs(source["kwargs"])
            == remove_null_kwargs(doc["kwargs"])
        )
        if not fields_match or not kwargs_match:
            dataset_mismatches.append(key)

        response = final_text(row)
        empty_final += int(not response.strip())
        request_errors += int(bool(row["model_output"].get("error")))
        choice = row["model_output"]["choices"][0]
        stop_reasons[str(choice.get("stop_reason") or choice.get("finish_reason"))] += 1
        strict, loose = author_score(
            author, source if source is not None else doc, response
        )
        prompt_strict += int(strict.follow_all_instructions)
        prompt_loose += int(loose.follow_all_instructions)
        instruction_strict += sum(strict.follow_instruction_list)
        instruction_loose += sum(loose.follow_instruction_list)
        instruction_total += len(strict.follow_instruction_list)

        author_prompt = {
            "prompt_level_strict": float(strict.follow_all_instructions),
            "prompt_level_loose": float(loose.follow_all_instructions),
        }
        evalscope_score, guard_hits = evalscope_diagnostic_score(
            evalscope_evaluator, doc, response
        )
        evalscope_prompt = {
            "prompt_level_strict": evalscope_score["prompt_level_strict"],
            "prompt_level_loose": evalscope_score["prompt_level_loose"],
        }
        evalscope_prompt_strict += int(evalscope_prompt["prompt_level_strict"])
        evalscope_prompt_loose += int(evalscope_prompt["prompt_level_loose"])
        if guard_hits:
            evalscope_diagnostic_corrections.append(
                {
                    "key": key,
                    "instruction_id_list": doc["instruction_id_list"],
                    "kind": EVALSCOPE_ALPHABET_GUARD,
                    "guarded_call_count": guard_hits,
                    "author": author_prompt,
                    "evalscope_after_author_guard": evalscope_prompt,
                }
            )
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
    duplicate_keys = sorted(key for key, value in seen.items() if value != 1)
    missing_keys = sorted(expected_keys - set(seen))
    unexpected_keys = sorted(set(seen) - expected_keys)
    if (
        len(rows) != len(expected_keys)
        or missing_keys
        or unexpected_keys
        or duplicate_keys
        or dataset_mismatches
    ):
        raise RuntimeError(
            "IFBench row identity or dataset binding differs: "
            f"rows={len(rows)}, expected={len(expected_keys)}, "
            f"missing={missing_keys}, unexpected={unexpected_keys}, "
            f"duplicates={duplicate_keys}, dataset_mismatches={sorted(set(dataset_mismatches))}"
        )
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
        "schema_version": SCHEMA_VERSION,
        "benchmark": "IFBench",
        "artifact_dir": str(args.artifact_dir),
        "official_repo_revision": revision,
        "official_primary_metric": "prompt_level_loose",
        "diagnostic_only_scorer": "EvalScope 1.8.1",
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
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "duplicate_keys": duplicate_keys,
            "dataset_field_mismatches": sorted(set(dataset_mismatches)),
            "empty_final": empty_final,
            "request_errors": request_errors,
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "evalscope_prompt_scorer_divergences": scorer_divergences,
            "evalscope_diagnostic_corrections": evalscope_diagnostic_corrections,
            "evalscope_corrected_diagnostic_prompt_metrics": {
                "denominator": len(rows),
                "prompt_level_strict": evalscope_prompt_strict / len(rows),
                "prompt_level_strict_correct": evalscope_prompt_strict,
                "prompt_level_loose": evalscope_prompt_loose / len(rows),
                "prompt_level_loose_correct": evalscope_prompt_loose,
                "role": "diagnostic_only_not_the_official_benchmark_metric",
            },
        },
        "prediction_sha256": {
            str(path): sha256(path) for path in prediction_files
        },
        "dataset_sha256": sha256(data_path),
        "official_scorer_sha256": {
            str(path.relative_to(args.official_repo)): sha256(path)
            for path in scorer_files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
