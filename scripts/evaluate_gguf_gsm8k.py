#!/usr/bin/env python3
"""Full paired GSM8K evaluation through llama.cpp's native HTTP API.

Uses OPTIC's 8-shot protocol.
See docs/gguf.md for preparation, server configuration and resumption commands.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import aiohttp


PROTOCOL_ID = "gemma4_ple_l1_wt2_piqa_gsm8k8_v1"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
FILES = {
    "gsm8k-full-reference-1319.jsonl": "4764da6f33fa8527950723b1c9180dedbcfa9f67ceced054330355844ffe175e",
    "gsm8k-fewshot-train-first-8.jsonl": "bf8be618fb3a0ac4804b85eb2ea531e2153d84ec10e374e0598303f749f1e00c",
}
MAX_NEW_TOKENS = 8192
CONTEXT_PER_SLOT = 10240
TEMPLATE = (
    "Here are some examples of how to solve similar problems:\n\n"
    "{fewshot}\n\n{question}\nPlease reason step by step, and put your final "
    "answer within \\boxed{{}}."
)
EMPTY_THOUGHT_SUFFIX = "<|turn>model\n<|channel>thought\n<channel|>"
MODEL_TURN_SUFFIX = "<|turn>model\n"
GENERATION = {
    "n_predict": MAX_NEW_TOKENS,
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "seed": 0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "samplers": ["temperature"],
    "cache_prompt": False,
    "stream": False,
    "return_tokens": True,
    "stop": [],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def append_record(path: Path, value: dict[str, Any]) -> None:
    with path.open("ab") as stream:
        stream.write(json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


async def http(session: aiohttp.ClientSession, url: str, path: str, body: Any = None) -> Any:
    method = session.get if body is None else session.post
    kwargs = {} if body is None else {"json": body}
    async with method(url.rstrip("/") + path, **kwargs) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}: {text[:2000]}")
        return json.loads(text)


def protocol() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": DATASET_REVISION,
        "test_n": 1319,
        "fewshot_train_indices": list(range(8)),
        "thinking": False,
        "user_template": TEMPLATE,
        "context_per_slot": CONTEXT_PER_SLOT,
        "generation": GENERATION,
        "primary_scorer": "evalscope.metrics.math_parser.extract_answer + math_equal(defaults)",
        "supplementary_scorer": "Decimal exact numeric equality of the same extracted answer",
        "lm_eval_reference": {
            "revision": "c1c4bea3777f73e188395264083adcf454913344",
            "task": "gsm8k/gsm8k.yaml",
            "note": "Inspected for reference; its 5-shot prompt is a different protocol.",
        },
    }


def validate_dataset(source: Path, official_cache: Path | None) -> tuple[list, list, dict]:
    loaded = {}
    for name, expected in FILES.items():
        path = source / name
        actual = file_sha(path)
        if actual != expected:
            raise ValueError(f"Dataset hash mismatch: {path}: {actual}")
        loaded[name] = read_jsonl(path)
    test, shots = loaded.keys()
    rows, examples = loaded[test], loaded[shots]
    if len(rows) != 1319 or len(examples) != 8:
        raise ValueError("Expected all 1,319 test rows and first 8 training examples")
    verified = {"files": FILES, "official_arrow_validation": None}
    if official_cache is not None:
        from datasets import Dataset

        details = {}
        for split, local_rows in (("test", rows), ("train", examples)):
            path = official_cache / f"gsm8k-{split}.arrow"
            reference = Dataset.from_file(str(path))
            for index, row in enumerate(local_rows):
                original = reference[index]
                reasoning, gold = original["answer"].split("#### ")
                if not (
                    row["source_row_index"] == index
                    and row["question"] == original["question"]
                    and str(row["gold_answer"]) == gold.strip()
                    and ("reasoning" not in row or row["reasoning"] == reasoning.strip())
                ):
                    raise ValueError(f"Official dataset differs at {split}[{index}]")
            details[split] = {"arrow_path": str(path), "sha256": file_sha(path), "matched_rows": len(local_rows)}
        verified["official_arrow_validation"] = details
    return rows, examples, verified


async def prepare(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rows, examples, identity = validate_dataset(args.source_dir, args.official_cache)
    shots = "\n\n".join(
        row["question"] + "\n\nReasoning:\n" + row["reasoning"]
        + "\n\nANSWER: \\boxed{" + str(row["gold_answer"]) + "}"
        for row in examples
    )
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        props = await http(session, args.url, "/props")
        prepared = []
        # Template work is CPU-only. Bound its concurrency to avoid starving a server.
        semaphore = asyncio.Semaphore(8)

        async def render(index: int, row: dict) -> dict:
            async with semaphore:
                user = TEMPLATE.format(fewshot=shots, question=row["question"])
                rendered = await http(session, args.url, "/apply-template", {
                    "messages": [{"role": "user", "content": user}],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "add_generation_prompt": True,
                })
                prompt = rendered["prompt"]
                # Official E4B GGUF's canonical template uses a plain model turn
                # when thinking is off. Older templates append an empty thought.
                if not prompt.endswith((MODEL_TURN_SUFFIX, EMPTY_THOUGHT_SUFFIX)) or "<|think|>" in prompt:
                    raise ValueError(f"Expected Gemma4 no-thinking generation prefix, got {prompt[-160:]!r}; use --jinja")
                bos = props.get("bos_token", "")
                bos_added = bool(bos and not prompt.startswith(bos))
                if bos_added:
                    prompt = bos + prompt
                tokens = (await http(session, args.url, "/tokenize", {
                    "content": prompt, "add_special": False, "parse_special": True,
                }))["tokens"]
                if len(tokens) + MAX_NEW_TOKENS > CONTEXT_PER_SLOT:
                    raise ValueError(f"Prompt {index} cannot fit the retained zero-truncation protocol: {len(tokens)} + {MAX_NEW_TOKENS}")
                return {
                    "index": index, "item_id": row["item_id"], "source_row_index": row["source_row_index"],
                    "question": row["question"], "gold": str(row["gold_answer"]), "user_content": user,
                    "prompt": prompt, "prompt_sha256": sha(prompt), "prompt_token_count": len(tokens),
                    "bos_added_to_server_rendering": bos_added,
                    "prompt_token_ids_sha256": sha(json_bytes(tokens)),
                    # Exact token IDs prevent implicit BOS insertion in /completion.
                    "prompt_token_ids": tokens,
                }

        prepared = await asyncio.gather(*(render(i, row) for i, row in enumerate(rows)))
    prompt_path = args.output / "prompts.jsonl"
    with prompt_path.open("wb") as stream:
        for row in prepared:
            stream.write(json_bytes(row) + b"\n")
    write_json(args.output / "fewshot.json", examples)
    write_json(args.output / "template-server-props.json", props)
    manifest = {
        "created_utc": now(), "protocol": protocol(), "dataset_identity": identity,
        "prompts_sha256": file_sha(prompt_path), "n": len(prepared),
        "min_prompt_tokens": min(row["prompt_token_count"] for row in prepared),
        "max_prompt_tokens": max(row["prompt_token_count"] for row in prepared),
        "chat_template_sha256": sha(props.get("chat_template", "")),
        "script_sha256": file_sha(Path(__file__)), "template_source_url": args.url,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def scorer_identity() -> dict[str, Any]:
    from evalscope.metrics import math_parser

    path = Path(inspect.getfile(math_parser))
    return {"package": "evalscope", "version": importlib.metadata.version("evalscope"),
            "math_parser_path": str(path), "math_parser_sha256": file_sha(path),
            "math_equal_defaults": {"include_percentage": True, "is_close": True, "timeout": False}}


def score(text: str, gold: str) -> dict[str, Any]:
    from evalscope.metrics.math_parser import extract_answer, math_equal

    prediction = extract_answer(text)
    correct = prediction is not None and math_equal(str(prediction), str(gold))
    strict = False
    try:
        left = Decimal(str(prediction).replace(",", "").replace("$", "").strip())
        right = Decimal(gold.replace(",", "").strip())
        strict = left.is_finite() and right.is_finite() and left == right
    except (InvalidOperation, ValueError):
        pass
    return {"prediction": prediction, "correct": bool(correct), "strict_numeric_correct": bool(strict),
            "parse_failure": prediction is None or not str(prediction).strip()}


def completed_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # The writer fsyncs records; tolerate only an incomplete final line after a crash.
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        cut = payload.rfind(b"\n") + 1
        backup = path.with_suffix(path.suffix + f".incomplete-{time.time_ns()}")
        backup.write_bytes(payload[cut:])
        path.write_bytes(payload[:cut])
    return read_jsonl(path)


def statistics_summary(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "n": n, "correct": correct, "accuracy": correct / n if n else None,
        "strict_numeric_correct": sum(row["strict_numeric_correct"] for row in rows),
        "strict_numeric_accuracy": sum(row["strict_numeric_correct"] for row in rows) / n if n else None,
        "correct_with_capped_outputs_counted_incorrect": sum(
            row["correct"] and not row["max_token_hit"] and not row["context_truncated"] for row in rows),
        "strict_numeric_correct_with_capped_outputs_counted_incorrect": sum(
            row["strict_numeric_correct"] and not row["max_token_hit"] and not row["context_truncated"] for row in rows),
        "parse_failures": sum(row["parse_failure"] for row in rows),
        "max_token_hits": sum(row["max_token_hit"] for row in rows),
        "context_truncations": sum(row["context_truncated"] for row in rows),
        "empty_completions": sum(not row["raw_completion"].strip() for row in rows),
        "total_output_tokens": sum(row["output_token_count"] for row in rows),
        "median_output_tokens": statistics.median(row["output_token_count"] for row in rows) if n else None,
        "max_output_tokens": max((row["output_token_count"] for row in rows), default=0),
        "request_seconds_total": sum(row["request_seconds"] for row in rows),
        "finish_reasons": {reason: sum(row["stop_type"] == reason for row in rows)
                           for reason in sorted({row["stop_type"] for row in rows})},
    }


async def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    prepared_manifest = json.loads((args.prepared / "manifest.json").read_text())
    prompt_path = args.prepared / "prompts.jsonl"
    if file_sha(prompt_path) != prepared_manifest["prompts_sha256"]:
        raise ValueError("Prepared prompts hash changed")
    all_prompts = read_jsonl(prompt_path)
    if len(all_prompts) != 1319 or prepared_manifest["protocol"] != protocol():
        raise ValueError("Prepared protocol does not match this runner")
    prompts = all_prompts[:args.pilot] if args.pilot else all_prompts
    scorer = scorer_identity()
    urls = [url.rstrip("/") for url in args.url]
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        props = await asyncio.gather(*(http(session, url, "/props") for url in urls))
        for url, item in zip(urls, props, strict=True):
            if sha(item.get("chat_template", "")) != prepared_manifest["chat_template_sha256"]:
                raise ValueError(f"Chat template differs on server {url}")
            actual_context = item.get("default_generation_settings", {}).get("n_ctx")
            if actual_context != CONTEXT_PER_SLOT:
                raise ValueError(f"Server {url} has context/slot={actual_context}, expected {CONTEXT_PER_SLOT}")
            if item.get("model_path") and Path(item["model_path"]).resolve() != args.model_path.resolve():
                raise ValueError(f"Server {url} model path differs from --model-path")
        identity = {
            "protocol": protocol(), "prompts_sha256": prepared_manifest["prompts_sha256"],
            "model_path": str(args.model_path.resolve()), "model_sha256": file_sha(args.model_path),
            "label": args.label, "pilot_n": args.pilot, "scorer": scorer,
            "urls": urls, "concurrency_per_server": args.concurrency,
            "server_props": dict(zip(urls, props, strict=True)),
            "external_server_config": json.loads(args.server_config.read_text()) if args.server_config else None,
        }
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            # Resumption can use different ports/parallelism; scientific inputs stay fixed.
            for key in ("protocol", "prompts_sha256", "model_sha256", "label", "pilot_n", "scorer"):
                if existing["identity"][key] != identity[key]:
                    raise ValueError(f"Cannot resume with changed {key}")
            manifest = existing
        else:
            manifest = {"created_utc": now(), "identity": identity, "sessions": [],
                        "prepared_directory": str(args.prepared.resolve()),
                        "prepared_manifest": prepared_manifest, "script_sha256": file_sha(Path(__file__))}
            (args.output / "runner-source.py").write_bytes(Path(__file__).read_bytes())
        session_record = {"started_utc": now(), "urls": urls, "concurrency_per_server": args.concurrency,
                          "server_props": identity["server_props"], "external_server_config": identity["external_server_config"]}
        manifest["sessions"].append(session_record)
        write_json(manifest_path, manifest)
        result_path = args.output / "outputs.jsonl"
        finished = completed_rows(result_path)
        by_index = {}
        for row in finished:
            index = row["index"]
            if index in by_index or index >= len(prompts) or row["prompt_sha256"] != prompts[index]["prompt_sha256"]:
                raise ValueError(f"Duplicate/foreign result at index {index}")
            by_index[index] = row
        queue: asyncio.Queue = asyncio.Queue()
        for item in prompts:
            if item["index"] not in by_index:
                queue.put_nowait(item)
        started = time.perf_counter()
        print(f"{now()} {args.label}: {len(by_index)} retained; {queue.qsize()} remaining; {len(urls) * args.concurrency} workers", flush=True)

        async def worker(url: str) -> None:
            while not queue.empty():
                item = queue.get_nowait()
                body = {**GENERATION, "prompt": item["prompt_token_ids"]}
                response = None
                attempts = []
                for attempt in range(args.retries + 1):
                    attempt_started = time.perf_counter()
                    try:
                        response = await http(session, url, "/completion", body)
                        request_seconds = time.perf_counter() - attempt_started
                        if not isinstance(response.get("content"), str):
                            raise RuntimeError(f"Malformed completion: {response}")
                        break
                    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
                        error = {"index": item["index"], "url": url, "attempt": attempt,
                                 "utc": now(), "error": repr(exc), "seconds": time.perf_counter() - attempt_started}
                        append_record(args.output / "errors.jsonl", error)
                        attempts.append(error)
                        if attempt == args.retries:
                            raise
                        await asyncio.sleep(min(2 ** attempt, 10))
                text = response["content"]
                tokens = response.get("tokens", [])
                count = int(response.get("timings", {}).get("predicted_n", len(tokens)))
                record = {
                    "index": item["index"], "item_id": item["item_id"], "gold": item["gold"],
                    "prompt_sha256": item["prompt_sha256"], "prompt_token_count": item["prompt_token_count"],
                    "raw_completion": text, "completion_sha256": sha(text), **score(text, item["gold"]),
                    "output_token_count": count, "stop_type": response.get("stop_type", "unknown"),
                    "max_token_hit": count >= MAX_NEW_TOKENS or response.get("stop_type") == "limit",
                    "context_truncated": bool(response.get("truncated", False)),
                    "request_seconds": request_seconds, "server_url": url, "finished_utc": now(),
                    "failed_attempts": attempts, "server_response": response,
                }
                append_record(result_path, record)
                by_index[item["index"]] = record
                queue.task_done()
                n = len(by_index)
                if n % 10 == 0 or n == len(prompts):
                    elapsed = time.perf_counter() - started
                    print(f"{now()} {args.label}: {n}/{len(prompts)} correct={sum(row['correct'] for row in by_index.values())}; session={elapsed:.1f}s", flush=True)

        tasks = [asyncio.create_task(worker(url)) for url in urls for _ in range(args.concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.perf_counter() - started
            session_record.update({"ended_utc": now(), "elapsed_seconds": elapsed, "completed_n": len(by_index)})
            write_json(manifest_path, manifest)
            result_rows = sorted(by_index.values(), key=lambda row: row["index"])
            summary = statistics_summary(result_rows)
            summary.update({"label": args.label, "formal": not bool(args.pilot) and len(result_rows) == 1319,
                            "expected_n": len(prompts), "complete": len(result_rows) == len(prompts),
                            "zero_truncation": not summary["max_token_hits"] and not summary["context_truncations"],
                            "session_elapsed_seconds": elapsed,
                            "all_session_elapsed_seconds": sum(row.get("elapsed_seconds", 0) for row in manifest["sessions"]),
                            "outputs_sha256": file_sha(result_path) if result_path.exists() else None,
                            "model_sha256": identity["model_sha256"], "prompts_sha256": identity["prompts_sha256"]})
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)


def compare(args: argparse.Namespace) -> None:
    paths = [args.baseline, args.candidate]
    manifests = [json.loads((path / "manifest.json").read_text()) for path in paths]
    validation_status = [json.loads((path / "validation-status.json").read_text())
                         if (path / "validation-status.json").exists() else None for path in paths]
    for key in ("protocol", "prompts_sha256", "scorer"):
        if manifests[0]["identity"][key] != manifests[1]["identity"][key]:
            raise ValueError(f"Paired runs differ in {key}")
    groups = [completed_rows(path / "outputs.jsonl") for path in paths]
    indexes = []
    for rows in groups:
        mapping = {row["index"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate output indices")
        indexes.append(mapping)
    if set(indexes[0]) != set(indexes[1]):
        raise ValueError("Paired runs have different completed item sets")
    pairs = [(indexes[0][i], indexes[1][i]) for i in sorted(indexes[0])]
    for baseline, candidate in pairs:
        if baseline["prompt_sha256"] != candidate["prompt_sha256"] or baseline["gold"] != candidate["gold"]:
            raise ValueError("Paired prompt/gold differs")
    lost = [b["index"] for a, b in pairs if a["correct"] and not b["correct"]]
    gained = [b["index"] for a, b in pairs if not a["correct"] and b["correct"]]
    n = len(pairs)
    delta = (len(gained) - len(lost)) / n if n else None
    # Paired Bernoulli difference SE; the raw discordant indices remain available.
    variance = ((len(gained) + len(lost)) / n - delta ** 2) / n if n else None
    result = {
        "n": n, "full_test_split": n == 1319,
        "formal": n == 1319 and all(status is None or status.get("quality_claim_eligible", True)
                                    for status in validation_status),
        "validation_status": {"baseline": validation_status[0], "candidate": validation_status[1]},
        "baseline": statistics_summary(groups[0]),
        "candidate": statistics_summary(groups[1]), "accuracy_delta": delta,
        "paired_standard_error": math.sqrt(max(variance, 0)) if n else None,
        "lost_correct_indices": lost, "gained_correct_indices": gained,
        "same_extracted_answer": sum(a["prediction"] == b["prediction"] for a, b in pairs),
        "identical_completion": sum(a["completion_sha256"] == b["completion_sha256"] for a, b in pairs),
        "baseline_path": str(args.baseline.resolve()), "candidate_path": str(args.candidate.resolve()),
        "prompts_sha256": manifests[0]["identity"]["prompts_sha256"],
        "scorer": manifests[0]["identity"]["scorer"],
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


def self_test() -> None:
    cases = [
        (r"ANSWER: \boxed{18}", "18", True),
        (r"\boxed{-2.50}", "-2.5", True),
        (r"\boxed{1,234.50}", "1234.5", True),
        ("#### -3", "-3", True),
        ("We computed 42. The final answer is 17.", "17", True),
        (r"\boxed{19}", "18", False),
        ("No numeric answer", "18", False),
    ]
    for text, gold, expected in cases:
        result = score(text, gold)
        if result["correct"] != expected or result["strict_numeric_correct"] != expected:
            raise AssertionError((text, result, expected))
    percentage = score(r"\boxed{1800}", "18")
    if not percentage["correct"] or percentage["strict_numeric_correct"]:
        raise AssertionError("evalscope percentage equivalence changed")
    print(json.dumps({"scorer": scorer_identity(), "cases": len(cases) + 1, "passed": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Validate sources and save exact server-rendered prompts/tokens")
    prep.add_argument("--source-dir", type=Path, required=True)
    prep.add_argument("--official-cache", type=Path)
    prep.add_argument("--url", required=True)
    prep.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run", help="Run or resume a full test (or explicitly labelled pilot)")
    execute.add_argument("--prepared", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--url", action="append", required=True, help="Repeat for identical-model server replicas")
    execute.add_argument("--model-path", type=Path, required=True)
    execute.add_argument("--label", required=True)
    execute.add_argument("--concurrency", type=int, default=4, help="Requests in flight per server")
    execute.add_argument("--server-config", type=Path, help="JSON with exact launch commands, binary/source revisions, GPU config")
    execute.add_argument("--pilot", type=int, default=0, help="First N only; never a formal benchmark result")
    execute.add_argument("--request-timeout", type=float, default=1800)
    execute.add_argument("--retries", type=int, default=2)
    paired = commands.add_parser("compare", help="Paired accuracy and item-level disagreement analysis")
    paired.add_argument("--baseline", type=Path, required=True)
    paired.add_argument("--candidate", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    commands.add_parser("self-test", help="Check numeric scorer semantics without any generation")
    args = parser.parse_args()
    if args.command == "prepare":
        asyncio.run(prepare(args))
    elif args.command == "run":
        if args.concurrency < 1 or args.pilot < 0 or args.pilot > 1319:
            parser.error("concurrency must be positive and pilot must be 0..1319")
        asyncio.run(run(args))
    elif args.command == "compare":
        compare(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
