#!/usr/bin/env python3
"""Compute per-task scores and seven-task retention for the paper tables.

The script reads completed task directories written by ``ple_eval.task_runner``
(or by ``python -m ple_eval.cli run``) to aggregate their task scores.

Retention of one compressed fit is the equal-weight geometric mean, over the
seven downstream tasks, of ``min(1, compressed score / uncompressed score)``,
where the uncompressed reference is the matched checkpoint named by each table
column.  If any capped ratio is zero, retention is zero.  Rows that list
several fits (fit seeds 0, 1 and 2) report the mean and the sample standard
deviation of the per-fit retention values.

Usage::

    python scripts/compute_paper_tables.py --tables configs/paper/paper_tables.json \
        --discover results --output main-table.json

Discovery reads each completed task's method, benchmark and model configuration.
An optional --runs JSON file can provide an explicit task-directory mapping.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

# Seven downstream task metrics in the order used by every paper table.
TASK_METRICS = (
    "piqa_accuracy",
    "gsm8k_accuracy",
    "gpqa_accuracy",
    "aime_accuracy",
    "ifbench_prompt_loose",
    "infographic_vqa_anls",
    "mmlu_redux_accuracy",
)
BENCHMARK_IDS = (
    "l1_quality_bundle",
    "gpqa_diamond",
    "aime_2026",
    "ifbench",
    "infographic_vqa",
    "mmlu_redux_30x100",
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_reports(integrity: dict[str, Any]) -> list[dict[str, Any]]:
    reports = []
    for value in integrity.get("report_files", []):
        path = Path(str(value))
        if not path.is_file():
            raise RuntimeError(f"EvalScope report is missing: {path}")
        reports.append(read_json(path))
    if not reports:
        raise RuntimeError("EvalScope integrity contains no reports")
    return reports


def _aime_micro_accuracy(integrity: dict[str, Any]) -> float:
    """Micro-average AIME accuracy over all attempts from the review files."""
    correct = 0.0
    denominator = 0
    for value in integrity.get("review_files", []):
        path = Path(str(value))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    correct += float(row["sample_score"]["score"]["value"]["acc"])
                    denominator += 1
    expected = integrity.get("review_rows")
    if expected is not None and denominator != int(expected):
        raise RuntimeError(f"AIME review row count differs: {denominator} != {expected}")
    if denominator <= 0:
        raise RuntimeError("AIME EvalScope reviews are empty")
    return correct / denominator


def extract_task_metrics(task_dir: Path) -> dict[str, float]:
    """Read the headline metric(s) of one completed task directory."""
    benchmark = read_json(task_dir / "benchmark.json")
    benchmark_id = str(benchmark["benchmark_id"])
    driver = str(benchmark["driver"])
    if driver == "l1_bundle":
        summary = read_json(task_dir / "full" / "summary.json")
        return {
            "wikitext_perplexity": float(summary["wikitext"]["perplexity"]),
            "piqa_accuracy": float(summary["piqa"]["accuracy"]),
            "gsm8k_accuracy": float(summary["gsm8k"]["accuracy"]),
        }
    if driver in {"gpqa_direct", "mmlu_redux_direct"}:
        summary = read_json(task_dir / "full" / "summary.json")
        name = "gpqa_accuracy" if benchmark_id == "gpqa_diamond" else "mmlu_redux_accuracy"
        return {name: float(summary["result"]["accuracy"])}
    if driver != "evalscope_api":
        raise RuntimeError(f"unsupported driver {driver} in {task_dir}")
    integrity = read_json(task_dir / "evalscope-integrity.json")
    if integrity.get("status") != "passed":
        raise RuntimeError(f"EvalScope integrity did not pass: {task_dir}")
    if benchmark_id == "ifbench":
        receipt = read_json(task_dir / "author-score.json")
        return {"ifbench_prompt_loose": float(receipt["metrics"]["prompt_level_loose"])}
    if benchmark_id == "aime_2026":
        return {"aime_accuracy": _aime_micro_accuracy(integrity)}
    if benchmark_id == "infographic_vqa":
        reports = _load_reports(integrity)
        if len(reports) != 1:
            raise RuntimeError(f"InfographicVQA must have exactly one report: {task_dir}")
        return {"infographic_vqa_anls": float(reports[0]["score"])}
    raise RuntimeError(f"unsupported EvalScope benchmark {benchmark_id} in {task_dir}")


METHOD_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "methods"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def discover(root: Path) -> dict[str, dict[str, Path]]:
    """Index completed task directories by method configuration and benchmark.

    A task runner copies the selected method configuration into ``method.json``;
    the configuration with identical content gives the run label.  The
    ``method_id`` field is registered as a second label.
    """
    by_content = {_canonical(read_json(path)): path.stem for path in sorted(METHOD_CONFIG_DIR.glob("*.json"))}
    runs: dict[str, dict[str, Path]] = {}
    duplicates: dict[tuple[str, str], list[Path]] = {}
    for status_path in sorted(root.rglob("status.json")):
        task_dir = status_path.parent
        if not (task_dir / "method.json").is_file() or not (task_dir / "benchmark.json").is_file():
            continue
        try:
            if read_json(status_path).get("status") != "completed":
                continue
            method = read_json(task_dir / "method.json")
            benchmark_id = str(read_json(task_dir / "benchmark.json")["benchmark_id"])
        except (OSError, ValueError, KeyError):
            continue
        method_id = str(method.get("method_id") or method.get("label") or task_dir.parent.name)
        labels = list(dict.fromkeys(
            [name for name in [by_content.get(_canonical(method))] if name] + [method_id]))
        model_path = task_dir / "selected-model-config.json"
        if model_path.is_file():
            # A model-qualified label separates one method configuration (for
            # example ``dense``) evaluated on several checkpoints.
            model_id = read_json(model_path).get("model_id")
            labels += [f"{label}@{model_id}" for label in list(labels)]
        for label in labels:
            slot = runs.setdefault(label, {})
            if benchmark_id in slot:
                duplicates.setdefault((label, benchmark_id), [slot[benchmark_id]]).append(task_dir)
            slot[benchmark_id] = task_dir
    for (method_id, benchmark_id), paths in duplicates.items():
        print(f"warning: {len(paths)} completed directories for {method_id}/{benchmark_id}; "
              f"using {paths[-1]}. Pass --runs to select one explicitly.")
    return runs


def load_runs(discover_roots: list[Path], runs_file: Path | None, results_root: Path) -> dict[str, dict[str, Path]]:
    """Combine discovered runs with an explicit run file.

    Relative task directories in the run file are resolved against
    ``results_root``. Explicit entries take precedence over discovered ones.
    """
    runs: dict[str, dict[str, Path]] = {}
    for root in discover_roots:
        for key, value in discover(root).items():
            runs.setdefault(key, {}).update(value)
    if runs_file:
        explicit = read_json(runs_file)
        for key, value in explicit.get("runs", explicit).items():
            runs[key] = {name: (Path(path) if Path(path).is_absolute() else results_root / path)
                         for name, path in value.items()}
    # A label such as ``dense@google/gemma-4-E4B-it#controlled-comparisons`` names a
    # separate uncompressed evaluation; with discovery alone it falls back to the
    # label before ``#``.
    return runs


def lookup(runs: dict[str, dict[str, Path]], label: str) -> dict[str, Path]:
    if label in runs:
        return runs[label]
    base = label.split("#", 1)[0]
    if base in runs:
        return runs[base]
    raise RuntimeError(f"no task directories for run {label}")


def run_scores(task_dirs: dict[str, Path]) -> dict[str, float]:
    missing = [benchmark_id for benchmark_id in BENCHMARK_IDS if benchmark_id not in task_dirs]
    if missing:
        raise RuntimeError(f"missing benchmarks {missing}")
    scores: dict[str, float] = {}
    for benchmark_id in BENCHMARK_IDS:
        scores.update(extract_task_metrics(Path(task_dirs[benchmark_id])))
    return scores


def compression_ratio(task_dirs: dict[str, Path], label: str) -> float | None:
    """Actual resident-byte ratio from the task's method record or, if that
    record lacks the field, from the method configuration of the same name."""
    method = read_json(Path(task_dirs["l1_quality_bundle"]) / "method.json")
    value = method.get("ple_compression_ratio")
    config = METHOD_CONFIG_DIR / f"{label.split('#', 1)[0]}.json"
    if value is None and config.is_file():
        value = read_json(config).get("ple_compression_ratio")
    return float(value) if value is not None else None


def retention_pct(scores: dict[str, float], reference: dict[str, float]) -> float:
    ratios = [min(1.0, scores[name] / reference[name]) for name in TASK_METRICS]
    if any(value <= 0.0 for value in ratios):
        return 0.0
    return 100.0 * math.exp(sum(math.log(value) for value in ratios) / len(ratios))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--discover", type=Path, action="append", default=[])
    parser.add_argument("--runs", type=Path, help="JSON file that maps run labels to task directories")
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                        help="directory against which relative task directories in --runs are resolved")
    parser.add_argument("--only", default="", help="comma-separated table labels to compute")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = load_runs(args.discover, args.runs, args.results_root)

    spec = read_json(args.tables)
    selected = {value.strip() for value in args.only.split(",") if value.strip()}
    cache: dict[str, dict[str, float]] = {}

    def scores_for(label: str) -> dict[str, float]:
        if label not in cache:
            cache[label] = run_scores(lookup(runs, label))
        return cache[label]

    output: list[dict[str, Any]] = []
    for table in spec["tables"]:
        if selected and table["table"] not in selected:
            continue
        print(f"\n== {table['table']}: {table.get('title', '')}")
        for column in table["columns"]:
            reference_label = column["reference"]
            try:
                reference = scores_for(reference_label)
            except RuntimeError as error:
                print(f"  [{column['column']}] reference unavailable: {error}")
                continue
            print(f"  [{column['column']}] reference {reference_label}: " + ", ".join(
                f"{name}={100 * reference[name]:.2f}" for name in TASK_METRICS))
            for row_label, fit_labels in column["rows"].items():
                values, ratios, failures = [], [], []
                per_fit = []
                for fit_label in fit_labels:
                    try:
                        scores = scores_for(fit_label)
                    except RuntimeError as error:
                        failures.append(f"{fit_label}: {error}")
                        continue
                    value = retention_pct(scores, reference)
                    values.append(value)
                    ratio = compression_ratio(lookup(runs, fit_label), fit_label)
                    if ratio is not None:
                        ratios.append(ratio)
                    per_fit.append({"run": fit_label, "retention_pct": value,
                                    "scores_pct": {name: 100 * scores[name] for name in TASK_METRICS}})
                record: dict[str, Any] = {
                    "table": table["table"], "column": column["column"], "row": row_label,
                    "reference": reference_label,
                    "reference_scores_pct": {name: 100 * reference[name] for name in TASK_METRICS},
                    "fits": per_fit, "unavailable": failures,
                }
                if values:
                    record["mean_retention_pct"] = statistics.fmean(values)
                    record["sample_standard_deviation_pct"] = statistics.stdev(values) if len(values) > 1 else None
                if ratios:
                    record["mean_actual_compression_ratio"] = statistics.fmean(ratios)
                output.append(record)
                if values:
                    spread = (f" +/- {record['sample_standard_deviation_pct']:.2f}"
                              if record["sample_standard_deviation_pct"] is not None else "")
                    ratio_text = f"R={record['mean_actual_compression_ratio']:.2f} " if ratios else ""
                    print(f"    {row_label}: {ratio_text}retention={record['mean_retention_pct']:.2f}{spread}"
                          f" over {len(values)} fit(s)")
                    if len(per_fit) == 1:
                        print("      per task: " + ", ".join(
                            f"{name}={per_fit[0]['scores_pct'][name]:.2f}" for name in TASK_METRICS))
                for failure in failures:
                    print(f"    {row_label}: unavailable ({failure})")
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
