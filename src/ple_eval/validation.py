from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import (
    ROOT,
    file_sha256,
    load_benchmark,
    load_external_dependencies,
    load_level,
    load_method,
    load_model,
    load_vlm,
    model_config_binding,
    read_json,
)


def validate_protocol_lock(lock_path: Path | None = None) -> dict[str, Any]:
    lock_path = lock_path or (
        ROOT / "configs" / "evaluation/protocol.json"
    )
    lock_path = Path(lock_path).expanduser().resolve(strict=True)
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if not isinstance(lock, dict) or lock.get("schema_version") not in {
        "ple-paper-protocol-lock-v1", "ple-paper-protocol-execution-lock-v1",
    }:
        raise ValueError("Unsupported protocol lock schema_version.")
    locked_files = lock.get("locked_files")
    if not isinstance(locked_files, dict) or not locked_files:
        raise ValueError("A protocol lock must contain a nonempty locked_files mapping.")
    for value, expected_sha in locked_files.items():
        if not isinstance(value, str) or not value or not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError("Every locked file requires a nonempty path and a lowercase SHA-256.")
    scope = lock.get("scope")
    if (not isinstance(scope, list) or not scope or
            any(level not in {"L1", "L2", "L3"} for level in scope) or len(scope) != len(set(scope))):
        raise ValueError("A protocol lock requires a nonempty, unique suite-level scope.")
    file_rows = []
    for value, expected_sha in lock["locked_files"].items():
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        actual_sha = file_sha256(path) if path.is_file() else None
        file_rows.append({
            "path": str(path),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "ok": actual_sha == expected_sha,
        })

    level_rows = []
    benchmark_ids: list[str] = []
    for level_name in lock["scope"]:
        expected = lock["level_entries"][level_name]
        actual = load_level(level_name)["benchmarks"]
        level_rows.append({
            "level": level_name,
            "expected": expected,
            "actual": actual,
            "ok": actual == expected and len(actual) == len(set(actual)),
        })
        benchmark_ids.extend(actual)

    benchmark_rows = []
    for benchmark_id in benchmark_ids:
        record = load_benchmark(benchmark_id)
        protocol = record["protocol"]
        benchmark_rows.append({
            "benchmark_id": benchmark_id,
            "expected_protocol_id": lock["protocol_ids"][benchmark_id],
            "actual_protocol_id": protocol.get("protocol_id"),
            "ok": bool(
                record.get("status") == "locked"
                and record.get("approved_for_paired_ple") is True
                and protocol.get("protocol_id") == lock["protocol_ids"][benchmark_id]
                and protocol.get("request_n") == 1
                and protocol.get("best_of") == 1
                and protocol.get("attempt_aggregation") != "best"
            ),
        })

    return {
        # These identify the exact lock bytes used by this check, rather than
        # a later read of a possibly changed path.
        "lock_path": str(lock_path),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "lock_id": lock["lock_id"],
        "status": lock["status"],
        "ok": bool(
            lock.get("status") == "locked"
            and lock.get("invariants", {}).get("one_protocol_per_benchmark") is True
            and all(row["ok"] for row in file_rows)
            and all(row["ok"] for row in level_rows)
            and all(row["ok"] for row in benchmark_rows)
        ),
        "files": file_rows,
        "levels": level_rows,
        "benchmarks": benchmark_rows,
    }


def require_protocol_lock_benchmark(
    lock_path: Path, benchmark_id: str, expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Check a selected execution lock and its coverage before model effects."""
    result = validate_protocol_lock(lock_path)
    if expected_sha256 is not None and result["lock_sha256"] != expected_sha256:
        raise RuntimeError("The selected protocol lock differs from the planned SHA-256.")
    if not result["ok"]:
        raise RuntimeError("The selected protocol lock failed its file or protocol checks.")
    if benchmark_id not in {row["benchmark_id"] for row in result["benchmarks"]}:
        raise RuntimeError(f"The selected protocol lock does not cover {benchmark_id}.")
    return result


def _path_check(path_value: str | None, expected_sha: str | None, deep_hash: bool) -> dict[str, Any]:
    if not path_value:
        return {"applicable": False, "ok": True}
    path = Path(path_value)
    row: dict[str, Any] = {
        "applicable": True,
        "path": str(path),
        "exists": path.is_file(),
        "expected_sha256": expected_sha,
    }
    if not path.is_file():
        row["ok"] = False
        return row
    row["bytes"] = path.stat().st_size
    if deep_hash:
        row["actual_sha256"] = file_sha256(path)
        row["ok"] = row["actual_sha256"] == expected_sha
    else:
        row["actual_sha256"] = None
        row["ok"] = expected_sha is not None
    return row


def validate(
    method_names: list[str],
    deep_hash: bool = False,
    suite_levels: list[str] | None = None,
    model_config: str | Path | None = None,
    protocol_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    selected_levels = list(suite_levels or ("L1", "L2"))
    unknown_levels = sorted(set(selected_levels) - {"L1", "L2", "L3"})
    if unknown_levels:
        raise ValueError(f"unknown suite levels: {unknown_levels}")
    model = load_model(model_config)
    selected_model_config = model_config_binding(model_config)
    vlm = load_vlm()
    findings: list[dict[str, Any]] = []
    protocol_lock = validate_protocol_lock(Path(protocol_lock_path) if protocol_lock_path is not None else None)
    findings.append({"kind": "protocol_lock", **protocol_lock})
    if protocol_lock_path is not None:
        if suite_levels is None:
            selected_levels = [row["level"] for row in protocol_lock["levels"]]
        covered_levels = {row["level"] for row in protocol_lock["levels"]}
        findings.append({"kind": "protocol_lock_level_coverage",
                         "ok": set(selected_levels).issubset(covered_levels),
                         "requested_levels": selected_levels,
                         "covered_levels": sorted(covered_levels)})

    snapshot = Path(model["snapshot"])
    weight = snapshot / model["weight_file"]
    resolved = weight.resolve() if weight.exists() else weight
    model_ok = snapshot.is_dir() and weight.is_file()
    if model_ok and not deep_hash:
        model_ok = resolved.name == model["weight_sha256"]
    if model_ok and deep_hash:
        model_ok = file_sha256(weight) == model["weight_sha256"]
    findings.append({
        "kind": "model",
        "ok": model_ok,
        "snapshot": str(snapshot),
        "resolved_weight": str(resolved),
        "revision": model["revision"],
        "model_config": selected_model_config,
    })

    dependencies = load_external_dependencies()
    bridge = dependencies["evalscope_bridge"]
    bridge_root = Path(bridge["root"])
    if not bridge_root.is_absolute():
        bridge_root = ROOT / bridge_root
    bridge_files = []
    for name, expected_sha in bridge["files"].items():
        path = bridge_root / name
        actual_sha = file_sha256(path) if path.is_file() else None
        bridge_files.append({
            "path": str(path),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "ok": actual_sha == expected_sha,
        })
    findings.append({
        "kind": "evalscope_bridge",
        "ok": all(row["ok"] for row in bridge_files),
        "role": bridge["role"],
        "files": bridge_files,
    })

    ifbench = dependencies["ifbench_scorer"]
    ifbench_path = Path(ifbench["repository"])
    if not ifbench_path.is_absolute():
        ifbench_path = ROOT / ifbench_path
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ifbench_path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    findings.append({
        "kind": "ifbench_scorer",
        "ok": revision == ifbench["revision"],
        "repository": str(ifbench_path),
        "expected_revision": ifbench["revision"],
        "actual_revision": revision,
    })

    findings.append({
        "kind": "vlm",
        "ok": (
            vlm.get("image_before_text") is True
            and vlm.get("max_visual_soft_tokens") == 1120
            and vlm.get("score_final_channel_only") is True
            and vlm.get("silent_text_fallback_for_missing_image") is False
        ),
        "interface_id": vlm.get("interface_id"),
    })

    for name in method_names:
        method = load_method(name)
        recon = _path_check(
            method.get("reconstruction_path"), method.get("reconstruction_sha256"), deep_hash
        )
        artifact = _path_check(
            method.get("compressed_artifact_path"),
            method.get("compressed_artifact_sha256"),
            deep_hash,
        )
        findings.append({
            "kind": "method",
            "name": name,
            "method_id": method.get("method_id"),
            "ok": bool(recon["ok"] and artifact["ok"]),
            "reconstruction": recon,
            "compressed_artifact": artifact,
        })
        if method.get("capability_mode") == "dense_reconstruction_injection":
            receipt_path = Path(
                method.get("roundtrip_receipt_path")
                or (ROOT / "receipts" / f"{name}-roundtrip.json")
            )
            receipt = read_json(receipt_path) if receipt_path.is_file() else {}
            expected_receipt_hashes = {
                receipt_key: method[path_key]
                for receipt_key, path_key in (
                    ("artifact_sha256", "compressed_artifact_sha256"),
                    ("reconstruction_sha256", "reconstruction_sha256"),
                    ("builder_sha256", "builder_sha256"),
                    ("online_adapter_sha256", "online_adapter_sha256"),
                    ("roundtrip_verifier_sha256", "roundtrip_verifier_sha256"),
                    ("training_metrics_sha256", "training_metrics_sha256"),
                    ("source_artifact_sha256", "source_artifact_sha256"),
                    ("source_builder_sha256", "source_builder_sha256"),
                    ("source_training_metrics_sha256", "source_training_metrics_sha256"),
                )
                if method.get(path_key) is not None
            }
            ratio = model.get("ple_dense_bytes", 5637144576) / method["ple_resident_bytes"]
            roundtrip = receipt.get("roundtrip", {})
            roundtrip_diff = roundtrip.get("max_abs_diff_after_bfloat16_materialization")
            tolerance = float(method.get("roundtrip_atol", 0.0))
            findings.append({
                "kind": "artifact_roundtrip",
                "name": name,
                "ok": bool(
                    receipt.get("status") == "passed"
                    and receipt.get("actual_hashes") == expected_receipt_hashes
                    and isinstance(roundtrip_diff, (int, float))
                    and roundtrip_diff <= tolerance
                    and roundtrip.get("persistent_resident_bytes") == method["ple_resident_bytes"]
                    and abs(ratio - method["ple_compression_ratio"]) < 1e-6
                ),
                "receipt": str(receipt_path),
                "computed_compression_ratio": ratio,
                "compression_ratio_scope": method.get("compression_ratio_scope"),
            })

    for level_name in selected_levels:
        level = load_level(level_name)
        for benchmark_id in level["benchmarks"]:
            benchmark = load_benchmark(benchmark_id)
            protocol = benchmark.get("protocol", {})
            approved = benchmark.get("approved_for_paired_ple")
            status = benchmark.get("status")
            status_ok = (
                status == "locked"
                if approved is True
                else status in {
                    "pending_validation",
                    "excluded",
                    "locked_protocol_pending_official_alignment",
                }
            )
            no_best_of = protocol.get("best_of") == 1
            attempt_aggregation = protocol.get("attempt_aggregation") != "best"
            request_n = protocol.get("request_n") == 1 or not benchmark.get(
                "approved_for_paired_ple"
            )
            findings.append({
                "kind": "benchmark",
                "benchmark_id": benchmark_id,
                "suite_level": level_name,
                "ok": bool(
                    benchmark.get("suite_level") == level_name
                    and status_ok
                    and isinstance(approved, bool)
                    and no_best_of
                    and attempt_aggregation
                    and request_n
                ),
                "status": status,
                "approved_for_paired_ple": approved,
                "request_n": protocol.get("request_n"),
            })

    return {
        "repository": str(ROOT),
        "deep_hash": deep_hash,
        "model_config": selected_model_config,
        "suite_levels": selected_levels,
        "ok": all(row["ok"] for row in findings),
        "findings": findings,
    }
