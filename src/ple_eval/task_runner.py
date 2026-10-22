from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from .config import (
    ROOT,
    load_benchmark,
    load_method,
    load_model,
    model_config_binding,
    resolve_model_config,
    write_json_atomic,
)
from .gpu_binding import bind_cuda_visible_device
from .validation import require_protocol_lock_benchmark


# EvalScope client files shipped with this package.
BATCH_ROOT = ROOT / "evaluation_client"
# Local EvalScope dataset cache; set PLE_EVALSCOPE_DATASET_DIR to override.
DATASET_DIR = Path(os.environ.get("PLE_EVALSCOPE_DATASET_DIR", "<CACHE_DIR>/evalscope"))


def method_env(method: dict[str, Any], audit_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_PLUGINS": "ple_eval_reconstruction",
        "PLE_EVAL_PLUGIN_AUDIT_DIR": str(audit_dir),
        "PLE_EVAL_RECON_PATH": method.get("reconstruction_path") or "",
        "PLE_EVAL_RECON_SHA256": method.get("reconstruction_sha256") or "",
        "MODELSCOPE_CACHE": os.environ.get("MODELSCOPE_CACHE", "<CACHE_DIR>/modelscope"),
        "NLTK_DATA": os.environ.get("NLTK_DATA", "<DATA_DIR>/nltk_data"),
    })
    return env


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, prefix.with_suffix(
        ".stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"command failed with {process.returncode}: {command}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_child_binding_receipt_and_stdout(
    receipt_path: Path,
    stdout_path: Path,
) -> dict[str, Any]:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        stdout_receipt = json.loads(stdout_path.read_text(encoding="utf-8").strip())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"child binding receipt/stdout is unreadable: {exc}") from exc
    if not isinstance(receipt, dict) or receipt != stdout_receipt:
        raise RuntimeError("child binding stdout differs from its exclusive receipt")
    return receipt


def validate_output_root_lease(
    lease_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"output-root lease is unreadable: {exc}") from exc
    if not isinstance(lease, dict):
        raise RuntimeError("output-root lease is not a JSON object")
    root = Path(str(lease.get("output_root", "")))
    if (
        lease.get("schema_version") != "ple-output-root-lease-v1"
        or lease.get("status") != "active"
        or lease_path.resolve(strict=True) != lease_path.absolute()
        or root.absolute() != root
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise RuntimeError("output-root lease path identity is invalid")
    root_stat = root.stat()
    if (
        lease.get("root_device") != root_stat.st_dev
        or lease.get("root_inode") != root_stat.st_ino
    ):
        raise RuntimeError("output-root lease inode identity changed")
    try:
        output_dir.absolute().relative_to(root)
    except ValueError as exc:
        raise RuntimeError("output directory escapes its leased root") from exc
    if output_dir == root:
        raise RuntimeError("task output may not equal the leased root")
    current = output_dir.parent
    while current != root:
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir() or current.resolve(strict=True) != current:
                raise RuntimeError(f"output ancestor is not a canonical directory: {current}")
        parent = current.parent
        if parent == current:
            raise RuntimeError("output ancestor walk escaped its leased root")
        current = parent
    return {
        "schema_version": "ple-output-root-lease-attestation-v1",
        "status": "passed",
        "lease_path": str(lease_path.absolute()),
        "lease_sha256": file_sha256(lease_path),
        "output_root": str(root),
        "root_device": root_stat.st_dev,
        "root_inode": root_stat.st_ino,
        "output_directory": str(output_dir.absolute()),
    }


def validate_child_gpu_binding(
    env: dict[str, str],
    device_binding: dict[str, Any],
    output_dir: Path,
    model: dict[str, Any],
) -> dict[str, Any] | None:
    expected_uuid = str(device_binding["expected_physical_gpu_uuid"])
    if not expected_uuid:
        return None
    receipt_path = output_dir / "child-device-binding.json"
    stdout_path = output_dir / "child-device-binding.stdout.log"
    run_logged(
        [
            model["runtime"]["python"],
            str(ROOT / "scripts" / "validate_vllm_child_gpu_binding.py"),
            "--expected-physical-gpu-uuid",
            expected_uuid,
            "--expected-host-ordinal",
            str(device_binding["verified_host_ordinal"]),
            "--output",
            str(receipt_path),
        ],
        cwd=ROOT,
        env=env,
        prefix=output_dir / "child-device-binding",
    )
    receipt = read_child_binding_receipt_and_stdout(receipt_path, stdout_path)
    return {
        "schema_version": "ple-task-child-binding-attestation-v1",
        "status": "passed",
        "child_receipt_path": str(receipt_path.resolve()),
        "child_receipt_sha256": file_sha256(receipt_path),
        "child_stdout_path": str(stdout_path.resolve()),
        "child_stdout_sha256": file_sha256(stdout_path),
        "device_binding_path": str((output_dir / "device-binding.json").resolve()),
        "device_binding_sha256": file_sha256(output_dir / "device-binding.json"),
        "expected_physical_gpu_uuid": expected_uuid,
        "expected_host_ordinal": device_binding["verified_host_ordinal"],
        "child_process_id": receipt.get("process_id"),
    }


def wait_ready(port: int, process: subprocess.Popen[Any], *,
               expected_runtime_identity: dict[str, str] | None = None,
               no_time_limits: bool = False) -> None:
    url = f"http://127.0.0.1:{port}/health"
    attempts = 0
    while no_time_limits or attempts < 240:
        attempts += 1
        if process.poll() is not None:
            raise RuntimeError("Evaluation server exited before health check")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                healthy = response.status == 200
                identity = json.load(response) if healthy and expected_runtime_identity is not None else None
        except (OSError, ValueError):
            time.sleep(2)
            continue
        if healthy:
            if expected_runtime_identity is not None and (
                    not isinstance(identity, dict)
                    or any(identity.get(key) != value for key, value in expected_runtime_identity.items())):
                raise RuntimeError("The listening evaluation server differs from this launch identity")
            if process.poll() is not None:
                raise RuntimeError("Evaluation server exited during its health check")
            return
        time.sleep(2)
    raise RuntimeError("Evaluation server health check timed out")


def start_server(
    benchmark: dict[str, Any],
    env: dict[str, str],
    port: int,
    output_dir: Path,
    model: dict[str, Any],
    model_config_path: Path | None = None,
    *,
    inherit_process_group: bool = False,
    no_time_limits: bool = False,
) -> tuple[subprocess.Popen[Any], dict[str, str]]:
    if type(inherit_process_group) is not bool:
        raise ValueError("Server process-group inheritance must be an explicit boolean.")
    if inherit_process_group and model.get("evaluation_runtime") is None:
        raise ValueError("Inherited server process groups require a bound evaluation runtime.")
    runtime = benchmark["runtime"]
    if model.get("evaluation_runtime") is not None:
        if model_config_path is None:
            raise ValueError("A bound evaluation server requires its exact model configuration path.")
        from .evaluation_runtime import validate_runtime_binding, validate_runtime_interface
        import uuid
        validate_runtime_binding(model["evaluation_runtime"])
        validate_runtime_interface(model["evaluation_runtime"], "chat")
        selected_config = model_config_binding(model_config_path)
        if load_model(model_config_path) != model:
            raise ValueError("The server model configuration differs from the selected task model.")
        identity = {"startup_id": str(uuid.uuid4()), "model_config_sha256": selected_config["sha256"]}
        command = [model["runtime"]["python"], "-m", "ple_eval.evaluation_runtime_server",
                   "--model-config", str(model_config_path), "--port", str(port),
                   "--max-model-len", str(runtime["max_model_len"]), "--seed", "42",
                   "--output-dir", str(output_dir / "evaluation-runtime-server"),
                   "--startup-id", identity['startup_id']]
        stdout = (output_dir / "evaluation-runtime-server.stdout.log").open("w", encoding="utf-8")
        stderr = (output_dir / "evaluation-runtime-server.stderr.log").open("w", encoding="utf-8")
        process = None
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=not inherit_process_group)
            process._ple_logs = (stdout, stderr)
            process._evaluation_server_inherits_process_group = inherit_process_group
            wait_ready(port, process, expected_runtime_identity=identity, no_time_limits=no_time_limits)
            return process, env
        except BaseException:
            if process is not None:
                stop_server(process, termination_grace_seconds=None if no_time_limits else 60.0,
                            kill_wait_seconds=None if no_time_limits else 30.0)
            else:
                stdout.close()
                stderr.close()
            raise
    command = [
        model["runtime"]["vllm"],
        "serve",
        model["snapshot"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model["served_model_name"],
        "--dtype",
        model["dtype"],
        "--load-format",
        str(model.get("load_format", "safetensors")),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        str(runtime["max_model_len"]),
        "--max-num-seqs",
        str(runtime["max_num_seqs"]),
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "gemma4",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--seed",
        "42",
    ]
    if model.get("quantization"):
        command.extend(["--quantization", str(model["quantization"])])
    if runtime["kind"] == "vision":
        hf_config = output_dir / "gemma4-vision-1120-hf-config" / "config.json"
        run_logged(
            [
                model["runtime"]["python"],
                str(BATCH_ROOT / "derive_vision_config.py"),
                str(Path(model["snapshot"]) / "config.json"),
                str(hf_config),
            ],
            cwd=ROOT,
            env=env,
            prefix=output_dir / "derive-vision-config",
        )
        command.extend([
            "--hf-config-path",
            str(hf_config.parent),
            "--limit-mm-per-prompt",
            json.dumps({"image": runtime["max_images"]}, separators=(",", ":")),
            "--mm-processor-kwargs",
            '{"max_soft_tokens":1120}',
        ])
    stdout = (output_dir / "vllm.stdout.log").open("w", encoding="utf-8")
    stderr = (output_dir / "vllm.stderr.log").open("w", encoding="utf-8")
    process = None
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        process._ple_logs = (stdout, stderr)  # type: ignore[attr-defined]
        wait_ready(port, process, no_time_limits=no_time_limits)
        return process, env
    except BaseException:
        if process is not None:
            stop_server(process, termination_grace_seconds=None if no_time_limits else 60.0,
                        kill_wait_seconds=None if no_time_limits else 30.0)
        else:
            stdout.close()
            stderr.close()
        raise


def _active_server_process_group_members(process_group_id: int) -> list[int]:
    """Inspect the Linux group; zombies no longer execute or hold a GPU context."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Evaluation server group cleanup requires Linux process inspection.")
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == process_group_id and fields[0] not in ("Z", "X"):
            members.append(int(entry.name))
    return members


def stop_server(process: subprocess.Popen[Any], *, termination_grace_seconds: float | None = 60.0,
                kill_wait_seconds: float | None = 30.0) -> None:
    """Collect the server, and its group only when this launcher created that group.

    With explicit inheritance the external host owns remaining group members.
    The suite must never signal that shared group, which includes its own runner.
    """
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0)
           for value in (termination_grace_seconds, kill_wait_seconds)):
        raise ValueError("Server cleanup intervals must be positive finite seconds.")
    if getattr(process, "_evaluation_server_cleanup_complete", False):
        return
    try:
        if termination_grace_seconds is None:
            if getattr(process, "_evaluation_server_inherits_process_group", False):
                process.terminate()
                process.wait()
            else:
                if process.pid == os.getpgrp():
                    raise RuntimeError("The suite cannot terminate its own process group.")
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
                while _active_server_process_group_members(process.pid):
                    time.sleep(0.1)
        elif getattr(process, "_evaluation_server_inherits_process_group", False):
            process.terminate()
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=kill_wait_seconds)
        else:
            if process.pid == os.getpgrp():
                raise RuntimeError("The suite cannot terminate its own process group.")
            remaining = _active_server_process_group_members(process.pid)
            for number, wait_seconds in ((signal.SIGTERM, termination_grace_seconds),
                                         (signal.SIGKILL, kill_wait_seconds)):
                if not remaining:
                    break
                try:
                    os.killpg(process.pid, number)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + wait_seconds
                while remaining and time.monotonic() < deadline:
                    time.sleep(min(0.02, max(0, deadline - time.monotonic())))
                    process.poll()
                    remaining = _active_server_process_group_members(process.pid)
            if remaining:
                raise RuntimeError(f"Evaluation server process group remains active: {remaining}")
            process.wait(timeout=kill_wait_seconds)
        process._evaluation_server_cleanup_complete = True
    finally:
        for handle in getattr(process, "_ple_logs", ()):
            handle.close()


def generation_config(protocol: dict[str, Any], seed: int | None = None, *,
                      no_time_limits: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "n": 1,
        "temperature": protocol["temperature"],
        "max_tokens": protocol["max_new_tokens"],
        "seed": protocol.get("seed", 42) if seed is None else seed,
        "stream": False,
        "timeout": None if no_time_limits else 1800,
        "retries": 2,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": protocol["thinking"]}},
    }
    if "top_p" in protocol:
        value["top_p"] = protocol["top_p"]
    if "top_k" in protocol:
        value["top_k"] = protocol["top_k"]
    return value


def evaluation_client_batch_size(benchmark: dict[str, Any], model: dict[str, Any]) -> int:
    """Bound client concurrency without changing the request set."""
    declared = benchmark["evalscope"]["eval_batch_size"]
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise ValueError("The benchmark client batch size must be a positive integer.")
    runtime = model["runtime"]
    field = "evaluation_client_maximum_concurrent_requests"
    if field not in runtime:
        return declared
    maximum = runtime[field]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("The evaluation client concurrency limit must be a positive integer.")
    return min(declared, maximum)


def run_evalscope(
    benchmark: dict[str, Any],
    port: int,
    output_dir: Path,
    env: dict[str, str],
    smoke: bool,
    model: dict[str, Any],
    *,
    no_time_limits: bool = False,
) -> None:
    config = benchmark["evalscope"]
    client_batch_size = evaluation_client_batch_size(benchmark, model)
    seeds = benchmark["protocol"].get("attempt_seeds", [benchmark["protocol"].get("seed", 42)])
    for seed in seeds:
        run_name = f"seed-{seed}" if len(seeds) > 1 else "full"
        work_dir = output_dir / run_name
        command = [
            model["runtime"]["python"],
            str(BATCH_ROOT / "run_evalscope.py"),
            "eval",
            "--model",
            model["served_model_name"],
            "--model-id",
            f"{model['revision'][:12]}-{benchmark['benchmark_id']}-{run_name}",
            "--eval-type",
            "openai_api",
            "--eval-backend",
            "Native",
            "--api-url",
            f"http://127.0.0.1:{port}/v1",
            "--api-key",
            "EMPTY",
            "--datasets",
            config["dataset"],
            "--dataset-hub",
            config["dataset_hub"],
            "--dataset-dir",
            str(DATASET_DIR),
            "--generation-config",
            json.dumps(generation_config(benchmark["protocol"], seed, no_time_limits=no_time_limits), separators=(",", ":")),
            "--eval-batch-size",
            str(client_batch_size),
            "--seed",
            str(seed),
            "--judge-strategy",
            "rule",
            "--work-dir",
            str(work_dir),
            "--no-timestamp",
            "--enable-progress-tracker",
            "--collect-perf",
        ]
        if no_time_limits:
            # EvalScope omits a request timeout when GenerateConfig.timeout is
            # None. Bind None at client construction as well, or the OpenAI
            # SDK silently retains its default ten-minute request timeout.
            command.extend(["--model-args", '{"timeout":null}'])
        if config.get("dataset_args"):
            command.extend(["--dataset-args", json.dumps(config["dataset_args"], separators=(",", ":"))])
        if smoke:
            command.extend(["--limit", "3"])
        eval_env = env.copy()
        eval_env["GEMMA4_BENCHMARK_OVERRIDE"] = config["override"]
        run_logged(command, cwd=BATCH_ROOT, env=eval_env, prefix=output_dir / f"evalscope-{run_name}")

    postprocess = benchmark.get("postprocess")
    if postprocess and postprocess["kind"] == "ifbench_author" and not smoke:
        repo = Path(postprocess["repository"])
        if not repo.is_absolute():
            repo = ROOT / repo
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if revision != postprocess["revision"]:
            raise RuntimeError(f"IFBench scorer revision mismatch: {revision}")
        run_logged(
            [
                model["runtime"]["python"],
                str(BATCH_ROOT / "rescore_ifbench_official_v2.py"),
                "--artifact-dir",
                str(output_dir / "full"),
                "--official-repo",
                str(repo),
                "--output",
                str(output_dir / "author-score.json"),
            ],
            cwd=BATCH_ROOT,
            env=env,
            prefix=output_dir / "ifbench-author-score",
        )
    audit_command = [
        model["runtime"]["python"],
        "-m",
        "ple_eval.evalscope_audit",
        "--benchmark",
        benchmark["benchmark_id"],
        "--output-dir",
        str(output_dir),
    ]
    if smoke:
        audit_command.append("--smoke")
    run_logged(
        audit_command,
        cwd=ROOT,
        env=env,
        prefix=output_dir / "evalscope-integrity",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--expected-physical-gpu-uuid", default="")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-root-lease", type=Path)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--protocol-lock-sha256")
    parser.add_argument("--inherit-server-process-group", action="store_true",
                        help="Keep a bound evaluation server in the external host's process group.")
    parser.add_argument("--no-time-limits", action="store_true",
                        help="Keep evaluation startup and generation requests free of elapsed-time cutoffs.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if bool(args.protocol_lock) != bool(args.protocol_lock_sha256):
        parser.error("--protocol-lock and --protocol-lock-sha256 must be provided together")
    selected_protocol_lock = None
    if args.protocol_lock is not None:
        selected_protocol_lock = require_protocol_lock_benchmark(
            args.protocol_lock, args.benchmark, args.protocol_lock_sha256,
        )

    model_config_path = resolve_model_config(args.model_config or None)
    model = load_model(model_config_path)
    benchmark = load_benchmark(args.benchmark)
    if benchmark["driver"] == "evalscope_api":
        evaluation_client_batch_size(benchmark, model)
    if model.get("evaluation_runtime") is not None:
        from .evaluation_runtime import validate_runtime_binding, validate_runtime_interface
        validate_runtime_binding(model["evaluation_runtime"])
        validate_runtime_interface(model["evaluation_runtime"],
            "chat" if benchmark["driver"] == "evalscope_api" else "direct")
    selected_model_config = model_config_binding(model_config_path)
    method = load_method(args.method)
    if model.get("evaluation_runtime") is not None:
        from .evaluation_runtime import validate_runtime_method
        validate_runtime_method(model["evaluation_runtime"], method)
    output_root_attestation: dict[str, Any] | None = None
    if args.expected_physical_gpu_uuid:
        if args.output_root_lease is None:
            raise RuntimeError("Scheduler-bound task requires an output-root lease")
        output_root_attestation = validate_output_root_lease(
            args.output_root_lease,
            args.output_dir,
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if selected_protocol_lock is not None:
        write_json_atomic(args.output_dir / "protocol-lock-verification.json", selected_protocol_lock)
    if output_root_attestation is not None:
        write_json_atomic(
            args.output_dir / "output-root-lease-attestation.json",
            output_root_attestation,
        )
    write_json_atomic(args.output_dir / "benchmark.json", benchmark)
    write_json_atomic(args.output_dir / "method.json", method)
    write_json_atomic(args.output_dir / "model-config-binding.json", selected_model_config)
    copied_model_config = args.output_dir / "selected-model-config.json"
    shutil.copyfile(model_config_path, copied_model_config)
    if file_sha256(copied_model_config) != selected_model_config["sha256"]:
        raise RuntimeError("copied model config differs from the selected source")
    env = method_env(method, args.output_dir / "plugin-audit")
    if args.no_time_limits:
        env["PLE_EVAL_NO_TIME_LIMITS"] = "1"
        # ZeroMQ treats any negative poll timeout as an indefinite wait.
        env["VLLM_ENGINE_READY_TIMEOUT_S"] = "-1"
    device_binding = bind_cuda_visible_device(
        env,
        logical_gpu=args.gpu,
        expected_physical_gpu_uuid=args.expected_physical_gpu_uuid,
    )
    write_json_atomic(args.output_dir / "device-binding.json", device_binding)
    started = time.time()
    status: dict[str, Any] = {"status": "running", "started_unix": started}
    write_json_atomic(args.output_dir / "status.json", status)
    server = None
    child_binding_attestation: dict[str, Any] | None = None
    try:
        child_binding_attestation = validate_child_gpu_binding(
            env,
            device_binding,
            args.output_dir,
            model,
        )
        if benchmark["driver"] in {"l1_bundle", "gpqa_direct", "mmlu_redux_direct"}:
            module = {
                "l1_bundle": "ple_eval.l1_runner",
                "gpqa_direct": "ple_eval.gpqa_runner",
                "mmlu_redux_direct": "ple_eval.mmlu_redux_runner",
            }[benchmark["driver"]]
            command = [
                model["runtime"]["python"], "-m", module,
                "--model-path", model["snapshot"],
                "--model-config", str(model_config_path),
                "--output-dir", str(args.output_dir / "full"),
            ]
            if args.smoke:
                if benchmark["driver"] == "l1_bundle":
                    command.append("--smoke")
                else:
                    command.extend(["--limit", "3"])
            run_logged(command, cwd=ROOT, env=env, prefix=args.output_dir / benchmark["driver"])
        else:
            server, env = start_server(benchmark, env, args.port, args.output_dir, model,
                                       model_config_path=model_config_path,
                                       inherit_process_group=args.inherit_server_process_group,
                                       no_time_limits=args.no_time_limits)
            if benchmark["driver"] == "evalscope_api":
                run_evalscope(benchmark, args.port, args.output_dir, env, args.smoke, model,
                              no_time_limits=args.no_time_limits)
            else:
                raise RuntimeError(f"unsupported driver: {benchmark['driver']}")
        status = {"status": "completed", "elapsed_seconds": time.time() - started}
    except Exception as exc:
        status = {
            "status": "failed",
            "elapsed_seconds": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise
    finally:
        if server is not None:
            stop_server(server, termination_grace_seconds=None if args.no_time_limits else 60.0,
                        kill_wait_seconds=None if args.no_time_limits else 30.0)
        write_json_atomic(args.output_dir / "status.json", status)
    if child_binding_attestation is not None:
        child_binding_attestation.update({
            "benchmark": args.benchmark,
            "method": args.method,
            "output_directory": str(args.output_dir.resolve()),
        })
        print(json.dumps(child_binding_attestation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
