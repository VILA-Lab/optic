from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path


def write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_source_identity(module: object) -> dict[str, object]:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"loaded module has no source path: {module!r}")
    path = Path(module_file).resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"loaded module source is not a regular file: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def validate_child_cuda_visible_devices(
    inherited: str | None,
    expected_host_ordinal: int,
) -> str:
    if expected_host_ordinal < 0:
        raise RuntimeError("vLLM child expected host ordinal must be nonnegative")
    expected = str(expected_host_ordinal)
    if inherited != expected:
        raise RuntimeError(
            "vLLM child expected the byte-exact verified host ordinal "
            f"{expected!r}, received {inherited!r}"
        )
    return inherited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--expected-host-ordinal", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inherited = validate_child_cuda_visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        args.expected_host_ordinal,
    )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("vLLM child CUDA runtime must expose exactly one device")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    runtime_uuid = str(properties.uuid)
    if not runtime_uuid.startswith("GPU-"):
        runtime_uuid = "GPU-" + runtime_uuid
    if runtime_uuid != args.expected_physical_gpu_uuid:
        raise RuntimeError(
            f"vLLM child CUDA UUID {runtime_uuid!r} differs from expected "
            f"{args.expected_physical_gpu_uuid!r}"
        )

    # Import the exact vLLM path that failed when it attempted int(UUID).
    from vllm.model_executor.layers.quantization.utils import w8a8_utils
    from vllm.platforms import current_platform
    from vllm.platforms import cuda as vllm_platform_cuda
    from vllm.platforms import interface as vllm_platform_interface

    capability = current_platform.get_device_capability(0)
    if capability is None:
        raise RuntimeError("vLLM did not report a CUDA device capability")

    allocation = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    allocation.fill_(1)
    torch.cuda.synchronize()
    process_csv = subprocess.check_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    process_rows = [
        {
            "gpu_uuid": row[0].strip(),
            "process_id": int(row[1].strip()),
            "process_name": row[2].strip(),
            "used_memory_mib": int(row[3].strip()),
        }
        for row in csv.reader(io.StringIO(process_csv))
        if len(row) == 4 and row[1].strip().isdigit()
    ]
    own_rows = [row for row in process_rows if row["process_id"] == os.getpid()]
    if len(own_rows) != 1 or own_rows[0]["gpu_uuid"] != args.expected_physical_gpu_uuid:
        raise RuntimeError(
            "vLLM child nvidia-smi telemetry did not bind this process uniquely "
            f"to {args.expected_physical_gpu_uuid}: {own_rows}"
        )

    import vllm

    receipt: dict[str, object] = {
        "schema_version": "ple-vllm-child-gpu-binding-preflight-v1",
        "status": "passed",
        "expected_physical_gpu_uuid": args.expected_physical_gpu_uuid,
        "expected_host_ordinal": args.expected_host_ordinal,
        "inherited_cuda_visible_devices": inherited,
        "cuda_visible_device_count": torch.cuda.device_count(),
        "runtime_visible_gpu_uuid": runtime_uuid,
        "runtime_device_name": properties.name,
        "vllm_version": vllm.__version__,
        "vllm_device_capability": str(capability),
        "vllm_cutlass_fp8_supported": bool(w8a8_utils.CUTLASS_FP8_SUPPORTED),
        "loaded_vllm_sources": {
            "vllm.model_executor.layers.quantization.utils.w8a8_utils": loaded_source_identity(
                w8a8_utils
            ),
            "vllm.platforms.cuda": loaded_source_identity(vllm_platform_cuda),
            "vllm.platforms.interface": loaded_source_identity(vllm_platform_interface),
        },
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "nvidia_smi_own_process_rows": own_rows,
        "allocated_tensor_bytes": allocation.numel() * allocation.element_size(),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_exclusive(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
