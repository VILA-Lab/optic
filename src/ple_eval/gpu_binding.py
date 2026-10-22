"""Bind each evaluation process to exactly one GPU selected by UUID."""

from __future__ import annotations

import csv
import io
import re
import subprocess
from typing import Any


_CANONICAL_HOST_ORDINAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_CANONICAL_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


def canonical_physical_gpu_uuid(value: str) -> str:
    if _CANONICAL_GPU_UUID.fullmatch(value) is None:
        raise RuntimeError(f"physical GPU UUID is not canonical: {value!r}")
    return "GPU-" + value[4:].lower()


def physical_gpu_inventory() -> list[dict[str, object]]:
    output = subprocess.check_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: list[dict[str, object]] = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 2:
            raise RuntimeError(f"nvidia-smi returned an invalid GPU inventory row: {row!r}")
        index_text, uuid = (value.strip() for value in row)
        if (
            _CANONICAL_HOST_ORDINAL.fullmatch(index_text) is None
            or _CANONICAL_GPU_UUID.fullmatch(uuid) is None
        ):
            raise RuntimeError(f"nvidia-smi returned an invalid GPU identity row: {row!r}")
        rows.append({"host_ordinal": int(index_text), "physical_gpu_uuid": uuid})
    if not rows:
        raise RuntimeError("nvidia-smi returned an empty GPU inventory")
    if len({int(row["host_ordinal"]) for row in rows}) != len(rows):
        raise RuntimeError("nvidia-smi returned duplicate host GPU ordinals")
    if len({str(row["physical_gpu_uuid"]) for row in rows}) != len(rows):
        raise RuntimeError("nvidia-smi returned duplicate physical GPU UUIDs")
    return rows


def bind_cuda_visible_device(
    env: dict[str, str],
    *,
    logical_gpu: int,
    expected_physical_gpu_uuid: str,
) -> dict[str, Any]:
    """Bind a child process to one GPU UUID, translating it for vLLM 0.19.1.

    The caller selects one physical GPU by UUID.
    vLLM 0.19.1 cannot parse UUID spellings in ``CUDA_VISIBLE_DEVICES`` and
    converts the selected token with ``int()``.  After checking the inherited
    UUID byte-for-byte, resolve that same UUID to one current host ordinal and
    expose only that integer to the child.  A separate child-runtime probe
    verifies the resulting CUDA UUID before any benchmark runs.
    """
    if expected_physical_gpu_uuid:
        if _CANONICAL_GPU_UUID.fullmatch(expected_physical_gpu_uuid) is None:
            raise RuntimeError(
                f"expected Scheduler physical GPU UUID is not canonical: {expected_physical_gpu_uuid!r}"
            )
        inherited = env.get("CUDA_VISIBLE_DEVICES")
        if logical_gpu != 0:
            raise RuntimeError(
                "a Scheduler-bound task must address its sole visible GPU as logical index 0"
            )
        if inherited != expected_physical_gpu_uuid:
            raise RuntimeError(
                "Scheduler GPU containment mismatch: expected the byte-exact single UUID "
                f"{expected_physical_gpu_uuid!r}, received {inherited!r}"
            )
        inventory = physical_gpu_inventory()
        matches = [
            row for row in inventory
            if row["physical_gpu_uuid"] == expected_physical_gpu_uuid
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "expected one current nvidia-smi mapping for Scheduler UUID "
                f"{expected_physical_gpu_uuid!r}, received {matches!r}"
            )
        host_ordinal = int(matches[0]["host_ordinal"])
        env["CUDA_VISIBLE_DEVICES"] = str(host_ordinal)
        return {
            "binding_mode": "scheduler_physical_uuid_to_verified_vllm_host_ordinal",
            "child_cuda_visible_devices": str(host_ordinal),
            "expected_physical_gpu_uuid": expected_physical_gpu_uuid,
            "scheduler_inherited_cuda_visible_devices": inherited,
            "logical_gpu": logical_gpu,
            "verified_host_ordinal": host_ordinal,
            "verified_physical_gpu_uuid": expected_physical_gpu_uuid,
        }

    env["CUDA_VISIBLE_DEVICES"] = str(logical_gpu)
    return {
        "binding_mode": "host_ordinal",
        "child_cuda_visible_devices": str(logical_gpu),
        "expected_physical_gpu_uuid": "",
        "scheduler_inherited_cuda_visible_devices": "",
        "logical_gpu": logical_gpu,
        "verified_host_ordinal": logical_gpu,
        "verified_physical_gpu_uuid": "",
    }


def validate_torch_cuda_binding(torch: Any, binding: dict[str, Any]) -> dict[str, Any]:
    """Reverse-check logical cuda:0 against a prior GPU UUID binding."""

    expected = canonical_physical_gpu_uuid(
        str(binding.get("expected_physical_gpu_uuid", ""))
    )
    if binding.get("binding_mode") != "scheduler_physical_uuid_to_verified_vllm_host_ordinal":
        raise RuntimeError("Scheduler physical GPU binding mode differs")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Scheduler physical binding must expose exactly one CUDA device")
    observed = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if observed and not observed.startswith("GPU-"):
        observed = "GPU-" + observed
    observed = canonical_physical_gpu_uuid(observed)
    if observed != expected:
        raise RuntimeError(
            f"logical cuda:0 UUID {observed!r} differs from Scheduler UUID {expected!r}"
        )
    receipt = {
        "schema_version": "ple-scheduler-torch-reverse-gpu-binding-v1",
        "status": "passed",
        "device_binding": dict(binding),
        "logical_cuda_device": "cuda:0",
        "runtime_visible_gpu_uuid": observed,
        "runtime_device_name": str(torch.cuda.get_device_name(0)),
        "runtime_visible_device_count": int(torch.cuda.device_count()),
        "reverse_uuid_verified": True,
    }
    validate_reverse_gpu_attestation(receipt)
    return receipt


def validate_reverse_gpu_attestation(value: Any) -> str:
    """Validate the portable reverse-UUID record embedded in stage receipts."""

    if not isinstance(value, dict):
        raise RuntimeError("GPU binding attestation is not an object")
    binding = value.get("device_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("GPU binding attestation has no device binding")
    expected = canonical_physical_gpu_uuid(
        str(binding.get("expected_physical_gpu_uuid", ""))
    )
    observed = canonical_physical_gpu_uuid(
        str(value.get("runtime_visible_gpu_uuid", ""))
    )
    if not (
        value.get("schema_version") == "ple-scheduler-torch-reverse-gpu-binding-v1"
        and value.get("status") == "passed"
        and value.get("reverse_uuid_verified") is True
        and value.get("logical_cuda_device") == "cuda:0"
        and value.get("runtime_visible_device_count") == 1
        and binding.get("binding_mode")
        == "scheduler_physical_uuid_to_verified_vllm_host_ordinal"
        and binding.get("scheduler_inherited_cuda_visible_devices") == expected
        and binding.get("verified_physical_gpu_uuid") == expected
        and binding.get("logical_gpu") == 0
        and observed == expected
    ):
        raise RuntimeError("reverse physical GPU UUID attestation differs")
    return expected


__all__ = [
    "bind_cuda_visible_device",
    "canonical_physical_gpu_uuid",
    "physical_gpu_inventory",
    "validate_reverse_gpu_attestation",
    "validate_torch_cuda_binding",
]
