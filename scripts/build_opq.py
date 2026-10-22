#!/usr/bin/env python3
"""Build the fixed 100x parametric-OPQ baseline for Gemma-4 PLE.

The PCA-PQ builder provides the resident codec, byte accounting, and exact
BF16 round-trip checks.  This entry point changes only the rotation
construction: PCA eigenvectors are assigned to equal-dimensional PQ subspaces
with the greedy determinant balancing algorithm from parametric OPQ.
Coordinate scales are required by the shared resident schema but must be an
all-ones array, and the baseline uses no token exceptions.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METHOD_BUILDERS = ROOT / "builders" / "methods"
sys.path.insert(0, str(METHOD_BUILDERS))

import dout_weighted_pca_rotated_pq_scaled_exceptions as base  # noqa: E402


METHOD_ID = "parametric_opq"
FORMAT_ID = "parametric_opq_npz"
_allocation_report: dict[str, Any] = {}


def _option(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"{name} is required") from exc


def _balanced_rotation(torch: Any, x_flat: Any, d: int, batch_size: int):
    subspaces = int(_option("--num-subspaces"))
    if d % subspaces:
        raise SystemExit("--num-subspaces must divide --d-ple")
    width = d // subspaces
    rows = int(x_flat.shape[0])
    mean = x_flat.mean(dim=0)
    covariance = torch.zeros((d, d), dtype=torch.float64)
    for start in range(0, rows, batch_size):
        stop = min(rows, start + batch_size)
        chunk = (x_flat[start:stop] - mean).to(torch.float64)
        covariance += chunk.T @ chunk
        del chunk

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    descending = torch.argsort(eigenvalues, descending=True).tolist()
    bins: list[list[int]] = [[] for _ in range(subspaces)]
    log_products = [0.0] * subspaces
    tiny = float(torch.finfo(torch.float64).tiny)
    for eigen_index in descending:
        eligible = [idx for idx, values in enumerate(bins) if len(values) < width]
        target = min(eligible, key=lambda idx: (log_products[idx], len(bins[idx]), idx))
        bins[target].append(int(eigen_index))
        value = max(float(eigenvalues[eigen_index].item()), tiny)
        log_products[target] += math.log(value)

    order = [index for values in bins for index in values]
    if len(order) != d or len(set(order)) != d:
        raise RuntimeError("parametric-OPQ eigenvalue allocation is not a permutation")
    rotation = eigenvectors[:, order].to(torch.float32).contiguous()
    _allocation_report.update(
        {
            "algorithm": "parametric_opq_greedy_eigenvalue_product_balancing",
            "subspaces": subspaces,
            "dimensions_per_subspace": width,
            "eigenvector_indices_by_subspace": bins,
            "log_eigenvalue_product_by_subspace": log_products,
        }
    )
    return mean, rotation


def _validate_fixed_controls() -> None:
    import numpy as np

    if int(_option("--exception-count")) != 0:
        raise SystemExit("the parametric-OPQ baseline forbids token exceptions")
    if float(_option("--coord-weight-alpha")) != 1.0:
        raise SystemExit("the parametric-OPQ baseline requires --coord-weight-alpha 1.0")
    weights = np.load(_option("--coord-weights"), allow_pickle=False)
    if tuple(weights.shape) != (int(_option("--num-layers")), int(_option("--d-ple"))):
        raise SystemExit("identity coordinate-weight geometry is inconsistent")
    if not bool(np.all(weights == 1.0)):
        raise SystemExit("the parametric-OPQ baseline requires exact all-ones coordinate weights")


def _append_method_record() -> None:
    import numpy as np

    output_path = Path(_option("--output"))
    artifact_path = Path(_option("--save-compressed-ple-npz"))
    with np.load(artifact_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
    metadata["rotation_method"] = _allocation_report["algorithm"]
    metadata["eigenvector_indices_by_subspace"] = _allocation_report["eigenvector_indices_by_subspace"]
    metadata["calibration_corpus_used"] = False
    payload = {"metadata": np.array(json.dumps(metadata)), **arrays}
    np.savez(artifact_path, **payload)

    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    metrics["rotation_method"] = _allocation_report
    metrics["calibration_corpus_used"] = False
    metrics["token_exceptions_used"] = False
    metrics["compressed_ple_sha256"] = base.sha256_file(artifact_path)
    metrics["compressed_ple_npz_total_bytes"] = artifact_path.stat().st_size
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    _validate_fixed_controls()
    base.METHOD_ID = METHOD_ID
    base.FORMAT_ID = FORMAT_ID
    base.LOG_PREFIX = "[parametric-opq]"
    base.build_rotation = _balanced_rotation
    base.main()
    _append_method_record()


if __name__ == "__main__":
    main()
