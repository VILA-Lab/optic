#!/usr/bin/env python3
"""Build an Additive Quantization (AQ) baseline for Gemma-4 PLE.

The optimization adapts the core loop of Babenko and Lempitsky's
implementation to PLE: beam-search assignment followed by one global sparse
least-squares codebook update.  Greedy residual k-means initializes the loop.

The final artifact is written when the configured stopping rule is met.  The
supported rules are a convergence rule and the fixed outer-sweep schedule of
the original implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import os
import tempfile

# Directory for large intermediate files.
SCRATCH_DIR = os.path.realpath(os.environ.get("PLE_SCRATCH_DIR") or tempfile.gettempdir())

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ONLINE_ADAPTER_PATH = ROOT / "scripts" / "aq_codec.py"

from aq_codec import (  # noqa: E402
    FORMAT_ID,
    PLE_KEY,
    RECON_KEY,
    artifact_sha256,
    audit_requested_rows,
    expected_resident_bytes,
    pack_nbit,
)


AUTHOR_REPOSITORY = "https://github.com/arbabenko/Quantizations"
AUTHOR_COMMIT = "149e8d5226b96c36b55ef2423bcdebc94aec4099"
AUTHOR_FILE = "aqCoding.py"
PRIMARY_PAPER = "Babenko and Lempitsky, Additive Quantization for Extreme Vector Compression, CVPR 2014"
METHOD_ID = "full_additive_quantization_ple"
ACCEPTED_LSMR_STOPS = {0, 1, 2, 4, 5}


class DeadlineSafeStop(RuntimeError):
    """Stop at a checked boundary without publishing a final artifact."""


class SchedulerDeadline:
    def __init__(self, seconds: float | None, reserve_seconds: float = 120.0) -> None:
        self.started = time.monotonic()
        self.safe_deadline = (
            None
            if seconds is None
            else self.started + max(float(seconds) - float(reserve_seconds), 0.0)
        )
        self.signal_name: str | None = None
        self._previous_handlers: dict[int, Any] = {}

    @classmethod
    def from_environment(cls) -> "SchedulerDeadline":
        raw = os.environ.get("PLE_SCHEDULER_WALL_CLOCK_DEADLINE_SECONDS")
        seconds = None if raw is None or not raw.strip() else float(raw)
        require(seconds is None or seconds > 120, "Scheduler deadline must exceed the shutdown reserve")
        return cls(seconds)

    def install_signal_handlers(self) -> None:
        def request_stop(signum: int, _frame: Any) -> None:
            self.signal_name = signal.Signals(signum).name

        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    def restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def check(self, phase: str) -> None:
        if self.signal_name is not None:
            raise DeadlineSafeStop(
                f"received {self.signal_name} during {phase}; stopping without a final artifact"
            )
        if self.safe_deadline is not None and time.monotonic() >= self.safe_deadline:
            raise DeadlineSafeStop(
                f"Scheduler wall-clock reserve reached during {phase}; stopping without a final artifact"
            )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def method_identity(contract: dict[str, Any]) -> dict[str, str]:
    configured = contract["config"].get("method_identity", {})
    require(isinstance(configured, dict), "method_identity must be an object")
    return {
        "method_id_prefix": str(configured.get("method_id_prefix", METHOD_ID)),
        "method_config_template": str(
            configured.get(
                "method_config_template",
                "configs/methods/full_additive_quantization_ple_{model}_{ratio_key}.json",
            )
        ),
        "publication_name": str(
            configured.get(
                "publication_name",
                "Full Additive Quantization (AQ) adapted to per-layer embeddings",
            )
        ),
    }


def method_config_path(contract: dict[str, Any]) -> Path:
    template = method_identity(contract)["method_config_template"]
    rendered = template.format(model=contract["model_key"], ratio_key=contract["ratio_key"])
    path = Path(rendered)
    return path if path.is_absolute() else ROOT / path


def completion_mode(convergence: dict[str, Any]) -> str:
    return str(convergence.get("mode", "objective_and_assignment_convergence"))


def completion_status(convergence: dict[str, Any]) -> str:
    return "completed_fixed_schedule" if completion_mode(convergence) == "fixed_outer_sweeps" else "converged"


def maximum_sweeps(convergence: dict[str, Any]) -> int:
    if completion_mode(convergence) == "fixed_outer_sweeps":
        return int(convergence["fixed_outer_sweeps"])
    return int(convergence["max_sweeps"])


def fit_completion_contract(fit: dict[str, Any]) -> dict[str, Any]:
    configured = fit.get("completion", fit.get("convergence", {}))
    require(isinstance(configured, dict), "AQ training completion contract must be an object")
    return configured


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _candidate_unique_topk(torch: Any, candidate_codes: Any, candidate_loss: Any, beam_size: int) -> Any:
    """Indices of the first score-ordered unique AQ tuples for every row.

    The same partial AQ tuple can be generated by at most ``depth`` parent
    orderings.  The caller therefore supplies the best ``beam*depth`` raw
    candidates, which is sufficient to recover ``beam`` unique tuples.
    """
    batch, candidates, _ = candidate_codes.shape
    selected_codes = torch.full(
        (batch, beam_size, candidate_codes.shape[-1]),
        -2,
        device=candidate_codes.device,
        dtype=candidate_codes.dtype,
    )
    selected_indices = torch.zeros((batch, beam_size), device=candidate_codes.device, dtype=torch.long)
    selected_valid = torch.zeros((batch, beam_size), device=candidate_codes.device, dtype=torch.bool)
    counts = torch.zeros(batch, device=candidate_codes.device, dtype=torch.long)
    rows = torch.arange(batch, device=candidate_codes.device)
    for rank in range(candidates):
        current = candidate_codes[:, rank]
        duplicate = ((selected_codes == current[:, None, :]).all(dim=-1) & selected_valid).any(dim=1)
        take = (~duplicate) & (counts < beam_size) & torch.isfinite(candidate_loss[:, rank])
        if bool(take.any()):
            take_rows = rows[take]
            slots = counts[take]
            selected_codes[take_rows, slots] = current[take]
            selected_indices[take_rows, slots] = rank
            selected_valid[take_rows, slots] = True
            counts[take] += 1
        if bool((counts == beam_size).all()):
            break
    if not bool((counts == beam_size).all()):
        raise RuntimeError("beam expansion did not contain enough unique AQ assignments")
    return selected_indices


def _beam_encode_chunk(torch: Any, x: Any, codebooks: Any, beam_size: int) -> tuple[Any, Any]:
    """Author-style AQ beam assignment for a device-resident row chunk."""
    batch, d_ple = x.shape
    m_count, clusters, codebook_d = codebooks.shape
    require(d_ple == codebook_d, "beam input/codebook width differs")
    beam = min(beam_size, m_count * clusters)
    codebook_norm = codebooks.square().sum(dim=-1)
    singleton = x.square().sum(dim=-1)[:, None, None] + codebook_norm[None, :, :]
    singleton = singleton - 2.0 * torch.einsum("bd,mkd->bmk", x, codebooks)
    losses, flat = singleton.reshape(batch, -1).topk(beam, dim=1, largest=False, sorted=True)
    books = torch.div(flat, clusters, rounding_mode="floor")
    words = flat.remainder(clusters)
    codes = torch.full((batch, beam, m_count), -1, device=x.device, dtype=torch.int16)
    codes.scatter_(2, books[..., None], words.to(torch.int16)[..., None])
    reconstruction = codebooks[books, words]

    for depth in range(2, m_count + 1):
        target_beam = min(beam_size, math.comb(m_count, depth) * (clusters**depth))
        residual = x[:, None, :] - reconstruction
        expansion = losses[:, :, None, None] + codebook_norm[None, None, :, :]
        expansion = expansion - 2.0 * torch.einsum("bhd,mkd->bhmk", residual, codebooks)
        expansion.masked_fill_((codes >= 0)[:, :, :, None], float("inf"))
        raw_count = min(expansion.shape[1] * m_count * clusters, target_beam * depth)
        raw_loss, raw_flat = expansion.reshape(batch, -1).topk(
            raw_count, dim=1, largest=False, sorted=True
        )
        parent = torch.div(raw_flat, m_count * clusters, rounding_mode="floor")
        remainder = raw_flat.remainder(m_count * clusters)
        new_books = torch.div(remainder, clusters, rounding_mode="floor")
        new_words = remainder.remainder(clusters)
        candidate_codes = codes.gather(1, parent[..., None].expand(-1, -1, m_count)).clone()
        candidate_codes.scatter_(2, new_books[..., None], new_words.to(torch.int16)[..., None])
        chosen = _candidate_unique_topk(torch, candidate_codes, raw_loss, target_beam)
        codes = candidate_codes.gather(1, chosen[..., None].expand(-1, -1, m_count))
        losses = raw_loss.gather(1, chosen)
        chosen_parent = parent.gather(1, chosen)
        chosen_books = new_books.gather(1, chosen)
        chosen_words = new_words.gather(1, chosen)
        parent_reconstruction = reconstruction.gather(
            1, chosen_parent[..., None].expand(-1, -1, d_ple)
        )
        reconstruction = parent_reconstruction + codebooks[chosen_books, chosen_words]
        beam = target_beam
    exact_loss = (x - reconstruction[:, 0]).square().sum(dim=-1)
    return codes[:, 0].to(torch.int64), exact_loss


def beam_encode(
    x: Any,
    codebooks: Any,
    *,
    beam_size: int,
    row_chunk: int,
    device: str | Any,
    deadline_check: Callable[[str], None] | None = None,
) -> tuple[Any, float]:
    """Encode all CPU rows; return CPU int64 codes and deployed-domain SSE."""
    import torch

    require(x.ndim == 2 and codebooks.ndim == 3, "beam geometry must be x[N,D], codebooks[M,K,D]")
    target_device = torch.device(device)
    resident = codebooks.to(device=target_device, dtype=torch.float32)
    codes = torch.empty((x.shape[0], codebooks.shape[0]), dtype=torch.int64)
    objective = 0.0
    with torch.inference_mode():
        for start in range(0, x.shape[0], row_chunk):
            if deadline_check is not None:
                deadline_check("beam_assignment")
            stop = min(x.shape[0], start + row_chunk)
            chunk_codes, chunk_loss = _beam_encode_chunk(
                torch, x[start:stop].to(device=target_device, dtype=torch.float32), resident, beam_size
            )
            codes[start:stop] = chunk_codes.cpu()
            objective += float(chunk_loss.double().sum().item())
    return codes, objective


def decode_codes(
    codes: Any,
    codebooks: Any,
    *,
    row_chunk: int,
    device: str | Any,
) -> Any:
    import torch

    target_device = torch.device(device)
    resident = codebooks.to(device=target_device, dtype=torch.float32)
    output = torch.empty((codes.shape[0], codebooks.shape[2]), dtype=torch.bfloat16)
    with torch.inference_mode():
        for start in range(0, codes.shape[0], row_chunk):
            stop = min(codes.shape[0], start + row_chunk)
            chosen = codes[start:stop].to(target_device)
            decoded = torch.zeros((stop - start, codebooks.shape[2]), device=target_device, dtype=torch.float32)
            for stage in range(codebooks.shape[0]):
                decoded.add_(resident[stage].index_select(0, chosen[:, stage]))
            output[start:stop] = decoded.to(torch.bfloat16).cpu()
    return output


def _kmeans(
    torch: Any,
    x: Any,
    clusters: int,
    iterations: int,
    row_chunk: int,
    device: Any,
    seed: int,
    deadline_check: Callable[[str], None] | None = None,
) -> tuple[Any, Any]:
    require(x.shape[0] >= clusters, "initialization subset has fewer rows than K")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(x.shape[0], generator=generator)[:clusters]
    centers = x.index_select(0, initial).to(device=device, dtype=torch.float32)
    codes = torch.empty(x.shape[0], dtype=torch.int64)
    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(clusters, device=device, dtype=torch.int64)
        center_norm = centers.square().sum(dim=1)[None, :]
        for start in range(0, x.shape[0], row_chunk):
            if deadline_check is not None:
                deadline_check("residual_kmeans_initialization")
            stop = min(x.shape[0], start + row_chunk)
            rows = x[start:stop].to(device=device, dtype=torch.float32)
            distances = rows.square().sum(dim=1, keepdim=True) + center_norm - 2.0 * rows @ centers.T
            assigned = distances.argmin(dim=1)
            codes[start:stop] = assigned.cpu()
            sums.index_add_(0, assigned, rows)
            counts.index_add_(0, assigned, torch.ones_like(assigned, dtype=torch.int64))
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        # Empty clusters retain their deterministic initial/previous center.
        centers = centers.to(torch.float16).to(torch.float32)
    return centers.cpu(), codes


def initialize_rvq(
    x: Any,
    *,
    m_count: int,
    clusters: int,
    iterations: int,
    row_chunk: int,
    device: str | Any,
    seed: int,
    deadline_check: Callable[[str], None] | None = None,
) -> tuple[Any, Any]:
    """Deterministic greedy residual k-means used strictly as AQ initialization."""
    import torch

    residual = x.to(dtype=torch.float32).clone()
    codebooks: list[Any] = []
    codes: list[Any] = []
    target_device = torch.device(device)
    for stage in range(m_count):
        centers, assigned = _kmeans(
            torch, residual, clusters, iterations, row_chunk, target_device, seed + stage,
            deadline_check,
        )
        codebooks.append(centers)
        codes.append(assigned)
        residual.sub_(centers.index_select(0, assigned))
    return torch.stack(codebooks).to(torch.float16), torch.stack(codes, dim=1)


def assignment_matrix(codes: Any, clusters: int) -> Any:
    import numpy as np
    from scipy import sparse

    values = codes.detach().cpu().numpy().astype(np.int64, copy=False)
    rows, m_count = values.shape
    row_index = np.repeat(np.arange(rows, dtype=np.int64), m_count)
    columns = (values + np.arange(m_count, dtype=np.int64)[None, :] * clusters).reshape(-1)
    data = np.ones(rows * m_count, dtype=np.float64)
    return sparse.csr_matrix((data, (row_index, columns)), shape=(rows, m_count * clusters))


def global_coordinate_lsmr_codebook_update_reference(
    x: Any,
    codes: Any,
    previous_codebooks: Any,
    *,
    clusters: int,
    atol: float,
    btol: float,
    conlim: float,
    maxiter: int,
) -> tuple[Any, dict[str, Any]]:
    """Reference coordinate-wise SciPy LSMR codebook update."""
    import numpy as np
    import torch
    from scipy.sparse.linalg import lsmr

    design = assignment_matrix(codes, clusters)
    target = x.detach().cpu().numpy().astype(np.float64, copy=False)
    coefficient_shape = (previous_codebooks.shape[0] * previous_codebooks.shape[1], x.shape[1])
    solved = np.empty(coefficient_shape, dtype=np.float64)
    istop: list[int] = []
    iterations: list[int] = []
    normr: list[float] = []
    normar: list[float] = []
    for coordinate in range(x.shape[1]):
        result = lsmr(
            design,
            target[:, coordinate],
            atol=atol,
            btol=btol,
            conlim=conlim,
            maxiter=maxiter,
        )
        solved[:, coordinate] = result[0]
        istop.append(int(result[1]))
        iterations.append(int(result[2]))
        normr.append(float(result[3]))
        normar.append(float(result[4]))
    deployed = torch.from_numpy(solved.reshape(previous_codebooks.shape)).to(torch.float16)
    residual_l2 = math.sqrt(sum(value * value for value in normr))
    normal_residual_l2 = math.sqrt(sum(value * value for value in normar))
    target_l2 = float(torch.linalg.vector_norm(x.double()).item())
    diagnostics = {
        "accepted": all(value in ACCEPTED_LSMR_STOPS for value in istop),
        "accepted_stop_codes": sorted(ACCEPTED_LSMR_STOPS),
        "istop_by_coordinate": istop,
        "iterations_by_coordinate": iterations,
        "normr_by_coordinate": normr,
        "normar_by_coordinate": normar,
        "residual_l2": residual_l2,
        "relative_residual_l2": residual_l2 / max(target_l2, 1e-30),
        "normal_equation_residual_l2": normal_residual_l2,
        "iteration_min": min(iterations),
        "iteration_mean": sum(iterations) / len(iterations),
        "iteration_max": max(iterations),
    }
    return deployed, diagnostics


def global_multi_rhs_codebook_update(
    x: Any,
    codes: Any,
    previous_codebooks: Any,
    *,
    clusters: int,
    atol: float,
    btol: float,
    conlim: float,
    maxiter: int,
    deadline_check: Callable[[str], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Solve the same unregularized AQ least-squares problem for all RHS.

    The coordinate-wise reference calls SciPy LSMR independently for each of
    the 256 output coordinates.  Every call uses the same assignment matrix
    ``A``, omits ``damp`` (therefore ``damp=0``), supplies no bounds, and uses
    the default zero initial state.  Its mathematical problem is consequently

        min_C ||A C - X||_F,

    with no ridge term or inequality constraint.  AQ has at least M-1 exact
    gauge null directions, so this routine does not invert ``A.T @ A``.  It
    computes the float64 spectrum, removes only numerically null directions,
    checks the retained spectrum against the configured LSMR ``conlim``, and
    applies the Moore-Penrose spectral solution to every RHS together.
    """
    import numpy as np
    import torch
    from scipy import linalg

    require(x.ndim == 2 and codes.ndim == 2, "multi-RHS AQ geometry differs")
    require(codes.shape[0] == x.shape[0], "AQ codes and targets have different rows")
    require(tuple(previous_codebooks.shape[1:]) == (clusters, x.shape[1]), "AQ codebook geometry differs")
    require(atol >= 0 and btol >= 0 and conlim > 1 and maxiter > 0, "LSMR contract is invalid")
    if deadline_check is not None:
        deadline_check("multi_rhs_sufficient_statistics")
    design = assignment_matrix(codes, clusters)
    target = x.detach().cpu().numpy().astype(np.float64, copy=False)
    gram = (design.T @ design).toarray().astype(np.float64, copy=False)
    right_hand_sides = np.asarray(design.T @ target, dtype=np.float64)
    require(bool(np.isfinite(gram).all() and np.isfinite(right_hand_sides).all()), "AQ sufficient statistics are non-finite")
    if deadline_check is not None:
        deadline_check("multi_rhs_spectral_factorization")
    eigenvalues, eigenvectors = linalg.eigh(
        gram,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    maximum_eigenvalue = float(eigenvalues[-1])
    require(math.isfinite(maximum_eigenvalue) and maximum_eigenvalue > 0, "AQ assignment matrix has zero rank")
    numeric_floor = maximum_eigenvalue * gram.shape[0] * np.finfo(np.float64).eps
    # Keep every numerically nonzero direction in the least-squares solution.
    rank_threshold = numeric_floor
    require(float(eigenvalues[0]) >= -numeric_floor * 8.0, "AQ Gram spectrum is numerically indefinite")
    retained = eigenvalues > rank_threshold
    retained_rank = int(retained.sum())
    require(retained_rank > 0, "AQ assignment matrix has no numerically stable subspace")
    smallest_retained = float(eigenvalues[retained][0])
    condition_estimate = math.sqrt(maximum_eigenvalue / smallest_retained)
    require(
        math.isfinite(condition_estimate) and condition_estimate <= conlim,
        "AQ assignment matrix exceeds the configured LSMR condition limit",
    )
    retained_vectors = eigenvectors[:, retained]
    projected_rhs = retained_vectors.T @ right_hand_sides
    solved = retained_vectors @ (projected_rhs / eigenvalues[retained, None])
    if deadline_check is not None:
        deadline_check("multi_rhs_residual_validation")
    residual = np.asarray(design @ solved - target, dtype=np.float64)
    normal_residual = np.asarray(design.T @ residual, dtype=np.float64)
    normr = np.linalg.norm(residual, axis=0)
    normar = np.linalg.norm(normal_residual, axis=0)
    rhs_norm = float(np.linalg.norm(right_hand_sides))
    normal_residual_l2 = float(np.linalg.norm(normal_residual))
    relative_normal_residual = normal_residual_l2 / max(rhs_norm, 1e-30)
    acceptance_tolerance = max(float(atol), float(btol), np.finfo(np.float64).eps) * 16.0
    accepted = bool(
        np.isfinite(solved).all()
        and np.isfinite(normr).all()
        and np.isfinite(normar).all()
        and relative_normal_residual <= acceptance_tolerance
    )
    deployed = torch.from_numpy(
        solved.reshape(previous_codebooks.shape)
    ).to(torch.float16)
    residual_l2 = float(np.linalg.norm(residual))
    target_l2 = float(np.linalg.norm(target))
    diagnostics = {
        "accepted": accepted,
        "solver": "float64_multi_rhs_moore_penrose_spectral_solve",
        "mathematical_problem": "min_C ||A C - X||_F",
        "reference_lsmr_damp": 0.0,
        "reference_lsmr_regularization": False,
        "reference_lsmr_bounds": None,
        "reference_lsmr_initial_state": "zero",
        "reference_lsmr_atol": float(atol),
        "reference_lsmr_btol": float(btol),
        "reference_lsmr_conlim": float(conlim),
        "reference_lsmr_maxiter": int(maxiter),
        "coefficient_count": int(gram.shape[0]),
        "retained_rank": retained_rank,
        "numerical_nullity": int(gram.shape[0] - retained_rank),
        "rank_threshold_eigenvalue": rank_threshold,
        "smallest_retained_eigenvalue": smallest_retained,
        "largest_eigenvalue": maximum_eigenvalue,
        "assignment_matrix_condition_estimate": condition_estimate,
        "normal_equation_relative_residual": relative_normal_residual,
        "normal_equation_acceptance_tolerance": acceptance_tolerance,
        "normr_by_coordinate": normr.tolist(),
        "normar_by_coordinate": normar.tolist(),
        "residual_l2": residual_l2,
        "relative_residual_l2": residual_l2 / max(target_l2, 1e-30),
        "normal_equation_residual_l2": normal_residual_l2,
    }
    return deployed, diagnostics


def fit_full_aq(
    x: Any,
    *,
    m_count: int,
    clusters: int,
    beam_size: int,
    row_chunk: int,
    device: str | Any,
    seed: int,
    initialization_iterations: int,
    convergence: dict[str, Any],
    lsmr_config: dict[str, Any],
    initial_codebooks: Any | None = None,
    resume_state: dict[str, Any] | None = None,
    deadline_check: Callable[[str], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit AQ until its convergence or fixed-schedule boundary."""
    import torch

    require(x.ndim == 2 and x.shape[0] >= clusters, "fit matrix geometry is invalid")
    denominator = float(x.double().square().sum().item())
    mode = completion_mode(convergence)
    fixed_schedule = mode == "fixed_outer_sweeps"
    require(
        fixed_schedule or mode == "objective_and_assignment_convergence",
        f"unsupported AQ completion mode: {mode}",
    )
    max_sweeps = maximum_sweeps(convergence)
    min_sweeps = 0 if fixed_schedule else int(convergence["min_sweeps"])
    patience = 0 if fixed_schedule else int(convergence["patience"])
    objective_tolerance = None if fixed_schedule else float(convergence["relative_improvement_tolerance"])
    churn_tolerance = None if fixed_schedule else float(convergence["assignment_churn_tolerance"])
    monotonicity_tolerance = None if fixed_schedule else float(convergence["monotonicity_tolerance"])
    terminal_status = completion_status(convergence)
    invocation_started = time.monotonic()

    def checkpoint(
        stage: str,
        *,
        codebooks: Any,
        codes: Any,
        trajectory: list[dict[str, Any]],
        stable_counter: int,
        next_sweep: int,
        training_status: str,
        status_reason: str,
        elapsed_before_invocation: float,
    ) -> None:
        if checkpoint_callback is not None:
            checkpoint_callback({
                "stage": stage,
                "codebooks": codebooks,
                "codes": codes,
                "trajectory": trajectory,
                "stable_counter": stable_counter,
                "next_sweep": next_sweep,
                "training_status": training_status,
                "status_reason": status_reason,
                "fit_elapsed_seconds": elapsed_before_invocation + time.monotonic() - invocation_started,
            })

    if resume_state is not None:
        require(initial_codebooks is None, "resume and explicit initialization are mutually exclusive")
        codebooks = resume_state["codebooks"].detach().cpu().to(torch.float16)
        codes = resume_state["codes"].detach().cpu().to(torch.long)
        trajectory = list(resume_state["trajectory"])
        stable = int(resume_state["stable_counter"])
        next_sweep = int(resume_state["next_sweep"])
        status = str(resume_state["training_status"])
        reason = str(resume_state["status_reason"])
        elapsed_before_invocation = float(resume_state["fit_elapsed_seconds"])
        require(tuple(codebooks.shape) == (m_count, clusters, x.shape[1]), "resumed AQ codebook geometry differs")
        require(tuple(codes.shape) == (x.shape[0], m_count), "resumed AQ code geometry differs")
        require(bool(trajectory) and int(trajectory[-1]["sweep"]) == next_sweep - 1, "resumed AQ trajectory boundary differs")
        objective = float(trajectory[-1]["objective_sse"])
        print(
            f"[full-aq] resumed after sweep {next_sweep - 1}; "
            f"next_sweep={next_sweep} stable_counter={stable} "
            f"objective_nmse={objective / max(denominator, 1e-30):.8e}",
            flush=True,
        )
        if status == terminal_status:
            return {
                "training_status": status,
                "status_reason": reason,
                "codebooks": codebooks,
                "codes": codes,
                "objective_sse": objective,
                "objective_nmse": objective / max(denominator, 1e-30),
                "trajectory": trajectory,
                "completed_sweeps": len(trajectory) - 1,
                "fit_elapsed_seconds": elapsed_before_invocation,
            }
        require(status == "in_progress", "only an in-progress or completed AQ checkpoint is resumable")
    else:
        elapsed_before_invocation = 0.0
        if initial_codebooks is None:
            codebooks, _ = initialize_rvq(
                x,
                m_count=m_count,
                clusters=clusters,
                iterations=initialization_iterations,
                row_chunk=row_chunk,
                device=device,
                seed=seed,
                deadline_check=deadline_check,
            )
        else:
            require(tuple(initial_codebooks.shape) == (m_count, clusters, x.shape[1]), "initial codebook geometry differs")
            codebooks = initial_codebooks.detach().cpu().to(torch.float16)
        codes, objective = beam_encode(
            x, codebooks, beam_size=beam_size, row_chunk=row_chunk, device=device,
            deadline_check=deadline_check,
        )
        trajectory = [
            {
                "sweep": 0,
                "phase": "post_initialization_beam_assignment",
                "objective_sse": objective,
                "objective_nmse": objective / max(denominator, 1e-30),
                "assignment_churn_rows": None,
                "assignment_churn_fraction": None,
                "relative_improvement": None,
                "solver": None,
            }
        ]
        stable = 0
        next_sweep = 1
        status = "in_progress"
        reason = "fixed_outer_schedule_in_progress" if fixed_schedule else "maximum_sweeps_exhausted_before_convergence"
        checkpoint(
            "post_initial_assignment",
            codebooks=codebooks,
            codes=codes,
            trajectory=trajectory,
            stable_counter=stable,
            next_sweep=next_sweep,
            training_status=status,
            status_reason=reason,
            elapsed_before_invocation=elapsed_before_invocation,
        )
        print(
            f"[full-aq] initial assignment objective_nmse={trajectory[0]['objective_nmse']:.8e}",
            flush=True,
        )

    for sweep in range(next_sweep, max_sweeps + 1):
        if deadline_check is not None:
            deadline_check(f"before_sweep_{sweep}")
        sweep_started = time.monotonic()
        print(f"[full-aq] sweep {sweep}/{max_sweeps} multi-RHS update started", flush=True)
        updated, solver = global_multi_rhs_codebook_update(
            x,
            codes,
            codebooks,
            clusters=clusters,
            atol=float(lsmr_config["atol"]),
            btol=float(lsmr_config["btol"]),
            conlim=float(lsmr_config["conlim"]),
            maxiter=int(lsmr_config["maxiter"]),
            deadline_check=deadline_check,
        )
        solver_seconds = time.monotonic() - sweep_started
        print(
            f"[full-aq] sweep {sweep}/{max_sweeps} multi-RHS update "
            f"finished in {solver_seconds:.2f}s rank={solver['retained_rank']} "
            f"nullity={solver['numerical_nullity']} "
            f"cond={solver['assignment_matrix_condition_estimate']:.6e} "
            f"normal_residual={solver['normal_equation_relative_residual']:.6e}",
            flush=True,
        )
        if not solver["accepted"]:
            status = "inconclusive"
            reason = "global_lsmr_did_not_reach_an_accepted_stop_condition"
            trajectory.append(
                {
                    "sweep": sweep,
                    "phase": "global_sparse_lsmr_codebook_update",
                    "objective_sse": None,
                    "objective_nmse": None,
                    "assignment_churn_rows": None,
                    "assignment_churn_fraction": None,
                    "relative_improvement": None,
                    "solver": solver,
                }
            )
            codebooks = updated
            checkpoint(
                f"sweep_{sweep}_solver_not_accepted",
                codebooks=codebooks,
                codes=codes,
                trajectory=trajectory,
                stable_counter=stable,
                next_sweep=sweep + 1,
                training_status=status,
                status_reason=reason,
                elapsed_before_invocation=elapsed_before_invocation,
            )
            break
        new_codes, new_objective = beam_encode(
            x, updated, beam_size=beam_size, row_chunk=row_chunk, device=device,
            deadline_check=deadline_check,
        )
        changed_rows = int((new_codes != codes).any(dim=1).sum().item())
        churn = changed_rows / int(x.shape[0])
        relative_improvement = (objective - new_objective) / max(abs(objective), 1e-30)
        trajectory.append(
            {
                "sweep": sweep,
                "phase": "global_sparse_lsmr_then_beam_assignment",
                "objective_sse": new_objective,
                "objective_nmse": new_objective / max(denominator, 1e-30),
                "assignment_churn_rows": changed_rows,
                "assignment_churn_fraction": churn,
                "relative_improvement": relative_improvement,
                "solver": solver,
            }
        )
        codebooks, codes, objective = updated, new_codes, new_objective
        print(
            f"[full-aq] sweep {sweep}/{max_sweeps} complete "
            f"objective_nmse={trajectory[-1]['objective_nmse']:.8e} "
            f"relative_improvement={relative_improvement:.8e} churn={churn:.8e} "
            f"elapsed={time.monotonic() - sweep_started:.2f}s",
            flush=True,
        )
        if not fixed_schedule and relative_improvement < -float(monotonicity_tolerance):
            status = "inconclusive"
            reason = "deployed_fp16_objective_violated_monotonicity_contract"
            checkpoint(
                f"sweep_{sweep}_monotonicity_violation",
                codebooks=codebooks,
                codes=codes,
                trajectory=trajectory,
                stable_counter=stable,
                next_sweep=sweep + 1,
                training_status=status,
                status_reason=reason,
                elapsed_before_invocation=elapsed_before_invocation,
            )
            break
        if fixed_schedule:
            if sweep == max_sweeps:
                status = "completed_fixed_schedule"
                reason = "author_fixed_outer_sweep_schedule_completed"
        else:
            sufficiently_stable = (
                sweep >= min_sweeps
                and 0.0 <= relative_improvement <= float(objective_tolerance)
                and churn <= float(churn_tolerance)
            )
            stable = stable + 1 if sufficiently_stable else 0
            if stable >= patience:
                status = "converged"
                reason = "objective_and_assignment_contract_satisfied"
        checkpoint(
            (
                "fit_completed_fixed_schedule"
                if status == "completed_fixed_schedule"
                else "fit_converged"
                if status == "converged"
                else f"sweep_{sweep}_complete"
            ),
            codebooks=codebooks,
            codes=codes,
            trajectory=trajectory,
            stable_counter=stable,
            next_sweep=sweep + 1,
            training_status=status,
            status_reason=reason,
            elapsed_before_invocation=elapsed_before_invocation,
        )
        if status == terminal_status:
            break
    if status == "in_progress":
        status = "inconclusive"
        reason = (
            "fixed_outer_schedule_did_not_reach_complete_boundary"
            if fixed_schedule
            else "maximum_sweeps_exhausted_before_convergence"
        )
    return {
        "training_status": status,
        "status_reason": reason,
        "codebooks": codebooks,
        "codes": codes,
        "objective_sse": objective,
        "objective_nmse": objective / max(denominator, 1e-30),
        "trajectory": trajectory,
        "completed_sweeps": len(trajectory) - 1,
        "fit_elapsed_seconds": elapsed_before_invocation + time.monotonic() - invocation_started,
    }


def validate_build_config(config_path: Path, model_key: str, ratio: int) -> dict[str, Any]:
    config = read_json(config_path)
    require(config.get("schema_version") == "full-additive-quantization-ple-w4-matrix-v1", "schema differs")
    require(config.get("status") == "fixed", "build matrix is not fixed")
    author = config.get("author_algorithm", {})
    require(author.get("repository") == AUTHOR_REPOSITORY, "AQ author repository differs")
    require(author.get("commit") == AUTHOR_COMMIT, "AQ author-code commit differs")
    require(author.get("source_file") == AUTHOR_FILE, "AQ author source file differs")
    models = config.get("models", {})
    require(model_key in models, f"unknown model selector: {model_key}")
    model = dict(models[model_key])
    rates = config.get("rates", {})
    ratio_key = f"r{ratio}"
    require(ratio_key in rates, f"unsupported ratio selector: {ratio}")
    rate = dict(rates[ratio_key])
    vocab, layers, d_ple = int(model["vocab_size"]), int(model["num_layers"]), int(model["d_ple"])
    rows = vocab * layers
    dense_bytes = rows * d_ple * 2
    require(int(model["dense_ple_bytes"]) == dense_bytes, "dense PLE byte contract differs")
    weights = Path(model["snapshot"]) / model["weight_file"]
    require(weights.is_file(), f"QAT parent weights are missing: {weights}")
    m_count, clusters, bits = int(rate["M"]), int(rate["K"]), int(rate["bits"])
    resident = expected_resident_bytes(rows, d_ple, m_count, clusters, bits)
    require(int(rate["expected_resident_bytes_by_model"][model_key]) == resident, "resident-byte contract differs")
    achieved_ratio = dense_bytes / resident
    require(achieved_ratio >= ratio, f"configured AQ geometry misses {ratio}x")
    fit = config.get("fit", {})
    require(int(fit.get("seed", -1)) == 0, "AQ seed must be zero")
    require(int(fit.get("fit_beam_size", 0)) == 16, "author-fixed AQ fitting must use beam 16")
    require(int(fit.get("final_beam_size", 0)) == 64, "author-fixed AQ final encoding must use beam 64")
    convergence = fit_completion_contract(fit)
    mode = completion_mode(convergence)
    if mode == "fixed_outer_sweeps":
        require(int(convergence.get("fixed_outer_sweeps", 0)) == 10, "author-fixed AQ schedule must use 10 outer sweeps")
        require(convergence.get("terminal_status") == "completed_fixed_schedule", "fixed-schedule terminal status differs")
        require(convergence.get("assignment_churn_role") == "diagnostic_only", "fixed-schedule churn must be diagnostic only")
        require(convergence.get("relative_improvement_role") == "diagnostic_only", "fixed-schedule objective change must be diagnostic only")
    else:
        require(mode == "objective_and_assignment_convergence", "AQ completion mode differs")
        require(int(convergence.get("max_sweeps", 0)) > int(convergence.get("min_sweeps", 0)) >= 1, "sweep contract differs")
        require(int(convergence.get("patience", 0)) >= 1, "convergence patience must be positive")
    return {
        "config": config,
        "model": model,
        "model_key": model_key,
        "ratio": ratio,
        "ratio_key": ratio_key,
        "rate": rate,
        "weights": weights,
        "rows": rows,
        "dense_bytes": dense_bytes,
        "resident_bytes": resident,
        "achieved_ratio": achieved_ratio,
    }


def deterministic_fit_row_ids(torch: Any, rows: int, requested: int, seed: int) -> Any:
    count = min(rows, requested)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(rows, generator=generator)[:count].sort().values


def load_fit_matrix(contract: dict[str, Any], row_ids: Any) -> Any:
    import torch
    from safetensors import safe_open

    model = contract["model"]
    vocab, layers, d_ple = int(model["vocab_size"]), int(model["num_layers"]), int(model["d_ple"])
    output = torch.empty((row_ids.numel(), d_ple), dtype=torch.float32)
    layer_ids = torch.div(row_ids, vocab, rounding_mode="floor")
    token_ids = row_ids.remainder(vocab)
    with safe_open(str(contract["weights"]), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(PLE_KEY)
        require(tuple(tensor_slice.get_shape()) == (vocab, layers * d_ple), "checkpoint PLE shape differs")
        for layer in range(layers):
            positions = torch.nonzero(layer_ids == layer, as_tuple=False).reshape(-1)
            if positions.numel() == 0:
                continue
            layer_values = tensor_slice[:, layer * d_ple : (layer + 1) * d_ple]
            output.index_copy_(0, positions, layer_values.index_select(0, token_ids.index_select(0, positions)).float())
    return output


def require_dev_shm(path: Path) -> None:
    resolved_parent = path.parent.resolve()
    require(str(resolved_parent) == SCRATCH_DIR or str(resolved_parent).startswith(SCRATCH_DIR + "/"), "capability reconstruction must remain under PLE_SCRATCH_DIR")


def validate_output_contract(contract: dict[str, Any], output_root: Path, reconstruction_path: Path) -> None:
    method_path = method_config_path(contract)
    method = read_json(method_path)
    expected_method_id = (
        f"{method_identity(contract)['method_id_prefix']}_{contract['model_key']}_"
        f"r{contract['ratio']}_seed{int(contract['config']['fit']['seed'])}"
    )
    require(method.get("status") == "build_required", "method config is not awaiting its first build")
    require(method.get("method_id") == expected_method_id, "method-config identity differs")
    require(method.get("method_family") == "full_additive_quantization", "method-config family differs")
    require(int(method.get("fit_assignment_beam_size", 0)) == 16, "method-config fit beam differs")
    require(int(method.get("final_encoding_beam_size", 0)) == 64, "method-config final encoding beam differs")
    require(
        Path(method["compressed_artifact_path"]).resolve() == (output_root / "compressed_ple.npz").resolve(),
        "--output-root differs from the method config",
    )
    require(
        Path(method["reconstruction_path"]).resolve() == reconstruction_path.resolve(),
        "--save-recon differs from the method config",
    )
    require(
        Path(method["runtime_receipt_path"]).resolve()
        == (output_root / "requested_row_runtime_receipt.json").resolve(),
        "runtime receipt path differs from the method config",
    )
    require(
        Path(method["online_adapter_path"]).resolve() == ONLINE_ADAPTER_PATH.resolve(),
        "online adapter path differs from the Full AQ codec",
    )


def fit_checkpoint_binding(
    contract: dict[str, Any],
    *,
    config_path: Path,
    row_ids: Any,
    seed: int,
) -> dict[str, Any]:
    model, rate = contract["model"], contract["rate"]
    binding = {
        "schema_version": "full-additive-quantization-ple-fit-binding-v1",
        "builder_sha256": sha256_file(Path(__file__)),
        "build_config_path": str(config_path.resolve()),
        "build_config_sha256": sha256_file(config_path),
        "model_selector": contract["model_key"],
        "ratio_bucket": contract["ratio_key"],
        "seed": int(seed),
        "source_model_id": model["model_id"],
        "source_revision": model["revision"],
        "source_weights_path": str(contract["weights"].resolve()),
        "source_weights_sha256": sha256_file(contract["weights"]),
        "fit_row_count": int(row_ids.numel()),
        "fit_row_ids_sha256": array_sha256(row_ids.detach().cpu().numpy()),
        "assignment_beams": {
            "fit": int(contract["config"]["fit"]["fit_beam_size"]),
            "final_encoding": int(contract["config"]["fit"]["final_beam_size"]),
        },
        "artifact_geometry": {
            "rows": int(contract["rows"]),
            "num_layers": int(model["num_layers"]),
            "d_ple": int(model["d_ple"]),
            "M": int(rate["M"]),
            "K": int(rate["K"]),
            "bits": int(rate["bits"]),
            "resident_bytes": int(contract["resident_bytes"]),
        },
    }
    return {**binding, "binding_sha256": canonical_json_sha256(binding)}


def write_fit_checkpoint(
    path: Path,
    *,
    stage: str,
    row_ids: Any,
    codebooks: Any | None,
    codes: Any | None,
    trajectory: list[dict[str, Any]],
    seed: int,
    binding: dict[str, Any],
    stable_counter: int,
    next_sweep: int,
    training_status: str,
    status_reason: str,
    fit_elapsed_seconds: float,
) -> None:
    import numpy as np
    import torch

    row_ids_array = row_ids.detach().cpu().numpy()
    codebooks_array = None if codebooks is None else codebooks.detach().cpu().to(torch.float16).numpy()
    codes_array = None if codes is None else codes.detach().cpu().numpy()
    content_hashes = {
        "fit_row_ids": array_sha256(row_ids_array),
        "codebooks_fp16": None if codebooks_array is None else array_sha256(codebooks_array),
        "fit_codes": None if codes_array is None else array_sha256(codes_array),
        "trajectory": canonical_json_sha256(trajectory),
    }
    payload: dict[str, Any] = {
        "fit_row_ids": row_ids_array,
        "metadata": np.array(json.dumps({
            "schema_version": "full-additive-quantization-ple-resumable-checkpoint-v2",
            "stage": stage,
            "seed": seed,
            "trajectory": trajectory,
            "stable_counter": int(stable_counter),
            "next_sweep": int(next_sweep),
            "training_status": training_status,
            "status_reason": status_reason,
            "fit_elapsed_seconds": float(fit_elapsed_seconds),
            "binding": binding,
            "content_hashes": content_hashes,
            "final_artifact": False,
        }, sort_keys=True)),
    }
    if codebooks_array is not None:
        payload["codebooks_fp16"] = codebooks_array
    if codes_array is not None:
        payload["fit_codes"] = codes_array
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_fit_checkpoint(
    path: Path,
    *,
    expected_binding: dict[str, Any],
    expected_row_ids: Any,
    maximum_sweeps: int,
) -> dict[str, Any]:
    """Load a complete sweep boundary, rejecting stale or partial state."""
    import numpy as np
    import torch

    require(path.is_file(), f"AQ resume checkpoint is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            require(
                set(checkpoint.files) == {"fit_row_ids", "metadata", "codebooks_fp16", "fit_codes"},
                "AQ resume checkpoint is partial or has unexpected arrays",
            )
            row_ids = np.asarray(checkpoint["fit_row_ids"])
            codebooks = np.asarray(checkpoint["codebooks_fp16"])
            codes = np.asarray(checkpoint["fit_codes"])
            metadata = json.loads(str(checkpoint["metadata"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"AQ resume checkpoint is corrupt: {path}") from error
    require(isinstance(metadata, dict), "AQ resume checkpoint metadata is not an object")
    require(
        metadata.get("schema_version") == "full-additive-quantization-ple-resumable-checkpoint-v2",
        "AQ resume checkpoint schema differs",
    )
    require(metadata.get("final_artifact") is False, "AQ resume checkpoint is incorrectly marked final")
    require(metadata.get("binding") == expected_binding, "AQ resume checkpoint binding differs")
    binding_without_digest = {key: value for key, value in expected_binding.items() if key != "binding_sha256"}
    require(
        expected_binding.get("binding_sha256") == canonical_json_sha256(binding_without_digest),
        "expected AQ resume binding digest differs",
    )
    content_hashes = metadata.get("content_hashes")
    require(isinstance(content_hashes, dict), "AQ resume checkpoint content hashes are missing")
    require(content_hashes.get("fit_row_ids") == array_sha256(row_ids), "AQ resume row IDs are corrupt")
    require(content_hashes.get("codebooks_fp16") == array_sha256(codebooks), "AQ resume codebooks are corrupt")
    require(content_hashes.get("fit_codes") == array_sha256(codes), "AQ resume codes are corrupt")
    trajectory = metadata.get("trajectory")
    require(isinstance(trajectory, list) and bool(trajectory), "AQ resume trajectory is partial")
    require(content_hashes.get("trajectory") == canonical_json_sha256(trajectory), "AQ resume trajectory is corrupt")
    expected_rows = expected_row_ids.detach().cpu().numpy()
    require(array_sha256(row_ids) == array_sha256(expected_rows), "AQ resume row partition differs")
    geometry = expected_binding["artifact_geometry"]
    m_count, clusters, d_ple = int(geometry["M"]), int(geometry["K"]), int(geometry["d_ple"])
    require(codebooks.dtype == np.float16, "AQ resume codebooks are not FP16 deployment values")
    require(tuple(codebooks.shape) == (m_count, clusters, d_ple), "AQ resume codebook geometry differs")
    require(tuple(codes.shape) == (row_ids.shape[0], m_count), "AQ resume code geometry differs")
    require(bool(np.issubdtype(codes.dtype, np.integer)), "AQ resume codes are not integers")
    require(bool((codes >= 0).all() and (codes < clusters).all()), "AQ resume codes are outside the codebook range")
    stable_counter = int(metadata.get("stable_counter", -1))
    next_sweep = int(metadata.get("next_sweep", -1))
    training_status = metadata.get("training_status")
    require(stable_counter >= 0, "AQ resume stable counter is invalid")
    require(1 <= next_sweep <= maximum_sweeps + 1, "AQ resume next sweep is invalid")
    require(int(trajectory[-1].get("sweep", -1)) == next_sweep - 1, "AQ resume trajectory boundary differs")
    require(training_status in {"in_progress", "converged", "completed_fixed_schedule"}, "AQ resume checkpoint is not resumable")
    require(
        metadata.get("stage")
        == (
            "fit_converged"
            if training_status == "converged"
            else "fit_completed_fixed_schedule"
            if training_status == "completed_fixed_schedule"
            else f"sweep_{next_sweep - 1}_complete"
        )
        or (training_status == "in_progress" and next_sweep == 1 and metadata.get("stage") == "post_initial_assignment"),
        "AQ resume checkpoint is not a complete sweep boundary",
    )
    fit_elapsed_seconds = float(metadata.get("fit_elapsed_seconds", float("nan")))
    require(math.isfinite(fit_elapsed_seconds) and fit_elapsed_seconds >= 0, "AQ resume elapsed time is invalid")
    return {
        "codebooks": torch.from_numpy(codebooks.copy()),
        "codes": torch.from_numpy(codes.copy()).to(torch.long),
        "trajectory": trajectory,
        "stable_counter": stable_counter,
        "next_sweep": next_sweep,
        "training_status": training_status,
        "status_reason": str(metadata.get("status_reason", "")),
        "fit_elapsed_seconds": fit_elapsed_seconds,
    }


def finalize_method_config(contract: dict[str, Any], summary_path: Path, summary: dict[str, Any]) -> Path:
    """Write this baseline's method config after completion."""
    method_path = method_config_path(contract)
    method = read_json(method_path)
    require(method.get("method_family") == "full_additive_quantization", "method-config family differs")
    require(method.get("method_id") == summary.get("method_id"), "method-config identity differs")
    require(method.get("status") == "build_required", "method config is not awaiting its first build")
    require(int(method.get("ple_resident_bytes", -1)) == contract["resident_bytes"], "method-config bytes differ")
    require(Path(method["compressed_artifact_path"]).resolve() == Path(summary["compressed_artifact_path"]).resolve(), "artifact path differs")
    require(Path(method["reconstruction_path"]).resolve() == Path(summary["reconstruction_path"]).resolve(), "reconstruction path differs")
    expected_runtime = Path(summary["compressed_artifact_path"]).parent / "requested_row_runtime_receipt.json"
    require(Path(method["runtime_receipt_path"]).resolve() == expected_runtime.resolve(), "runtime receipt path differs")
    require(Path(method["online_adapter_path"]).resolve() == ONLINE_ADAPTER_PATH.resolve(), "online adapter path differs")
    method.update(
        {
            "status": "ready_for_evaluation",
            "builder_sha256": sha256_file(Path(__file__)),
            "compressed_artifact_sha256": summary["compressed_artifact_sha256"],
            "compressed_artifact_file_bytes": summary["compressed_artifact_file_bytes"],
            "reconstruction_sha256": summary["reconstruction_sha256"],
            "reported_reconstruction_nmse": summary["reconstruction_nmse"],
            "training_metrics_sha256": sha256_file(summary_path),
            "roundtrip_receipt_sha256": sha256_file(summary["roundtrip_receipt_path"]),
            "online_adapter_sha256": sha256_file(ONLINE_ADAPTER_PATH),
        }
    )
    write_json(method_path, method)
    return method_path


def build_final_artifacts(
    contract: dict[str, Any],
    fit_result: dict[str, Any],
    *,
    output_root: Path,
    reconstruction_path: Path,
    final_beam_size: int,
    row_chunk: int,
    device: str,
    seed: int,
    deadline_check: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    require_dev_shm(reconstruction_path)
    model, rate = contract["model"], contract["rate"]
    vocab, layers, d_ple = int(model["vocab_size"]), int(model["num_layers"]), int(model["d_ple"])
    rows, m_count, clusters, bits = contract["rows"], int(rate["M"]), int(rate["K"]), int(rate["bits"])
    codebooks = fit_result["codebooks"].detach().cpu().to(torch.float16)
    code_dtype = np.uint8 if clusters <= 256 else np.uint16
    all_codes = np.empty((rows, m_count), dtype=code_dtype)
    reconstruction = torch.empty((vocab, layers * d_ple), dtype=torch.bfloat16)
    source_sse = 0.0
    source_norm = 0.0
    with safe_open(str(contract["weights"]), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(PLE_KEY)
        require(tuple(tensor_slice.get_shape()) == tuple(reconstruction.shape), "checkpoint PLE shape differs")
        for layer in range(layers):
            source = tensor_slice[:, layer * d_ple : (layer + 1) * d_ple].float().contiguous()
            layer_codes, _ = beam_encode(
                source,
                codebooks,
                beam_size=final_beam_size,
                row_chunk=row_chunk,
                device=device,
                deadline_check=deadline_check,
            )
            decoded = decode_codes(layer_codes, codebooks, row_chunk=row_chunk, device=device)
            start, stop = layer * vocab, (layer + 1) * vocab
            all_codes[start:stop] = layer_codes.numpy().astype(code_dtype, copy=False)
            reconstruction[:, layer * d_ple : (layer + 1) * d_ple] = decoded
            source_sse += float((decoded.float() - source).double().square().sum().item())
            source_norm += float(source.double().square().sum().item())
            print(f"[full-aq] encoded layer {layer + 1}/{layers}", flush=True)

    payload: dict[str, Any] = {"codebooks_fp16": codebooks.numpy()}
    for stage in range(m_count):
        payload[f"codes_m{stage:02d}"] = pack_nbit(all_codes[:, stage], bits)
    actual_resident = sum(int(value.nbytes) for value in payload.values())
    require(actual_resident == contract["resident_bytes"], "serialized arrays violate resident-byte contract")
    identity = method_identity(contract)
    metadata = {
        "schema_version": "full-additive-quantization-ple-artifact-v1",
        "compressed_ple_format_id": FORMAT_ID,
        "method_id": f"{identity['method_id_prefix']}_{contract['model_key']}_r{contract['ratio']}_seed{seed}",
        "method_family": "full_additive_quantization",
        "publication_name": identity["publication_name"],
        "publication_role": "external_baseline",
        "primary_paper": PRIMARY_PAPER,
        "author_repository": AUTHOR_REPOSITORY,
        "author_commit": AUTHOR_COMMIT,
        "author_algorithm_core": ["encodePointsBeamSearch", "learnCodebooksAQ sparse LSMR update"],
        "training_status": fit_result["training_status"],
        "training_status_reason": fit_result["status_reason"],
        "initialization": "weights-only greedy residual k-means",
        "optimization": "alternating author-style beam assignment and a multi-RHS Moore-Penrose solution of the same unregularized global sparse least-squares codebook update",
        "fit_assignment_beam_size": int(contract["config"]["fit"]["fit_beam_size"]),
        "final_encoding_beam_size": int(final_beam_size),
        "calibration_corpus_used": False,
        "hidden_raw_copy_allowed": False,
        "full_ple_coverage": True,
        "deployment_mode": "online_requested_token_row_decode",
        "capability_mode": "dense_reconstruction_injection_ephemeral_only",
        "source_model_id": model["model_id"],
        "source_revision": model["revision"],
        "ple_key": PLE_KEY,
        "ple_shape": [vocab, layers * d_ple],
        "num_layers": layers,
        "d_ple": d_ple,
        "M": m_count,
        "K": clusters,
        "bits": bits,
        "storage_bytes": actual_resident,
        "dense_ple_bytes": contract["dense_bytes"],
        "compression_ratio": contract["achieved_ratio"],
        "seed": seed,
    }
    payload["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    artifact_path = output_root / "compressed_ple.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(artifact_path, **payload)
    reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({RECON_KEY: reconstruction.contiguous()}, reconstruction_path)
    audit_ids = sorted({0, 1, vocab // 2, vocab - 2, vocab - 1})
    audit = audit_requested_rows(
        artifact_path,
        reconstruction_path,
        token_ids=audit_ids,
        device=device,
    )
    write_json(output_root / "requested_row_roundtrip_receipt.json", audit)
    return {
        "compressed_artifact_path": str(artifact_path),
        "compressed_artifact_sha256": artifact_sha256(artifact_path),
        "compressed_artifact_file_bytes": artifact_path.stat().st_size,
        "reconstruction_path": str(reconstruction_path),
        "reconstruction_sha256": artifact_sha256(reconstruction_path),
        "roundtrip_receipt_path": str(output_root / "requested_row_roundtrip_receipt.json"),
        "resident_bytes": actual_resident,
        "compression_ratio": contract["achieved_ratio"],
        "reconstruction_nmse": source_sse / max(source_norm, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/fit/aq_w4a16.json"))
    parser.add_argument("--model", choices=("e2b", "e4b"), required=True)
    parser.add_argument("--ratio", choices=(20, 40, 100), type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--save-recon", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only from a fully validated AQ sweep-boundary checkpoint",
    )
    args = parser.parse_args()

    import numpy as np
    import torch

    started = time.time()
    config_path = Path(args.config).resolve()
    contract = validate_build_config(config_path, args.model, args.ratio)
    fit_config = contract["config"]["fit"]
    training_completion = fit_completion_contract(fit_config)
    output_root = Path(args.output_root).resolve()
    reconstruction_path = Path(args.save_recon)
    require_dev_shm(reconstruction_path)
    validate_output_contract(contract, output_root, reconstruction_path)
    artifact_path = output_root / "compressed_ple.npz"
    require(not artifact_path.exists(), f"refusing to overwrite existing final artifact: {artifact_path}")
    seed = int(fit_config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    row_ids = deterministic_fit_row_ids(torch, contract["rows"], int(fit_config["fit_rows"]), seed)
    checkpoint_binding = fit_checkpoint_binding(
        contract,
        config_path=config_path,
        row_ids=row_ids,
        seed=seed,
    )
    fit_matrix = load_fit_matrix(contract, row_ids)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "fit_checkpoint.npz"
    resume_state = None
    if checkpoint_path.exists():
        require(args.resume, "AQ checkpoint exists; pass --resume to validate and continue it")
        resume_state = load_fit_checkpoint(
            checkpoint_path,
            expected_binding=checkpoint_binding,
            expected_row_ids=row_ids,
            maximum_sweeps=maximum_sweeps(training_completion),
        )
    else:
        write_fit_checkpoint(
            checkpoint_path,
            stage="fit_inputs_ready",
            row_ids=row_ids,
            codebooks=None,
            codes=None,
            trajectory=[],
            seed=seed,
            binding=checkpoint_binding,
            stable_counter=0,
            next_sweep=1,
            training_status="not_started",
            status_reason="initialization_not_complete",
            fit_elapsed_seconds=0.0,
        )

    def checkpoint_fit(state: dict[str, Any]) -> None:
        write_fit_checkpoint(
            checkpoint_path,
            stage=str(state["stage"]),
            row_ids=row_ids,
            codebooks=state["codebooks"],
            codes=state["codes"],
            trajectory=state["trajectory"],
            seed=seed,
            binding=checkpoint_binding,
            stable_counter=int(state["stable_counter"]),
            next_sweep=int(state["next_sweep"]),
            training_status=str(state["training_status"]),
            status_reason=str(state["status_reason"]),
            fit_elapsed_seconds=float(state["fit_elapsed_seconds"]),
        )

    deadline = SchedulerDeadline.from_environment()
    deadline.install_signal_handlers()
    try:
        fit_result = fit_full_aq(
            fit_matrix,
            m_count=int(contract["rate"]["M"]),
            clusters=int(contract["rate"]["K"]),
            beam_size=int(fit_config["fit_beam_size"]),
            row_chunk=int(fit_config["row_chunk"]),
            device=args.device,
            seed=seed,
            initialization_iterations=int(fit_config["initialization_kmeans_iterations"]),
            convergence=dict(training_completion),
            lsmr_config=dict(fit_config["lsmr"]),
            resume_state=resume_state,
            deadline_check=deadline.check,
            checkpoint_callback=checkpoint_fit,
        )
        trajectory_path = output_root / "training_trajectory.jsonl"
        write_jsonl(trajectory_path, fit_result["trajectory"])
        identity = method_identity(contract)
        expected_completion_status = completion_status(training_completion)
        common = {
            "schema_version": "full-additive-quantization-ple-build-summary-v1",
            "method_family": "full_additive_quantization",
            "method_id": f"{identity['method_id_prefix']}_{args.model}_r{args.ratio}_seed{seed}",
            "publication_name": identity["publication_name"],
            "publication_role": "external_baseline",
            "primary_paper": PRIMARY_PAPER,
            "author_repository": AUTHOR_REPOSITORY,
            "author_commit": AUTHOR_COMMIT,
            "model_selector": args.model,
            "ratio_bucket": f"r{args.ratio}",
            "source_model_id": contract["model"]["model_id"],
            "source_revision": contract["model"]["revision"],
            "source_weights": str(contract["weights"]),
            "source_weights_sha256": checkpoint_binding["source_weights_sha256"],
            "fit_row_count": int(row_ids.numel()),
            "fit_row_sampling": "seeded uniform sample without replacement from layer-major PLE rows",
            "fit_row_ids_sha256": hashlib.sha256(row_ids.numpy().tobytes()).hexdigest(),
            "fit_assignment_beam_size": int(fit_config["fit_beam_size"]),
            "final_encoding_beam_size": int(fit_config["final_beam_size"]),
            "training_status": fit_result["training_status"],
            "training_status_reason": fit_result["status_reason"],
            "completed_sweeps": fit_result["completed_sweeps"],
            "fit_elapsed_seconds": fit_result["fit_elapsed_seconds"],
            "fit_objective_sse": fit_result["objective_sse"],
            "fit_objective_nmse": fit_result["objective_nmse"],
            "training_trajectory_path": str(trajectory_path),
            "training_completion_contract": training_completion,
            "lsmr_contract": fit_config["lsmr"],
            "resident_bytes": contract["resident_bytes"],
            "compression_ratio": contract["achieved_ratio"],
            "elapsed_seconds_before_full_encoding": time.time() - started,
        }
        if completion_mode(training_completion) == "objective_and_assignment_convergence":
            common["convergence_contract"] = training_completion
        if fit_result["training_status"] != expected_completion_status:
            state_path = output_root / "inconclusive_fit_state.npz"
            np.savez(
                state_path,
                codebooks_fp16=fit_result["codebooks"].numpy(),
                fit_codes=fit_result["codes"].numpy(),
                fit_row_ids=row_ids.numpy(),
                metadata=np.array(json.dumps({**common, "final_artifact": False}, sort_keys=True)),
            )
            summary = {
                **common,
                "final_artifact_emitted": False,
                "inconclusive_fit_state_path": str(state_path),
                "inconclusive_fit_state_sha256": sha256_file(state_path),
            }
            write_json(output_root / "summary.json", summary)
            print(f"[full-aq] INCONCLUSIVE: {fit_result['status_reason']}", flush=True)
            raise SystemExit(2)

        artifacts = build_final_artifacts(
            contract,
            fit_result,
            output_root=output_root,
            reconstruction_path=reconstruction_path,
            final_beam_size=int(fit_config["final_beam_size"]),
            row_chunk=int(fit_config["row_chunk"]),
            device=args.device,
            seed=seed,
            deadline_check=deadline.check,
        )
        summary = {
            **common,
            **artifacts,
            "final_artifact_emitted": True,
            "elapsed_seconds": time.time() - started,
        }
        summary_path = output_root / "summary.json"
        write_json(summary_path, summary)
        method_path = finalize_method_config(contract, summary_path, summary)
        print(
            f"[full-aq] {fit_result['training_status']}; NMSE={artifacts['reconstruction_nmse']:.8f} "
            f"resident={artifacts['resident_bytes']} ratio={artifacts['compression_ratio']:.6f}x; "
            f"method_config={method_path}",
            flush=True,
        )
    except DeadlineSafeStop as error:
        receipt = {
            "schema_version": "full-additive-quantization-ple-deadline-safe-termination-v1",
            "status": "compute_limited_inconclusive",
            "reason": str(error),
            "final_artifact_emitted": False,
            "fit_checkpoint_path": str(checkpoint_path),
            "fit_checkpoint_sha256": sha256_file(checkpoint_path),
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "deadline_safe_termination.json", receipt)
        print(f"[full-aq] DEADLINE-SAFE STOP: {error}", flush=True)
        raise SystemExit(3)
    finally:
        deadline.restore_signal_handlers()


if __name__ == "__main__":
    main()
