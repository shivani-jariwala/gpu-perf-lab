"""Bottleneck classification: correlate timing breakdown with GPU telemetry.

This is the heart of the third resume bullet. Given one run's step breakdown
(compute / dataloader / comm shares) and its aggregated GPU telemetry (util,
SM-active, memory-bandwidth util), decide the single most likely limiter:

    INPUT_PIPELINE : the dataloader is a large share of the step -> GPU starved.
    COMM           : DDP gradient all-reduce is a large share -> comm-bound.
    MEMORY         : GPU busy but memory-bandwidth util is high -> memory-bound.
    COMPUTE        : GPU busy, not comm/input/memory limited -> compute-bound.
    UNKNOWN        : not enough signal (e.g. GPU nearly idle, no telemetry).

Order matters: we check the host-side limiters (input, comm) first because a
run can show high GPU util yet still be gated by them between/around kernels.
All thresholds are provisional (tuned on SIMULATED data) and live in the config.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Bottleneck, GpuTelemetry, RunRecord


# Provisional defaults, overridden by config `thresholds:` when available.
DEFAULT_THRESHOLDS = {
    "comm_bound_share": 0.30,
    "input_bound_share": 0.25,
    "compute_bound_util": 90.0,
    "memory_bound_mem_util": 70.0,
}


@dataclass
class _AggTelemetry:
    gpu_util_pct: float = 0.0
    sm_active_pct: float = 0.0
    mem_util_pct: float = 0.0
    mem_used_pct: float = 0.0
    have: bool = False


def _aggregate_telemetry(telemetry: list[GpuTelemetry]) -> _AggTelemetry:
    """Average the per-GPU telemetry into node-level numbers for classification."""
    if not telemetry:
        return _AggTelemetry(have=False)
    n = len(telemetry)
    return _AggTelemetry(
        gpu_util_pct=sum(t.gpu_util_pct for t in telemetry) / n,
        sm_active_pct=sum(t.sm_active_pct for t in telemetry) / n,
        mem_util_pct=sum(t.mem_util_pct for t in telemetry) / n,
        mem_used_pct=sum(t.mem_used_pct for t in telemetry) / n,
        have=True,
    )


def classify(record: RunRecord, thresholds: dict | None = None) -> tuple[Bottleneck, str]:
    """Return (bottleneck, human reason) for one run."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    shares = record.breakdown.share()   # compute/dataloader/comm, sum ~= 1
    tel = _aggregate_telemetry(record.telemetry)

    data_share = shares["dataloader"]
    comm_share = shares["comm"]

    # 1) Host-side input pipeline: GPU waiting on data.
    if data_share >= th["input_bound_share"]:
        return (Bottleneck.INPUT_PIPELINE,
                f"dataloader is {data_share:.0%} of the step "
                f"(>= {th['input_bound_share']:.0%}); GPU is starved for input")

    # 2) Communication: gradient all-reduce dominates.
    if comm_share >= th["comm_bound_share"]:
        return (Bottleneck.COMM,
                f"exposed all-reduce is {comm_share:.0%} of the step "
                f"(>= {th['comm_bound_share']:.0%}); communication-bound")

    # 3) GPU-bound: split compute vs memory using telemetry (if we have it).
    if tel.have:
        if tel.mem_util_pct >= th["memory_bound_mem_util"]:
            return (Bottleneck.MEMORY,
                    f"memory-bandwidth util {tel.mem_util_pct:.0f}% "
                    f"(>= {th['memory_bound_mem_util']:.0f}%) with low comm/input; memory-bound")
        if tel.gpu_util_pct >= th["compute_bound_util"]:
            return (Bottleneck.COMPUTE,
                    f"GPU util {tel.gpu_util_pct:.0f}% (>= {th['compute_bound_util']:.0f}%), "
                    f"comm {comm_share:.0%} / input {data_share:.0%} low; compute-bound")
        if tel.gpu_util_pct < 50.0:
            return (Bottleneck.UNKNOWN,
                    f"GPU util only {tel.gpu_util_pct:.0f}% but no dominant "
                    f"input/comm share; inconclusive")
        return (Bottleneck.COMPUTE,
                f"GPU util {tel.gpu_util_pct:.0f}%, compute is the largest share "
                f"({shares['compute']:.0%}); compute-bound")

    # 4) No telemetry (e.g. Colab without DCGM): fall back to breakdown only.
    if shares["compute"] >= 0.5:
        return (Bottleneck.COMPUTE,
                f"no telemetry; compute is {shares['compute']:.0%} of the step "
                f"(comm {comm_share:.0%}, input {data_share:.0%})")
    return (Bottleneck.UNKNOWN, "no telemetry and no dominant timing share")


def classify_in_place(record: RunRecord, thresholds: dict | None = None) -> RunRecord:
    """Set record.bottleneck + reason and return it (convenience)."""
    b, reason = classify(record, thresholds)
    record.bottleneck = b
    record.bottleneck_reason = reason
    return record
