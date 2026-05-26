"""Build strong/weak scaling comparisons from a set of RunRecords.

Groups training runs by scaling mode and constructs a ``Comparison`` (which
computes speedup and efficiency against the 1-GPU baseline). NCCL records are
kept aside as context for interpreting comm-bound runs.
"""

from __future__ import annotations

from ..models import Comparison, RunRecord, ScalingMode


def split_records(records: list[RunRecord]) -> tuple[list[RunRecord], list[RunRecord]]:
    """Separate training runs from NCCL microbenchmark runs."""
    training = [r for r in records if not r.nccl]
    nccl = [r for r in records if r.nccl]
    return training, nccl


def build_comparisons(records: list[RunRecord]) -> list[Comparison]:
    """One Comparison per scaling mode that has >= 2 GPU configurations.

    A run tagged SINGLE is a standalone baseline and isn't a comparison on its
    own; strong/weak groups each contain their own 1-GPU and 2-GPU points.
    """
    training, _ = split_records(records)
    by_mode: dict[ScalingMode, list[RunRecord]] = {}
    for r in training:
        by_mode.setdefault(r.config.scaling_mode, []).append(r)

    comparisons: list[Comparison] = []
    for mode in (ScalingMode.STRONG, ScalingMode.WEAK):
        runs = by_mode.get(mode, [])
        # need distinct GPU counts to compare
        gpu_counts = {r.config.num_gpus for r in runs}
        if len(runs) >= 2 and len(gpu_counts) >= 2:
            comparisons.append(Comparison(
                title=f"{mode.value.capitalize()} scaling (model+dataset fixed)",
                scaling_mode=mode,
                runs=sorted(runs, key=lambda r: r.config.num_gpus),
                simulated=any(r.simulated for r in runs),
            ))
    return comparisons
