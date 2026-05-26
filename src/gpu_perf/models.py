"""The shared data contract for the whole performance lab.

Design decision (Phase 0): every measurement this lab produces — a DDP training
run, an NCCL microbenchmark, a single GPU-config in a scaling comparison —
serializes into ONE shape: ``RunRecord``. Downstream code (telemetry
correlation, bottleneck classification, comparison reports, charts) is written
once against this contract and never needs to know how a given number was
captured (real hardware vs a SIMULATED fixture).

This is the backbone of the project. Keep it small, explicit, and JSON-round-
trippable so a report generated on the Mac from SIMULATED fixtures is byte-for-
byte the same code path as one generated on the lab box from real captures.

Concepts this schema is built to make defensible in an interview:
  * step time / throughput           -> how fast one training step is, in ms and samples/s
  * compute vs dataloader vs comm     -> WHERE the step time goes (the bottleneck story)
  * GPU util / memory / SM activity   -> DCGM/pynvml telemetry aligned to the run
  * algbw vs busbw                    -> NCCL collective bandwidth (bus bw is the hw-comparable one)
  * strong vs weak scaling            -> how we vary batch size across GPU configs
  * speedup / scaling efficiency      -> the comparison-report headline numbers
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Enums: small, closed vocabularies that keep reports consistent.
# All inherit from ``str`` so they serialize to plain strings in JSON with no
# custom encoder.
# --------------------------------------------------------------------------- #


class Backend(str, Enum):
    """Which launcher actually ran the job.

    The launcher (Phase 1) prefers Slurm if ``sbatch``/``sinfo`` exist, else
    falls back to torchrun. We record which path was taken so a report never
    silently implies Slurm when torchrun was used.
    """

    SLURM = "slurm"
    TORCHRUN = "torchrun"
    UNKNOWN = "unknown"


class ScalingMode(str, Enum):
    """How batch size is varied as GPU count changes.

    STRONG: global batch size is FIXED; per-GPU batch = global / num_gpus.
            Adding GPUs should make one step faster -> measures SPEEDUP.
    WEAK:   per-GPU batch size is FIXED; global batch = per_gpu * num_gpus.
            Each GPU does constant work -> measures EFFICIENCY at constant load.
    SINGLE: a single-GPU baseline (no scaling dimension yet).
    """

    STRONG = "strong"
    WEAK = "weak"
    SINGLE = "single"


class Bottleneck(str, Enum):
    """What most limited a run, from correlating timings with telemetry.

    COMPUTE:        GPU busy (~high SM/util), step time ~ compute time.
    MEMORY:         high memory-bandwidth/occupancy pressure limits compute.
    INPUT_PIPELINE: GPU starved waiting on the dataloader (host/input bound).
    COMM:           a large share of step time is DDP gradient sync (NCCL).
    UNKNOWN:        not enough signal to classify (or not yet analyzed).
    """

    COMPUTE = "compute"
    MEMORY = "memory"
    INPUT_PIPELINE = "input_pipeline"
    COMM = "comm"
    UNKNOWN = "unknown"


class NcclOp(str, Enum):
    """NCCL collective operations we microbenchmark (nccl-tests binaries)."""

    ALL_REDUCE = "all_reduce"   # all_reduce_perf  -- the DDP gradient-sync collective
    ALL_GATHER = "all_gather"   # all_gather_perf
    BROADCAST = "broadcast"     # broadcast_perf


# --------------------------------------------------------------------------- #
# Leaf value objects
# --------------------------------------------------------------------------- #


@dataclass
class TimingStats:
    """Summary statistics for a series of per-step timings, in milliseconds.

    We keep the distribution (not just the mean) because tail latency (p90/p99)
    is where input-pipeline stalls and comm hiccups show up.
    """

    mean_ms: float = 0.0
    median_ms: float = 0.0
    p90_ms: float = 0.0
    std_ms: float = 0.0
    count: int = 0

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "TimingStats":
        """Build stats from raw per-step timings (already warmup-trimmed)."""
        if not samples_ms:
            return cls()
        s = sorted(samples_ms)
        n = len(s)
        # nearest-rank p90; simple and dependency-free.
        p90_idx = max(0, min(n - 1, int(round(0.90 * n)) - 1))
        return cls(
            mean_ms=round(statistics.fmean(s), 4),
            median_ms=round(statistics.median(s), 4),
            p90_ms=round(s[p90_idx], 4),
            std_ms=round(statistics.pstdev(s), 4) if n > 1 else 0.0,
            count=n,
        )


@dataclass
class StepBreakdown:
    """Where a single training step's wall-clock time goes, in milliseconds.

    These are mean-per-step components. They do not have to sum exactly to the
    step time (compute and comm can overlap in DDP), but their RELATIVE sizes
    are what the bottleneck classifier reasons about:

        dataloader_ms : time the step spent WAITING on input data (host side)
        compute_ms    : forward + backward compute on the GPU
        comm_ms       : DDP gradient all-reduce (NCCL) time attributable to the step
    """

    compute_ms: float = 0.0
    dataloader_ms: float = 0.0
    comm_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return round(self.compute_ms + self.dataloader_ms + self.comm_ms, 4)

    def share(self) -> dict[str, float]:
        """Fraction of the (summed) step time in each component, 0..1."""
        t = self.total_ms
        if t <= 0:
            return {"compute": 0.0, "dataloader": 0.0, "comm": 0.0}
        return {
            "compute": round(self.compute_ms / t, 4),
            "dataloader": round(self.dataloader_ms / t, 4),
            "comm": round(self.comm_ms / t, 4),
        }


@dataclass
class GpuTelemetry:
    """A per-GPU telemetry summary sampled DURING a run.

    Source is DCGM (``dcgmi dmon``) primary, pynvml/nvidia-smi fallback. Values
    are aggregated (mean/peak) over the sampling window so one run has one row
    per GPU. Field names mirror the DCGM field concepts.
    """

    gpu_index: int = 0
    gpu_util_pct: float = 0.0        # DCGM GR active / nvidia-smi utilization.gpu (mean)
    sm_active_pct: float = 0.0       # DCGM SM active (mean) -- finer than gross util
    mem_used_mb: float = 0.0         # peak memory used
    mem_total_mb: float = 0.0        # device capacity
    mem_util_pct: float = 0.0        # DCGM memory (bandwidth) utilization (mean)
    power_w: float = 0.0             # mean board power draw
    power_limit_w: float = 0.0       # enforced power limit
    sample_count: int = 0            # number of telemetry samples aggregated

    @property
    def mem_used_pct(self) -> float:
        return round(100.0 * self.mem_used_mb / self.mem_total_mb, 2) if self.mem_total_mb else 0.0


@dataclass
class NcclResult:
    """One row of an NCCL microbenchmark sweep (from nccl-tests output).

    algbw vs busbw (the concept an interviewer will probe):
      algbw ("algorithm bandwidth") = message_size / time. What the ALGORITHM
        moved from the caller's point of view.
      busbw ("bus bandwidth")       = algbw * a collective-specific factor
        (for ring all-reduce: 2*(n-1)/n). It reflects the ACTUAL traffic on the
        interconnect and is the number you compare against hardware peak
        (NVLink/PCIe), because it's independent of GPU count for a given link.
    """

    op: NcclOp = NcclOp.ALL_REDUCE
    size_bytes: int = 0
    time_us: float = 0.0             # collective latency for this size
    algbw_gbps: float = 0.0          # algorithm bandwidth, GB/s
    busbw_gbps: float = 0.0          # bus bandwidth, GB/s (hardware-comparable)
    num_gpus: int = 0
    wrong_count: int = 0             # correctness errors reported by nccl-tests (should be 0)


# --------------------------------------------------------------------------- #
# Run configuration (the fixed model/dataset + the knob we vary)
# --------------------------------------------------------------------------- #


@dataclass
class RunConfig:
    """The identity of a run: what was fixed and what was varied.

    Model + dataset are FIXED across the whole study (ResNet-50 on synthetic
    ImageNet-shaped data). The variables are num_gpus and batch size, driven by
    ``scaling_mode``. Two configs that differ only in num_gpus/batch are what a
    comparison report puts side by side.
    """

    model: str = "resnet50"
    dataset: str = "synthetic-imagenet"
    num_gpus: int = 1
    global_batch_size: int = 0       # total batch across all GPUs
    per_gpu_batch_size: int = 0      # micro-batch each GPU processes
    scaling_mode: ScalingMode = ScalingMode.SINGLE
    precision: str = "amp"           # "fp32" | "amp" (mixed) | "tf32"
    steps: int = 0                   # measured steps (after warmup)
    warmup_steps: int = 0
    backend: Backend = Backend.UNKNOWN
    label: str = ""                  # short human tag, e.g. "strong-2gpu"


# --------------------------------------------------------------------------- #
# The backbone record + a comparison aggregate
# --------------------------------------------------------------------------- #


@dataclass
class RunRecord:
    """The single, uniform record for one measured run. THE backbone.

    A training run fills in step_time/throughput/breakdown/telemetry; an NCCL
    microbenchmark fills in ``nccl``. Every artifact this lab reads or writes is
    a RunRecord (or a list of them), whether it came from real hardware or a
    SIMULATED fixture.
    """

    run_id: str
    config: RunConfig
    started_at: str = ""                              # ISO-8601 UTC

    # Training measurements ----------------------------------------------------
    step_time: TimingStats = field(default_factory=TimingStats)
    throughput_samples_per_s: float = 0.0             # global samples/sec (images/sec)
    breakdown: StepBreakdown = field(default_factory=StepBreakdown)

    # Telemetry (one entry per GPU) -------------------------------------------
    telemetry: list[GpuTelemetry] = field(default_factory=list)

    # NCCL microbenchmark rows (empty for pure training runs) -----------------
    nccl: list[NcclResult] = field(default_factory=list)

    # Analysis (filled by Phase 4 correlation) --------------------------------
    bottleneck: Bottleneck = Bottleneck.UNKNOWN
    bottleneck_reason: str = ""

    # Provenance --------------------------------------------------------------
    hostname: str = ""
    gpu_model: str = ""                               # e.g. "NVIDIA RTX A5000 (24GB)"
    simulated: bool = False                           # True => NOT captured from real hardware
    notes: str = ""

    # --- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict turns nested dataclasses to dicts but leaves Enums as Enum;
        # normalize every enum to its string value for clean JSON.
        d["config"]["scaling_mode"] = self.config.scaling_mode.value
        d["config"]["backend"] = self.config.backend.value
        d["bottleneck"] = self.bottleneck.value
        # asdict() leaves the NcclOp enum on each nccl row as an Enum instance;
        # normalize each to its string value using the original objects.
        for orig, row in zip(self.nccl, d["nccl"]):
            row["op"] = orig.op.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        cfg = d.get("config", {}) or {}
        config = RunConfig(
            model=cfg.get("model", "resnet50"),
            dataset=cfg.get("dataset", "synthetic-imagenet"),
            num_gpus=cfg.get("num_gpus", 1),
            global_batch_size=cfg.get("global_batch_size", 0),
            per_gpu_batch_size=cfg.get("per_gpu_batch_size", 0),
            scaling_mode=ScalingMode(cfg.get("scaling_mode", "single")),
            precision=cfg.get("precision", "amp"),
            steps=cfg.get("steps", 0),
            warmup_steps=cfg.get("warmup_steps", 0),
            backend=Backend(cfg.get("backend", "unknown")),
            label=cfg.get("label", ""),
        )
        st = d.get("step_time", {}) or {}
        bd = d.get("breakdown", {}) or {}
        return cls(
            run_id=d["run_id"],
            config=config,
            started_at=d.get("started_at", ""),
            step_time=TimingStats(
                mean_ms=st.get("mean_ms", 0.0), median_ms=st.get("median_ms", 0.0),
                p90_ms=st.get("p90_ms", 0.0), std_ms=st.get("std_ms", 0.0),
                count=st.get("count", 0),
            ),
            throughput_samples_per_s=d.get("throughput_samples_per_s", 0.0),
            breakdown=StepBreakdown(
                compute_ms=bd.get("compute_ms", 0.0),
                dataloader_ms=bd.get("dataloader_ms", 0.0),
                comm_ms=bd.get("comm_ms", 0.0),
            ),
            telemetry=[GpuTelemetry(**t) for t in (d.get("telemetry", []) or [])],
            nccl=[
                NcclResult(
                    op=NcclOp(r.get("op", "all_reduce")),
                    size_bytes=r.get("size_bytes", 0),
                    time_us=r.get("time_us", 0.0),
                    algbw_gbps=r.get("algbw_gbps", 0.0),
                    busbw_gbps=r.get("busbw_gbps", 0.0),
                    num_gpus=r.get("num_gpus", 0),
                    wrong_count=r.get("wrong_count", 0),
                )
                for r in (d.get("nccl", []) or [])
            ],
            bottleneck=Bottleneck(d.get("bottleneck", "unknown")),
            bottleneck_reason=d.get("bottleneck_reason", ""),
            hostname=d.get("hostname", ""),
            gpu_model=d.get("gpu_model", ""),
            simulated=d.get("simulated", False),
            notes=d.get("notes", ""),
        )


@dataclass
class Comparison:
    """A side-by-side of runs that share a fixed model/dataset.

    Populated in Phase 4. The baseline is the single-GPU run; speedup and
    scaling efficiency are computed against it. Defined here so the contract for
    the comparison report lives with the rest of the schema.

    speedup(N)    = throughput(N) / throughput(1 GPU)
    efficiency(N) = speedup(N) / N        (1.0 == perfect/linear scaling)
    """

    title: str
    scaling_mode: ScalingMode
    runs: list[RunRecord] = field(default_factory=list)
    simulated: bool = False

    def baseline(self) -> Optional[RunRecord]:
        singles = [r for r in self.runs if r.config.num_gpus == 1]
        return min(singles, key=lambda r: r.config.num_gpus, default=None)

    def speedup_efficiency(self) -> list[dict[str, Any]]:
        base = self.baseline()
        base_tp = base.throughput_samples_per_s if base else 0.0
        rows: list[dict[str, Any]] = []
        for r in sorted(self.runs, key=lambda x: x.config.num_gpus):
            speedup = (r.throughput_samples_per_s / base_tp) if base_tp else 0.0
            eff = (speedup / r.config.num_gpus) if r.config.num_gpus else 0.0
            rows.append({
                "num_gpus": r.config.num_gpus,
                "global_batch_size": r.config.global_batch_size,
                "per_gpu_batch_size": r.config.per_gpu_batch_size,
                "throughput_samples_per_s": round(r.throughput_samples_per_s, 2),
                "speedup": round(speedup, 3),
                "efficiency": round(eff, 3),
                "bottleneck": r.bottleneck.value,
            })
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "scaling_mode": self.scaling_mode.value,
            "simulated": self.simulated,
            "baseline_run_id": self.baseline().run_id if self.baseline() else None,
            "scaling": self.speedup_efficiency(),
            "runs": [r.to_dict() for r in self.runs],
        }


# --------------------------------------------------------------------------- #
# Small timing helper (mirrors the sibling project's check_timer)
# --------------------------------------------------------------------------- #


class run_timer:
    """Context manager that stamps a run's start time and wall-clock duration.

        with run_timer() as t:
            ... run the job ...
        record.started_at = t.started_at   # t.duration_s also available
    """

    def __init__(self) -> None:
        self.started_at: str = ""
        self.duration_s: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "run_timer":
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.duration_s = round(time.perf_counter() - self._t0, 3)
