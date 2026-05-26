#!/usr/bin/env python3
"""Generate ALL committed SIMULATED evidence for the repo.

Everything this writes is synthetic, sized to an ASSUMED 2x NVIDIA RTX A5000
(24 GB, PCIe) node, and clearly labelled SIMULATED in both the filename and the
file content. It exists so the parsers, classifier, comparison report, and
charts can be exercised and reviewed BEFORE real hardware is available.

The important property: fixtures flow through the SAME code paths as real
captures — the NCCL log is parsed by the real parser, and the report/charts are
produced by the real report pipeline. Swapping in real lab-box captures later
changes only the input files, not the code.

Run:
    python scripts/make_simulated_evidence.py

Outputs (all under evidence/samples/, which is committed):
    train-*.SIMULATED.json          per-config training RunRecords
    nccl-allreduce.SIMULATED.log    raw nccl-tests output
    nccl-allreduce.SIMULATED.json   parsed NcclResult RunRecord
    env-check.SIMULATED.txt         nvidia-smi / versions snapshot
    nvidia-smi-topo.SIMULATED.txt   topology matrix (PCIe, no NVLink)
    dcgm-dmon.SIMULATED.txt         a dcgmi dmon capture (parseable)
    reports/report.SIMULATED.{md,json}
    charts/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src/` importable when run directly from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpu_perf.models import (  # noqa: E402
    Backend, GpuTelemetry, RunConfig, RunRecord, ScalingMode, StepBreakdown, TimingStats,
)
from gpu_perf.nccl.parse import parse_nccl_output  # noqa: E402
from gpu_perf.nccl.run import parse_to_record  # noqa: E402
from gpu_perf.models import NcclOp  # noqa: E402

SAMPLES = ROOT / "evidence" / "samples"
GPU_MODEL = "NVIDIA RTX A5000 (24GB)"
VRAM_MB = 24564.0
SIM_NOTE = ("SIMULATED - not captured from real hardware. Assumed 2x NVIDIA RTX "
            "A5000 (24GB, PCIe). Provisional; replace with a real lab-box capture.")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Header note satisfies "SIMULATED marker inside the file" for JSON too.
    doc = {"_simulated": True, "_note": SIM_NOTE, **payload}
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Training RunRecords
# --------------------------------------------------------------------------- #

def _telemetry(n_gpus: int, util: float, sm: float, mem_util: float, mem_used: float):
    return [GpuTelemetry(
        gpu_index=i, gpu_util_pct=util, sm_active_pct=sm, mem_util_pct=mem_util,
        mem_used_mb=mem_used, mem_total_mb=VRAM_MB, power_w=210.0,
        power_limit_w=230.0, sample_count=50,
    ) for i in range(n_gpus)]


def _train_record(label, mode, gpus, gbs, pbs, thpt, compute, data, comm,
                  util, sm, mem_util, mem_used) -> RunRecord:
    step = gbs / thpt * 1000.0
    return RunRecord(
        run_id=f"train-{label}-SIM",
        config=RunConfig(
            model="resnet50", dataset="synthetic-imagenet", num_gpus=gpus,
            global_batch_size=gbs, per_gpu_batch_size=pbs, scaling_mode=mode,
            precision="amp", steps=50, warmup_steps=10, backend=Backend.TORCHRUN,
            label=label,
        ),
        started_at="2026-09-13T00:00:00+00:00",
        step_time=TimingStats(mean_ms=round(step, 3), median_ms=round(step, 3),
                              p90_ms=round(step * 1.05, 3), std_ms=round(step * 0.02, 3),
                              count=50),
        throughput_samples_per_s=round(thpt, 2),
        breakdown=StepBreakdown(compute_ms=compute, dataloader_ms=data, comm_ms=comm),
        telemetry=_telemetry(gpus, util, sm, mem_util, mem_used),
        gpu_model=GPU_MODEL, simulated=True, notes=SIM_NOTE,
    )


def training_records() -> list[RunRecord]:
    # Realistic ResNet-50 AMP throughput on A5000-class: ~750 img/s per GPU.
    # These come out compute-bound, which is the honest result for ResNet-50 on
    # this hardware. The classifier can identify the other three states too, but
    # we don't fabricate a comm-/memory-/input-bound ResNet-50 run just for show.
    return [
        _train_record("baseline-1gpu", ScalingMode.SINGLE, 1, 128, 128, 750, 165, 5, 0,  96, 91, 55, 9000),
        _train_record("strong-1gpu",   ScalingMode.STRONG, 1, 256, 256, 748, 335, 7, 0,  96, 91, 58, 15000),
        _train_record("strong-2gpu",   ScalingMode.STRONG, 2, 256, 128, 1370, 165, 5, 17, 93, 88, 57, 9000),
        _train_record("weak-1gpu",     ScalingMode.WEAK,   1, 128, 128, 750, 165, 5, 0,  96, 91, 55, 9000),
        _train_record("weak-2gpu",     ScalingMode.WEAK,   2, 256, 128, 1400, 165, 5, 13, 94, 89, 56, 9000),
    ]


# --------------------------------------------------------------------------- #
# NCCL log (all_reduce sweep), sized to 2x A5000 PCIe
# --------------------------------------------------------------------------- #

def nccl_log_text() -> str:
    base_lat_us = 13.0
    peak_bw = 11.6  # GB/s; PCIe-bound. all_reduce factor at n=2 is 1.0 => algbw==busbw
    rows = []
    size = 8
    while size <= 536870912:
        t_transfer = size / (peak_bw * 1000.0)
        time_us = max(base_lat_us, t_transfer)
        algbw = size / time_us / 1000.0
        rows.append((size, size // 4, time_us, algbw, algbw))
        size *= 2
    hdr = [
        f"# {SIM_NOTE}",
        "# nThread 1 nGpus 2 minBytes 8 maxBytes 536870912 step: 2(factor) warmup iters: 5 iters: 20 validation: 1",
        "#",
        "# Using devices",
        "#  Rank  0 Pid  1 on gpu-lab01 device  0 [0x01] NVIDIA RTX A5000",
        "#  Rank  1 Pid  1 on gpu-lab01 device  1 [0x02] NVIDIA RTX A5000",
        "#",
        "#                                                              out-of-place                       in-place",
        "#       size         count      type   redop     time   algbw   busbw #wrong     time   algbw   busbw #wrong",
        "#        (B)    (elements)                        (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)",
    ]
    lines = list(hdr)
    for size, count, time_us, algbw, busbw in rows:
        ip_time = time_us * 0.98
        ip_alg = size / ip_time / 1000.0
        lines.append(f"{size:>12} {count:>13} {'float':>9} {'sum':>7} "
                     f"{time_us:>8.2f} {algbw:>7.2f} {busbw:>7.2f} {'0e+00':>6} "
                     f"{ip_time:>8.2f} {ip_alg:>7.2f} {ip_alg:>7.2f} {'0e+00':>6}")
    avg = sum(r[4] for r in rows) / len(rows)
    lines += ["# Out of bounds values : 0 OK", f"# Avg bus bandwidth    : {avg:.2f}", "#"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Plain-text evidence snapshots
# --------------------------------------------------------------------------- #

def env_check_text() -> str:
    return f"""# {SIM_NOTE}
$ nvidia-smi
Sun Sep 13 00:00:00 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08   Driver Version: 535.161.08   CUDA Version: 12.2      |
|-----------------------------------------+----------------------+------------+
|   0  NVIDIA RTX A5000     Off | 00000000:01:00.0 Off |          Off |
|   1  NVIDIA RTX A5000     Off | 00000000:02:00.0 Off |          Off |
+-----------------------------------------------------------------------------+
$ python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
2.4.0  12.2  2
$ torchrun --version
torch-2.4.0
"""


def topo_text() -> str:
    return f"""# {SIM_NOTE}
$ nvidia-smi topo -m
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-15            0
GPU1    PHB      X      0-15            0

Legend:
  X   = Self
  PHB = Connection traversing PCIe as well as a PCIe Host Bridge (no NVLink)

# Note: PHB (PCIe host bridge), NOT NVLink -> inter-GPU bandwidth is PCIe-bound.
"""


def dcgm_dmon_text() -> str:
    # A short dcgmi dmon capture the parser (telemetry/dcgm.py) can read.
    lines = [f"# {SIM_NOTE}",
             "#Entity   GRACT   SMACT   DRAMA   POWER   FBUSD"]
    samples = [
        (0.93, 0.88, 0.57, 208.4, 9000), (1, 0.94, 0.89, 0.56, 210.1, 9010),
        (0.92, 0.87, 0.58, 207.8, 8990), (1, 0.93, 0.88, 0.57, 209.5, 9005),
    ]
    # interleave GPU 0 and GPU 1 rows across ticks
    ticks = [(0.93, 0.88, 0.57, 208.4, 9000), (0.94, 0.89, 0.56, 210.1, 9010),
             (0.92, 0.87, 0.58, 207.8, 8990), (0.93, 0.885, 0.575, 209.0, 9005)]
    for gract, smact, drama, power, fb in ticks:
        lines.append(f"GPU 0    {gract:.3f}   {smact:.3f}   {drama:.3f}   {power:.1f}   {fb}")
        lines.append(f"GPU 1    {gract-0.01:.3f}   {smact-0.01:.3f}   {drama+0.01:.3f}   {power-1.2:.1f}   {fb-30}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)

    # 1) training records
    trecs = training_records()
    for r in trecs:
        _write_json(SAMPLES / f"train-{r.config.label}.SIMULATED.json", r.to_dict())

    # 2) NCCL: write raw log, then parse it with the REAL parser into a RunRecord
    log = nccl_log_text()
    (SAMPLES / "nccl-allreduce.SIMULATED.log").write_text(log, encoding="utf-8")
    parse = parse_nccl_output(log, op=NcclOp.ALL_REDUCE, num_gpus=2)
    nrec = parse_to_record(parse, gpu_model=GPU_MODEL, simulated=True,
                           raw_notes="parsed_from=nccl-allreduce.SIMULATED.log")
    _write_json(SAMPLES / "nccl-allreduce.SIMULATED.json", nrec.to_dict())

    # 3) text snapshots
    (SAMPLES / "env-check.SIMULATED.txt").write_text(env_check_text(), encoding="utf-8")
    (SAMPLES / "nvidia-smi-topo.SIMULATED.txt").write_text(topo_text(), encoding="utf-8")
    (SAMPLES / "dcgm-dmon.SIMULATED.txt").write_text(dcgm_dmon_text(), encoding="utf-8")

    # 4) run the REAL report pipeline over the fixtures -> committed report + charts
    from gpu_perf.report.render import analyze, build_report_dict, load_records, render_markdown
    from gpu_perf.report import charts as charts_mod

    records = load_records([str(SAMPLES / "*.json")])
    training, nccl, comparisons = analyze(records)
    report_dir = SAMPLES / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "report.SIMULATED.json", build_report_dict(training, nccl, comparisons))
    (report_dir / "report.SIMULATED.md").write_text(
        render_markdown(training, nccl, comparisons), encoding="utf-8")

    chart_dir = SAMPLES / "charts"
    n_charts = 0
    if charts_mod.charts_available():
        n_charts = len(charts_mod.generate_charts(training, nccl, comparisons, str(chart_dir)))

    print("SIMULATED evidence generated under evidence/samples/:")
    print(f"  training records : {len(trecs)}")
    print(f"  nccl rows        : {len(nrec.nccl)} (peak busbw {parse.peak_busbw_gbps:.2f} GB/s)")
    print(f"  text snapshots   : env-check, nvidia-smi-topo, dcgm-dmon")
    print(f"  report           : reports/report.SIMULATED.md + .json")
    print(f"  charts           : {n_charts} PNG(s) in charts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
