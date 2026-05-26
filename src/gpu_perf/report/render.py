"""Load RunRecords, classify, compare, and render console / JSON / Markdown."""

from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

from ..models import Bottleneck, RunRecord
from .compare import build_comparisons, split_records
from .correlate import classify_in_place


def load_records(inputs: list[str]) -> list[RunRecord]:
    """Load RunRecords from files and/or globs. Skips non-RunRecord JSON."""
    paths: list[str] = []
    for item in inputs:
        matched = glob.glob(item)
        paths.extend(matched if matched else [item])

    records: list[RunRecord] = []
    for p in sorted(set(paths)):
        fp = Path(p)
        if not fp.is_file() or fp.suffix != ".json":
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            if "run_id" in d and "config" in d:
                records.append(RunRecord.from_dict(d))
        except (ValueError, KeyError):
            continue
    return records


def analyze(records: list[RunRecord], thresholds: dict | None = None):
    """Classify training runs in place; return (training, nccl, comparisons)."""
    training, nccl = split_records(records)
    for r in training:
        classify_in_place(r, thresholds)
    comparisons = build_comparisons(training)
    return training, nccl, comparisons


# --------------------------------------------------------------------------- #
# Console
# --------------------------------------------------------------------------- #

def render_console(training, nccl, comparisons) -> str:
    lines: list[str] = []
    sim = any(r.simulated for r in (training + nccl))
    lines.append("=" * 78)
    lines.append("GPU Performance Report" + ("   [SIMULATED DATA]" if sim else ""))
    lines.append("=" * 78)

    lines.append("\nRuns (bottleneck classification):")
    lines.append(f"  {'label':<16}{'gpus':>5}{'gbs':>6}{'thpt(smp/s)':>13}"
                 f"{'step(ms)':>10}  {'compute/data/comm':<20}{'bottleneck':<15}")
    for r in sorted(training, key=lambda x: (x.config.scaling_mode.value, x.config.num_gpus)):
        sh = r.breakdown.share()
        share_str = f"{sh['compute']:.0%}/{sh['dataloader']:.0%}/{sh['comm']:.0%}"
        lines.append(
            f"  {r.config.label:<16}{r.config.num_gpus:>5}{r.config.global_batch_size:>6}"
            f"{r.throughput_samples_per_s:>13.1f}{r.step_time.mean_ms:>10.2f}  "
            f"{share_str:<20}{r.bottleneck.value:<15}")
    for r in training:
        if r.bottleneck_reason:
            lines.append(f"    - {r.config.label}: {r.bottleneck_reason}")

    for comp in comparisons:
        lines.append(f"\n{comp.title}:")
        lines.append(f"  {'gpus':>5}{'global_bs':>11}{'per_gpu':>9}"
                     f"{'thpt':>12}{'speedup':>9}{'efficiency':>12}{'  bottleneck':<14}")
        for row in comp.speedup_efficiency():
            lines.append(
                f"  {row['num_gpus']:>5}{row['global_batch_size']:>11}{row['per_gpu_batch_size']:>9}"
                f"{row['throughput_samples_per_s']:>12.1f}{row['speedup']:>9.2f}"
                f"{row['efficiency']:>12.2f}  {row['bottleneck']:<14}")

    if nccl:
        lines.append("\nNCCL microbenchmarks (context):")
        for r in nccl:
            peak = max((x.busbw_gbps for x in r.nccl), default=0.0)
            op = r.nccl[0].op.value if r.nccl else "?"
            lines.append(f"  {op:<12} gpus={r.config.num_gpus}  peak busbw={peak:.2f} GB/s  "
                         f"rows={len(r.nccl)}  wrong={sum(x.wrong_count for x in r.nccl)}")

    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# JSON + Markdown
# --------------------------------------------------------------------------- #

def build_report_dict(training, nccl, comparisons) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulated": any(r.simulated for r in (training + nccl)),
        "runs": [r.to_dict() for r in training],
        "comparisons": [c.to_dict() for c in comparisons],
        "nccl": [r.to_dict() for r in nccl],
    }


def render_markdown(training, nccl, comparisons) -> str:
    sim = any(r.simulated for r in (training + nccl))
    md: list[str] = []
    md.append("# GPU Performance Report")
    if sim:
        md.append("\n> **SIMULATED DATA** — generated from synthetic fixtures, not a real "
                  "hardware capture. Numbers are provisional pending a lab-box run.")
    md.append("\n## Runs and bottleneck classification\n")
    md.append("| label | gpus | global_bs | throughput (smp/s) | step (ms) | compute/data/comm | bottleneck |")
    md.append("|---|---:|---:|---:|---:|:--:|---|")
    for r in sorted(training, key=lambda x: (x.config.scaling_mode.value, x.config.num_gpus)):
        sh = r.breakdown.share()
        md.append(f"| {r.config.label} | {r.config.num_gpus} | {r.config.global_batch_size} "
                  f"| {r.throughput_samples_per_s:.1f} | {r.step_time.mean_ms:.2f} "
                  f"| {sh['compute']:.0%}/{sh['dataloader']:.0%}/{sh['comm']:.0%} "
                  f"| **{r.bottleneck.value}** |")
    md.append("")
    for r in training:
        if r.bottleneck_reason:
            md.append(f"- `{r.config.label}` — {r.bottleneck_reason}")

    for comp in comparisons:
        md.append(f"\n## {comp.title}\n")
        md.append("| gpus | global_bs | per_gpu | throughput | speedup | efficiency | bottleneck |")
        md.append("|---:|---:|---:|---:|---:|---:|---|")
        for row in comp.speedup_efficiency():
            md.append(f"| {row['num_gpus']} | {row['global_batch_size']} | {row['per_gpu_batch_size']} "
                      f"| {row['throughput_samples_per_s']:.1f} | {row['speedup']:.2f} "
                      f"| {row['efficiency']:.2f} | {row['bottleneck']} |")

    if nccl:
        md.append("\n## NCCL microbenchmarks (context)\n")
        md.append("| collective | gpus | peak busbw (GB/s) | sizes | #wrong |")
        md.append("|---|---:|---:|---:|---:|")
        for r in nccl:
            peak = max((x.busbw_gbps for x in r.nccl), default=0.0)
            op = r.nccl[0].op.value if r.nccl else "?"
            md.append(f"| {op} | {r.config.num_gpus} | {peak:.2f} | {len(r.nccl)} "
                      f"| {sum(x.wrong_count for x in r.nccl)} |")
    md.append("")
    return "\n".join(md)
