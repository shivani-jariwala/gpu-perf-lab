"""Parser for `dcgmi dmon` streaming output (pure text -> per-GPU samples).

Like the NCCL parser, this has no GPU dependency so it's testable on the Mac.
`dcgmi dmon -e <fields> -d <ms>` prints a header naming the fields, then one row
per entity per tick:

    #Entity   GRACT   SMACT   DRAMA   POWER   FBUSD
    GPU 0     0.912   0.884   0.421   241.3   18342
    GPU 1     0.905   0.878   0.418   239.8   18310
    GPU 0     0.918   0.889   0.430   243.1   18355
    ...

Field short-names we care about (DCGM):
    GRACT  graphics/GR engine active     (0..1)  -> gpu utilization
    SMACT  SM active                     (0..1)  -> finer compute occupancy
    DRAMA  memory (DRAM) active/bandwidth (0..1)  -> memory-bandwidth pressure
    POWER  board power draw              (W)
    FBUSD  framebuffer (memory) used     (MiB)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import GpuTelemetry

# Map DCGM dmon short-names to our GpuTelemetry concepts. Values in 0..1 are
# scaled to percent on aggregation.
_FRACTION_FIELDS = {"GRACT", "SMACT", "DRAMA", "GRACT%", "SMACT%"}


@dataclass
class DmonSamples:
    """Raw per-GPU samples keyed by gpu index; each value is a list of dicts."""

    fields: list[str]
    per_gpu: dict[int, list[dict[str, float]]] = field(default_factory=dict)

    def gpu_indices(self) -> list[int]:
        return sorted(self.per_gpu.keys())


def parse_dmon(text: str) -> DmonSamples:
    """Parse dcgmi dmon output into structured samples.

    Detects the header (line containing 'Entity' and field names) to learn the
    column order, then reads 'GPU <idx> <vals...>' rows.
    """
    fields: list[str] = []
    per_gpu: dict[int, list[dict[str, float]]] = {}

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # Header: contains 'Entity' and the field short-names.
        if "Entity" in s:
            toks = s.lstrip("#").split()
            # drop the leading 'Entity' label; the rest are field names
            fields = [t for t in toks if t.lower() != "entity"]
            continue
        # A comment/units line without data.
        if s.startswith("#"):
            continue
        toks = s.split()
        # Data rows look like: GPU 0 v1 v2 ...
        if len(toks) < 3 or toks[0].upper() != "GPU" or not toks[1].isdigit():
            continue
        idx = int(toks[1])
        vals = toks[2:]
        if not fields or len(vals) < len(fields):
            # Fall back to positional generic names if header was missing/short.
            fields = fields or [f"f{i}" for i in range(len(vals))]
        sample: dict[str, float] = {}
        for name, tok in zip(fields, vals):
            try:
                sample[name] = float(tok)
            except ValueError:
                continue
        per_gpu.setdefault(idx, []).append(sample)

    return DmonSamples(fields=fields, per_gpu=per_gpu)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _get(samples: list[dict[str, float]], key: str) -> list[float]:
    return [s[key] for s in samples if key in s]


def _scale_pct(vals: list[float], name: str) -> float:
    """Return mean as a percent. Fraction fields (0..1) are ×100."""
    m = _mean(vals)
    return round(m * 100.0, 2) if name in _FRACTION_FIELDS else round(m, 2)


def aggregate(samples: DmonSamples, mem_total_mb: float = 0.0) -> list[GpuTelemetry]:
    """Collapse a stream of samples into one GpuTelemetry per GPU (mean/peak)."""
    out: list[GpuTelemetry] = []
    for idx in samples.gpu_indices():
        rows = samples.per_gpu[idx]
        fb = _get(rows, "FBUSD")
        out.append(GpuTelemetry(
            gpu_index=idx,
            gpu_util_pct=_scale_pct(_get(rows, "GRACT"), "GRACT"),
            sm_active_pct=_scale_pct(_get(rows, "SMACT"), "SMACT"),
            mem_util_pct=_scale_pct(_get(rows, "DRAMA"), "DRAMA"),
            mem_used_mb=round(max(fb), 1) if fb else 0.0,      # peak framebuffer use
            mem_total_mb=mem_total_mb,
            power_w=round(_mean(_get(rows, "POWER")), 1),
            sample_count=len(rows),
        ))
    return out
