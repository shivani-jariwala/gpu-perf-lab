"""Parser for nccl-tests output (pure text -> NcclResult rows).

This module has no GPU dependency, so it runs and is fully testable on the Mac
against captured or SIMULATED logs. The lab-box wrapper (run.py) feeds it real
output; the fixture generator (Phase 5) feeds it synthetic output. Same parser.

nccl-tests table shape (one collective, swept over message sizes):

    #   size   count  type  redop  [root]   time  algbw  busbw #wrong   time  algbw  busbw #wrong
    #    (B) (elems)                        (us) (GB/s) (GB/s)         (us) (GB/s) (GB/s)
        8       2   float   sum          14.20   0.00   0.00  0e+00   13.9   0.00   0.00 0e+00
      ...
    # Avg bus bandwidth : 11.83

There are always two perf groups on a data row — OUT-OF-PLACE then IN-PLACE,
four columns each (time, algbw, busbw, #wrong). We report the out-of-place
group (the conventional headline) and take the size/count from the front.

algbw vs busbw (the concept to defend):
  * algbw = message_size / time  -> bandwidth from the ALGORITHM's viewpoint.
  * busbw = algbw * factor        -> reflects ACTUAL bytes on the interconnect,
    so it's comparable to hardware peak (NVLink/PCIe) and roughly independent of
    GPU count. For a ring all-reduce the factor is 2*(n-1)/n; note that at n=2
    it equals 1.0, so algbw == busbw on a 2-GPU node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models import NcclOp, NcclResult


def busbw_factor(op: NcclOp, num_gpus: int) -> float:
    """The collective-specific factor relating busbw to algbw (ring algorithm)."""
    n = max(1, num_gpus)
    if op is NcclOp.ALL_REDUCE:
        return 2.0 * (n - 1) / n
    if op in (NcclOp.ALL_GATHER,):
        return (n - 1) / n
    if op is NcclOp.BROADCAST:
        return 1.0
    return 1.0


def _to_float(tok: str) -> Optional[float]:
    try:
        v = float(tok)
    except (TypeError, ValueError):
        return None
    # nccl prints "nan"/"inf" for degenerate rows; treat as missing.
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


@dataclass
class NcclParse:
    """Result of parsing one nccl-tests run."""

    op: NcclOp
    num_gpus: int
    rows: list[NcclResult]
    avg_busbw_gbps: Optional[float] = None   # from the "# Avg bus bandwidth" footer
    out_of_bounds: Optional[int] = None      # from "# Out of bounds values : N"

    @property
    def peak_busbw_gbps(self) -> float:
        return max((r.busbw_gbps for r in self.rows), default=0.0)

    @property
    def total_wrong(self) -> int:
        return sum(r.wrong_count for r in self.rows)


_NGPUS_RE = re.compile(r"nGpus\s+(\d+)")
_AVG_RE = re.compile(r"Avg bus bandwidth\s*:\s*([-\d.eE+]+)")
_OOB_RE = re.compile(r"Out of bounds values\s*:\s*(\d+)")


def parse_nccl_output(
    text: str,
    *,
    op: NcclOp = NcclOp.ALL_REDUCE,
    num_gpus: Optional[int] = None,
) -> NcclParse:
    """Parse raw nccl-tests stdout into structured rows.

    op/num_gpus: the parser can read nGpus from the header if present; otherwise
    it uses the passed value. op must be supplied (the table doesn't name it).
    """
    rows: list[NcclResult] = []
    avg_busbw: Optional[float] = None
    oob: Optional[int] = None
    header_gpus: Optional[int] = None

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            m = _NGPUS_RE.search(s)
            if m:
                header_gpus = int(m.group(1))
            m = _AVG_RE.search(s)
            if m:
                avg_busbw = _to_float(m.group(1))
            m = _OOB_RE.search(s)
            if m:
                oob = int(m.group(1))
            continue

        toks = s.split()
        # A data row starts with two integers (size in bytes, element count) and
        # carries two 4-column perf groups at the end (>= 10 tokens total).
        if len(toks) < 10 or not (toks[0].isdigit() and toks[1].isdigit()):
            continue

        size = int(toks[0])
        oop = toks[-8:-4]  # out-of-place group: [time_us, algbw, busbw, #wrong]
        time_us = _to_float(oop[0])
        algbw = _to_float(oop[1])
        busbw = _to_float(oop[2])
        wrong_f = _to_float(oop[3])

        rows.append(NcclResult(
            op=op,
            size_bytes=size,
            time_us=round(time_us, 3) if time_us is not None else 0.0,
            algbw_gbps=round(algbw, 4) if algbw is not None else 0.0,
            busbw_gbps=round(busbw, 4) if busbw is not None else 0.0,
            num_gpus=(num_gpus or header_gpus or 0),
            wrong_count=int(wrong_f) if wrong_f is not None else 0,
        ))

    resolved_gpus = num_gpus or header_gpus or 0
    for r in rows:
        if r.num_gpus == 0:
            r.num_gpus = resolved_gpus

    return NcclParse(
        op=op, num_gpus=resolved_gpus, rows=rows,
        avg_busbw_gbps=avg_busbw, out_of_bounds=oob,
    )
