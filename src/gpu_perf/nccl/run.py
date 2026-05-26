"""Locate, build, and run the nccl-tests binaries; turn output into a RunRecord.

Execution requires the lab box (2 GPUs + nccl-tests built locally). On the Mac
only the command-building and the parser are exercised (via a captured/SIMULATED
log). The RunRecord this produces carries ``nccl`` rows and no training fields.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .._util import hostname, make_run_id, utc_now_iso
from ..models import Backend, NcclOp, RunConfig, RunRecord, ScalingMode
from .parse import NcclParse, busbw_factor, parse_nccl_output

# nccl-tests binary name per op.
_BINARY = {
    NcclOp.ALL_REDUCE: "all_reduce_perf",
    NcclOp.ALL_GATHER: "all_gather_perf",
    NcclOp.BROADCAST: "broadcast_perf",
}


def find_binary(op: NcclOp, tests_dir: str | None = None) -> Optional[str]:
    """Resolve the nccl-tests binary path.

    Search order: explicit tests_dir -> $NCCL_TESTS_DIR/build -> PATH.
    """
    name = _BINARY[op]
    candidates: list[Path] = []
    for base in (tests_dir, os.environ.get("NCCL_TESTS_DIR")):
        if base:
            candidates += [Path(base) / name, Path(base) / "build" / name]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return shutil.which(name)


def build_command(
    op: NcclOp,
    *,
    num_gpus: int,
    min_bytes: str = "8",
    max_bytes: str = "512M",
    step_factor: int = 2,
    binary: str | None = None,
) -> list[str]:
    """Assemble the nccl-tests argv.

    -b/-e : begin/end message size (supports K/M/G suffixes)
    -f    : size step factor (geometric sweep)
    -g    : number of GPUs (single process managing all GPUs on the node)
    """
    exe = binary or _BINARY[op]
    return [exe, "-b", str(min_bytes), "-e", str(max_bytes),
            "-f", str(step_factor), "-g", str(num_gpus)]


def parse_to_record(
    parse: NcclParse,
    *,
    gpu_model: str = "",
    simulated: bool = False,
    raw_notes: str = "",
) -> RunRecord:
    """Wrap a parsed sweep into the project's backbone RunRecord."""
    n = parse.num_gpus or 0
    factor = busbw_factor(parse.op, n) if n else 0.0
    rc = RunConfig(
        num_gpus=n,
        scaling_mode=ScalingMode.SINGLE,   # NCCL sweep isn't a training scaling run
        backend=Backend.UNKNOWN,
        label=f"nccl-{parse.op.value}-{n}gpu",
    )
    notes = (f"op={parse.op.value} gpus={n} busbw_factor={factor:.3f} "
             f"peak_busbw={parse.peak_busbw_gbps:.2f}GB/s "
             f"avg_busbw={parse.avg_busbw_gbps} wrong={parse.total_wrong}")
    if raw_notes:
        notes = f"{notes} | {raw_notes}"

    return RunRecord(
        run_id=make_run_id("nccl"),
        config=rc,
        started_at=utc_now_iso(),
        nccl=parse.rows,
        hostname=hostname(),
        gpu_model=gpu_model,
        simulated=simulated,
        notes=notes,
    )


def run_nccl(
    op: NcclOp,
    *,
    num_gpus: int,
    min_bytes: str = "8",
    max_bytes: str = "512M",
    step_factor: int = 2,
    tests_dir: str | None = None,
    gpu_model: str = "",
    timeout_s: int = 600,
) -> tuple[RunRecord, str]:
    """Execute a real nccl-tests sweep (lab box) and return (record, raw_output).

    Raises FileNotFoundError if the binary can't be located.
    """
    binary = find_binary(op, tests_dir)
    if not binary:
        raise FileNotFoundError(
            f"nccl-tests binary '{_BINARY[op]}' not found. Build nccl-tests and "
            f"set $NCCL_TESTS_DIR or pass --tests-dir."
        )
    cmd = build_command(op, num_gpus=num_gpus, min_bytes=min_bytes,
                        max_bytes=max_bytes, step_factor=step_factor, binary=binary)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    raw = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    parse = parse_nccl_output(raw, op=op, num_gpus=num_gpus)
    record = parse_to_record(parse, gpu_model=gpu_model, simulated=False,
                             raw_notes=f"exit={proc.returncode}")
    return record, raw


def parse_file(path: str, *, op: NcclOp, num_gpus: int | None = None) -> tuple[RunRecord, NcclParse]:
    """Parse an existing nccl-tests log file (works anywhere, incl. Mac)."""
    text = Path(path).read_text(encoding="utf-8")
    simulated = "SIMULATED" in text or "simulated" in Path(path).name.lower()
    parse = parse_nccl_output(text, op=op, num_gpus=num_gpus)
    record = parse_to_record(parse, simulated=simulated, raw_notes=f"parsed_from={path}")
    return record, parse
