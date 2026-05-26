"""Backend detection: is Slurm here, or do we fall back to torchrun?

Detection is deliberately cheap and side-effect-free (just looks for binaries),
so it runs anywhere — including the Mac, where it correctly reports "torchrun"
because there is no `sbatch`.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from ..models import Backend


def slurm_available() -> bool:
    """True only if BOTH `sbatch` and `sinfo` are on PATH.

    We require both because a box can have stray Slurm client bits without a
    working scheduler; needing `sinfo` too is a cheap sanity gate.
    """
    return shutil.which("sbatch") is not None and shutil.which("sinfo") is not None


def torchrun_available() -> bool:
    """True if the `torchrun` console script is on PATH.

    (Even without it, `python -m torch.distributed.run` works when torch is
    installed; we treat the `torchrun` shim as the canonical check.)
    """
    return shutil.which("torchrun") is not None


def detect_backend(prefer_slurm: bool = True) -> Backend:
    """Pick the launch backend.

    prefer_slurm=True (default): Slurm if available, else torchrun.
    prefer_slurm=False: force torchrun (useful to compare, or on Colab).
    """
    if prefer_slurm and slurm_available():
        return Backend.SLURM
    return Backend.TORCHRUN


@dataclass
class BackendStatus:
    """A human-readable snapshot of what's available, for `info`/`launch` output."""

    chosen: Backend
    slurm_available: bool
    torchrun_available: bool
    slurm_partitions: str = ""   # best-effort `sinfo` summary, empty if none

    def reason(self) -> str:
        if self.chosen is Backend.SLURM:
            return "sbatch + sinfo found on PATH"
        if self.slurm_available:
            return "Slurm available but torchrun was forced (--no-prefer-slurm)"
        return "sbatch not found -> falling back to torchrun"


def probe_backend(prefer_slurm: bool = True) -> BackendStatus:
    """Full detection snapshot, including a best-effort Slurm partition list."""
    slurm = slurm_available()
    partitions = ""
    if slurm:
        try:
            # `sinfo -h -o %P` -> partition names, one per line. Short timeout so
            # a wedged scheduler can't hang the CLI.
            out = subprocess.run(
                ["sinfo", "-h", "-o", "%P"],
                capture_output=True, text=True, timeout=5,
            )
            partitions = " ".join(sorted(set(out.stdout.split())))
        except (subprocess.SubprocessError, OSError):
            partitions = "(sinfo query failed)"
    return BackendStatus(
        chosen=detect_backend(prefer_slurm),
        slurm_available=slurm,
        torchrun_available=torchrun_available(),
        slurm_partitions=partitions,
    )
