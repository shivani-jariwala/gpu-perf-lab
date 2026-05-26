"""Build (and optionally submit) the launch command for one experiment.

Single-node, up to 2 GPUs. Both backends ultimately run the SAME per-process
entrypoint via torchrun's elastic launcher:

  * torchrun path : run torchrun directly on the node.
  * slurm path    : write an sbatch script that requests the GPUs and runs the
                    exact same torchrun command INSIDE the allocation, then
                    submit it with `sbatch`.

Keeping torchrun as the inner launcher in both cases means the training code
sees an identical distributed environment (RANK/LOCAL_RANK/WORLD_SIZE), so
"Slurm vs torchrun" is purely about *scheduling*, not about how ranks come up.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Backend, RunConfig
from .detect import detect_backend, probe_backend

# Default per-process entrypoint. This module is added in Phase 2; until then
# the launcher can still build/print the command and run the built-in probe.
DEFAULT_TARGET = "gpu_perf.train.ddp_train"

# Built-in distributed smoke target — lets us validate the launcher wiring
# (rank/world spawn + optional NCCL all-reduce) before the real trainer exists.
PROBE_TARGET = "gpu_perf.launch.probe"


# The sbatch template. Rendered per-experiment. Uses --standalone torchrun since
# this is single-node; MASTER_ADDR/PORT are handled by torchrun's rendezvous.
SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={log_dir}/{job_name}-%j.out
#SBATCH --error={log_dir}/{job_name}-%j.err
# NOTE: partition/time/account are site-specific — add #SBATCH lines as your
# lab box requires (e.g. #SBATCH --partition=gpu --time=00:30:00).

set -euo pipefail
echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID gpus={num_gpus}"

# Same inner launcher as the torchrun path, inside the Slurm allocation:
{torchrun_cmd}
"""


@dataclass
class LaunchPlan:
    """Everything needed to describe, print, or execute a launch."""

    backend: Backend
    num_gpus: int
    argv: list[str]                      # the command actually executed
    torchrun_cmd: str = ""               # the inner torchrun command (string form)
    sbatch_path: str | None = None       # written sbatch script, if slurm
    sbatch_text: str = ""                # rendered sbatch content, if slurm
    reason: str = ""                     # why this backend was chosen
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return " ".join(self.argv)


def _torchrun_argv(num_gpus: int, target: str, target_args: list[str]) -> list[str]:
    """Build the canonical torchrun command as an argv list.

    --standalone : single-node rendezvous (no external etcd / master addr wiring)
    --nnodes=1   : we are honestly single-node
    -m           : interpret `target` as a python module (so we can run our package)
    """
    return [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        "-m", target,
        *target_args,
    ]


def build_plan(
    run_config: RunConfig,
    *,
    target: str = DEFAULT_TARGET,
    target_args: list[str] | None = None,
    prefer_slurm: bool = True,
    log_dir: str = "evidence/logs",
    cpus_per_task: int = 8,
) -> LaunchPlan:
    """Construct a LaunchPlan for one experiment (no execution)."""
    target_args = list(target_args or [])
    num_gpus = max(1, run_config.num_gpus)
    status = probe_backend(prefer_slurm)
    backend = status.chosen

    torchrun_argv = _torchrun_argv(num_gpus, target, target_args)
    torchrun_cmd = " ".join(torchrun_argv)

    notes: list[str] = []
    if not status.torchrun_available:
        notes.append("torchrun not on PATH here — dry-run only; install torch on the GPU box to execute.")

    if backend is Backend.SLURM:
        job_name = f"gpuperf-{run_config.label or f'{num_gpus}gpu'}"
        sbatch_text = SBATCH_TEMPLATE.format(
            job_name=job_name,
            num_gpus=num_gpus,
            cpus=cpus_per_task,
            log_dir=log_dir,
            torchrun_cmd=torchrun_cmd,
        )
        return LaunchPlan(
            backend=backend, num_gpus=num_gpus,
            argv=["sbatch", f"<{job_name}.sbatch>"],  # placeholder until written at submit time
            torchrun_cmd=torchrun_cmd,
            sbatch_text=sbatch_text,
            reason=status.reason(),
            notes=notes,
        )

    # torchrun path
    return LaunchPlan(
        backend=backend, num_gpus=num_gpus,
        argv=torchrun_argv,
        torchrun_cmd=torchrun_cmd,
        reason=status.reason(),
        notes=notes,
    )


def submit(plan: LaunchPlan, *, log_dir: str = "evidence/logs") -> int:
    """Execute the plan for real. Returns the child process exit code.

    Slurm path : write the rendered sbatch file, then `sbatch` it.
    torchrun    : exec torchrun in the foreground (streams training output).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if plan.backend is Backend.SLURM:
        # Name the script after the job for traceability.
        script = Path(log_dir) / (plan.argv[1].strip("<>"))
        script.write_text(plan.sbatch_text, encoding="utf-8")
        script.chmod(0o755)
        plan.sbatch_path = str(script)
        cmd = ["sbatch", str(script)]
        proc = subprocess.run(cmd)
        return proc.returncode

    # torchrun: replace the placeholder console-script with the current
    # interpreter's module launcher if `torchrun` isn't directly on PATH.
    argv = list(plan.argv)
    import shutil as _sh
    if _sh.which("torchrun") is None:
        # torchrun == `python -m torch.distributed.run`
        argv = [sys.executable, "-m", "torch.distributed.run", *argv[1:]]
    proc = subprocess.run(argv, env={**os.environ})
    return proc.returncode
