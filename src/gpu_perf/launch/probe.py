"""A minimal distributed smoke target: `python -m gpu_perf.launch.probe`.

Its only job is to prove the launcher wired up a distributed environment
correctly BEFORE the real trainer (Phase 2) exists:

  * Reads the rank/world topology torchrun (or Slurm+torchrun) injected via
    environment variables (RANK, LOCAL_RANK, WORLD_SIZE).
  * If torch is installed, initializes the process group (NCCL on CUDA, else
    Gloo), runs one tiny all-reduce, and confirms every rank agrees on the sum.
  * If torch is absent (e.g. on the Mac), it just prints the env-derived
    topology so the launcher can still be exercised end-to-end.

On the lab box under 2 GPUs you should see two lines (local ranks 0 and 1) and
an all-reduce result of world_size (each rank contributes 1.0).
"""

from __future__ import annotations

import os


def _env_topology() -> dict[str, str]:
    keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
            "SLURM_JOB_ID", "SLURM_NODENAME")
    return {k: os.environ.get(k, "") for k in keys}


def main() -> int:
    topo = _env_topology()
    rank = topo.get("RANK") or "0"
    local_rank = topo.get("LOCAL_RANK") or "0"
    world = topo.get("WORLD_SIZE") or "1"
    prefix = f"[probe rank={rank} local_rank={local_rank} world={world}]"

    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print(f"{prefix} torch not installed — env-only probe. "
              f"master={topo.get('MASTER_ADDR')}:{topo.get('MASTER_PORT')} "
              f"slurm_job={topo.get('SLURM_JOB_ID') or '-'}")
        return 0

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    # torchrun sets everything we need; only init if launched distributed.
    is_distributed = int(world) > 1 or "RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend=backend)
        rank_i = dist.get_rank()
        world_i = dist.get_world_size()
    else:
        rank_i, world_i = 0, 1

    if use_cuda:
        torch.cuda.set_device(int(local_rank))
        dev = torch.device("cuda", int(local_rank))
        gpu_name = torch.cuda.get_device_name(int(local_rank))
    else:
        dev = torch.device("cpu")
        gpu_name = "cpu"

    # Each rank contributes 1.0; after all-reduce every rank should read world_i.
    x = torch.ones(1, device=dev)
    if is_distributed:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

    ok = abs(x.item() - world_i) < 1e-6
    print(f"[probe rank={rank_i}/{world_i} local_rank={local_rank}] "
          f"backend={backend} device={gpu_name} "
          f"all_reduce(ones)={x.item():.1f} expected={world_i} "
          f"{'OK' if ok else 'MISMATCH'}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
