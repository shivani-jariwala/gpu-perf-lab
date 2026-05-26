"""Instrumented single-node DDP training run: `python -m gpu_perf.train.ddp_train`.

This is the entrypoint the launcher targets. Under torchrun it runs one process
per GPU; each process trains ResNet-50 on synthetic data for a fixed number of
steps and times where each step goes. Rank 0 aggregates the measurements into a
``RunRecord`` (the project backbone) and writes it as JSON.

Measurement design (the part worth understanding):
  * dataloader_ms : HOST wall-time spent waiting for the next batch + H2D copy.
                    A large value here means the GPU is starved -> input-bound.
  * compute region: forward + backward + optimizer, timed with CUDA events so
                    the async GPU work is measured correctly.
  * comm_ms       : DDP gradient all-reduce cost. DDP OVERLAPS all-reduce with
                    the backward pass, so it can't be read off a single timer.
                    We isolate the EXPOSED (non-overlapped) comm by comparing
                    backward time with all-reduce ON vs OFF (DDP's no_sync()):
                        comm_ms ~= mean(backward_sync) - mean(backward_no_sync)
                    On 1 GPU there is no all-reduce, so comm_ms == 0.
  * compute_ms    : compute region minus the exposed comm (pure GPU compute).

Runs on CUDA (real target) or CPU+gloo (for laptop validation) unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .._util import hostname, make_run_id, utc_now_iso
from ..config import load_config
from ..models import (
    Backend, Bottleneck, GpuTelemetry, RunConfig, RunRecord, ScalingMode,
    StepBreakdown, TimingStats,
)
from .metrics import DeviceTimer, HostTimer


# --------------------------------------------------------------------------- #
# Distributed setup
# --------------------------------------------------------------------------- #

def _dist_env() -> tuple[int, int, int]:
    """Read torchrun-injected topology; default to single-process if unset."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world


def _setup(world: int):
    import torch
    import torch.distributed as dist

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return use_cuda, backend


def _resolve_backend() -> Backend:
    """Which launcher backend is this process running under (for provenance)."""
    if os.environ.get("SLURM_JOB_ID"):
        return Backend.SLURM
    if os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("RANK"):
        return Backend.TORCHRUN
    return Backend.UNKNOWN


# --------------------------------------------------------------------------- #
# Config resolution (single source of truth = experiments.yaml)
# --------------------------------------------------------------------------- #

def _resolve_run_config(args) -> RunConfig:
    """Build the RunConfig from --label in experiments.yaml, applying overrides."""
    if args.label:
        cfg = load_config(args.config)
        matches = [rc for rc in cfg.experiments() if rc.label == args.label]
        if not matches:
            raise SystemExit(f"error: no experiment labelled '{args.label}' in {cfg.source_path}")
        rc = matches[0]
    else:
        # Ad-hoc run (no config) — used mainly for laptop smoke tests.
        rc = RunConfig(scaling_mode=ScalingMode.SINGLE, label="adhoc")

    # CLI overrides (handy for a fast CPU smoke test).
    if args.per_gpu_batch is not None:
        rc.per_gpu_batch_size = args.per_gpu_batch
    if args.steps is not None:
        rc.steps = args.steps
    if args.warmup is not None:
        rc.warmup_steps = args.warmup
    if rc.per_gpu_batch_size <= 0:
        rc.per_gpu_batch_size = 32
    return rc


# --------------------------------------------------------------------------- #
# The measured training
# --------------------------------------------------------------------------- #

def _calibrate_comm(model, criterion, x, y, device, use_amp, scaler, iters: int) -> float:
    """Estimate exposed all-reduce time via backward with sync ON vs OFF.

    Returns comm_ms >= 0. Only meaningful when world_size > 1 (DDP wraps model).
    """
    import torch

    def _backward_once(sync: bool) -> float:
        ctx = model.no_sync() if (not sync and hasattr(model, "no_sync")) else _null_ctx()
        with ctx:
            with DeviceTimer(device) as t:
                with _autocast(device, use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            model.zero_grad(set_to_none=True)
        return t.ms

    # A couple of warmups so the two measurements are on equal footing.
    for _ in range(2):
        _backward_once(sync=True)

    sync_ms = sum(_backward_once(sync=True) for _ in range(iters)) / iters
    nosync_ms = sum(_backward_once(sync=False) for _ in range(iters)) / iters
    return max(0.0, sync_ms - nosync_ms)


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def _autocast(device, use_amp: bool):
    import torch
    if use_amp and getattr(device, "type", "") == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return _null_ctx()


def train(args) -> int:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank, local_rank, world = _dist_env()
    use_cuda, backend = _setup(world)
    rc = _resolve_run_config(args)
    rc.num_gpus = world
    rc.backend = _resolve_backend()
    # Keep global batch consistent with the actual world size.
    rc.global_batch_size = rc.per_gpu_batch_size * world

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cudnn.benchmark = True   # autotune convolutions for fixed shapes
    else:
        device = torch.device("cpu")

    is_dist = world > 1
    use_amp = (rc.precision == "amp") and use_cuda

    # --- build model + DDP ---------------------------------------------------
    from .model import build_model, parameter_count
    model = build_model(rc.model, num_classes=1000).to(device)
    n_params = parameter_count(model)
    if is_dist:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # --- data ----------------------------------------------------------------
    from .data import build_dataloader
    loader, sampler = build_dataloader(
        per_gpu_batch_size=rc.per_gpu_batch_size,
        num_workers=args.num_workers,
        input_delay_ms=args.input_delay_ms,
        distributed=is_dist,
    )
    if sampler is not None:
        sampler.set_epoch(0)
    data_iter = iter(loader)

    def _next_batch():
        nonlocal data_iter
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        return xb, yb

    model.train()

    # --- telemetry sampler (rank 0 covers the whole node) --------------------
    from ..telemetry.sampler import make_sampler
    sampler = None
    if rank == 0 and use_cuda:
        gpu_indices = list(range(min(world, torch.cuda.device_count())))
        sampler = make_sampler(args.telemetry, gpu_indices, interval_ms=args.telemetry_interval_ms)

    # --- warmup (discarded) --------------------------------------------------
    for _ in range(rc.warmup_steps):
        xb, yb = _next_batch()
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            loss = criterion(model(xb), yb)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
    if use_cuda:
        torch.cuda.synchronize(device)

    # --- comm calibration (only meaningful multi-GPU) ------------------------
    comm_ms = 0.0
    if is_dist:
        xb, yb = _next_batch()
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        comm_ms = _calibrate_comm(model, criterion, xb, yb, device, use_amp, scaler,
                                  iters=max(3, args.comm_calib_steps))

    # --- measured loop -------------------------------------------------------
    step_ms_samples: list[float] = []
    data_ms_samples: list[float] = []
    region_ms_samples: list[float] = []

    if sampler is not None:
        sampler.start()

    for _ in range(rc.steps):
        step_t = HostTimer()
        with step_t:
            with HostTimer() as data_t:
                xb, yb = _next_batch()
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
            with DeviceTimer(device) as region_t:
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, use_amp):
                    out = model(xb)
                    loss = criterion(out, yb)
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
        step_ms_samples.append(step_t.ms)
        data_ms_samples.append(data_t.ms)
        region_ms_samples.append(region_t.ms)

    sampled_telemetry = sampler.stop() if sampler is not None else []

    # --- aggregate -----------------------------------------------------------
    step_stats = TimingStats.from_samples(step_ms_samples)
    mean_region = sum(region_ms_samples) / len(region_ms_samples)
    mean_data = sum(data_ms_samples) / len(data_ms_samples)
    compute_ms = max(0.0, mean_region - comm_ms)
    breakdown = StepBreakdown(
        compute_ms=round(compute_ms, 4),
        dataloader_ms=round(mean_data, 4),
        comm_ms=round(comm_ms, 4),
    )

    # Throughput uses the SLOWEST rank's mean step (all-reduce max) so the global
    # number reflects the real synchronized pace, not the fastest rank.
    mean_step_s = step_stats.mean_ms / 1000.0
    if is_dist:
        t = torch.tensor([mean_step_s], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        mean_step_s = float(t.item())
    throughput = (rc.global_batch_size / mean_step_s) if mean_step_s > 0 else 0.0

    # Telemetry: prefer the DCGM/pynvml samples captured during the loop; fall
    # back to a torch memory-only entry if no sampler was active.
    telemetry = sampled_telemetry
    if use_cuda and not telemetry:
        _, total_b = torch.cuda.mem_get_info(local_rank)
        telemetry = [GpuTelemetry(
            gpu_index=local_rank,
            mem_used_mb=round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
            mem_total_mb=round(total_b / 1e6, 1),
            sample_count=1,
        )]
    # Backfill device memory capacity if the sampler didn't provide it.
    if use_cuda:
        _, total_b = torch.cuda.mem_get_info(local_rank)
        for t in telemetry:
            if t.mem_total_mb <= 0:
                t.mem_total_mb = round(total_b / 1e6, 1)

    record = RunRecord(
        run_id=make_run_id("train"),
        config=rc,
        started_at=utc_now_iso(),
        step_time=step_stats,
        throughput_samples_per_s=round(throughput, 2),
        breakdown=breakdown,
        telemetry=telemetry,
        bottleneck=Bottleneck.UNKNOWN,   # classified in Phase 4
        hostname=hostname(),
        gpu_model=(torch.cuda.get_device_name(local_rank) if use_cuda else "cpu"),
        simulated=False,
        notes=(f"params={n_params/1e6:.1f}M backend={backend} amp={use_amp} "
               f"world={world} num_workers={args.num_workers} "
               f"input_delay_ms={args.input_delay_ms}"),
    )

    # --- rank 0 writes + prints ----------------------------------------------
    if rank == 0:
        out_path = Path(args.out or f"evidence/logs/train-{rc.label}-{world}gpu.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        share = breakdown.share()
        print(f"[rank0] {rc.label}: world={world} global_bs={rc.global_batch_size} "
              f"step={step_stats.mean_ms:.2f}ms thpt={throughput:.1f} samp/s")
        print(f"[rank0] breakdown ms: compute={breakdown.compute_ms} "
              f"dataloader={breakdown.dataloader_ms} comm={breakdown.comm_ms} "
              f"(share compute={share['compute']:.0%} data={share['dataloader']:.0%} "
              f"comm={share['comm']:.0%})")
        print(f"[rank0] wrote {out_path}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Instrumented DDP training run (ResNet-50 / synthetic).")
    p.add_argument("--config", default=None, help="Path to experiments.yaml")
    p.add_argument("--label", default=None, help="Experiment label from experiments.yaml")
    p.add_argument("--per-gpu-batch", type=int, default=None, help="Override per-GPU batch size")
    p.add_argument("--steps", type=int, default=None, help="Override measured steps")
    p.add_argument("--warmup", type=int, default=None, help="Override warmup steps")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per rank")
    p.add_argument("--input-delay-ms", type=float, default=0.0,
                   help="Per-sample host stall to simulate a slow input pipeline")
    p.add_argument("--comm-calib-steps", type=int, default=5,
                   help="Iterations for the comm (all-reduce) calibration")
    p.add_argument("--telemetry", default="auto",
                   choices=["auto", "dcgm", "pynvml", "none"],
                   help="Telemetry source sampled during the run (rank 0).")
    p.add_argument("--telemetry-interval-ms", type=int, default=200,
                   help="Telemetry sampling interval.")
    p.add_argument("--out", default=None, help="Where to write the RunRecord JSON")
    return p


def main() -> int:
    return train(build_argparser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
