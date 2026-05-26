"""Chart generation for the comparison report (matplotlib, optional [viz] extra).

Kept separate and import-guarded so the core report (console/JSON/Markdown) has
no hard dependency on matplotlib. Charts are generated from the same RunRecords,
so they work identically for SIMULATED and real data.
"""

from __future__ import annotations

from pathlib import Path


def charts_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def generate_charts(training, nccl, comparisons, out_dir: str) -> list[str]:
    """Write PNG charts to out_dir; return the list of written paths.

    Returns [] (no error) if matplotlib isn't installed.
    """
    if not charts_available():
        return []

    import matplotlib
    matplotlib.use("Agg")  # headless — safe on servers and CI
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    sim = any(r.simulated for r in (training + nccl))
    tag = " [SIMULATED]" if sim else ""

    # 1) Speedup vs ideal, per scaling comparison.
    for comp in comparisons:
        rows = comp.speedup_efficiency()
        if len(rows) < 2:
            continue
        gpus = [r["num_gpus"] for r in rows]
        speedup = [r["speedup"] for r in rows]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(gpus, gpus, "--", color="gray", label="ideal (linear)")
        ax.plot(gpus, speedup, "o-", color="#76b900", label="measured")  # NVIDIA green
        ax.set_xlabel("GPUs"); ax.set_ylabel("speedup vs 1 GPU")
        ax.set_title(f"{comp.scaling_mode.value.capitalize()} scaling: speedup{tag}")
        ax.set_xticks(gpus); ax.legend(); ax.grid(True, alpha=0.3)
        fp = out / f"speedup-{comp.scaling_mode.value}.png"
        fig.tight_layout(); fig.savefig(fp, dpi=120); plt.close(fig)
        written.append(str(fp))

    # 2) Step-time breakdown (stacked) across all training runs.
    if training:
        runs = sorted(training, key=lambda r: (r.config.scaling_mode.value, r.config.num_gpus))
        labels = [r.config.label for r in runs]
        compute = [r.breakdown.compute_ms for r in runs]
        data = [r.breakdown.dataloader_ms for r in runs]
        comm = [r.breakdown.comm_ms for r in runs]
        fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.3), 4))
        ax.bar(labels, compute, label="compute", color="#76b900")
        ax.bar(labels, data, bottom=compute, label="dataloader", color="#f2a900")
        bottom2 = [c + d for c, d in zip(compute, data)]
        ax.bar(labels, comm, bottom=bottom2, label="comm", color="#0071c5")
        ax.set_ylabel("mean step time (ms)")
        ax.set_title(f"Step-time breakdown per run{tag}")
        ax.legend(); plt.xticks(rotation=30, ha="right"); ax.grid(True, axis="y", alpha=0.3)
        fp = out / "step-breakdown.png"
        fig.tight_layout(); fig.savefig(fp, dpi=120); plt.close(fig)
        written.append(str(fp))

    # 3) NCCL busbw vs message size (log-x).
    for r in nccl:
        if not r.nccl:
            continue
        sizes = [x.size_bytes for x in r.nccl]
        busbw = [x.busbw_gbps for x in r.nccl]
        op = r.nccl[0].op.value
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(sizes, busbw, "o-", color="#76b900")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("message size (bytes)"); ax.set_ylabel("busbw (GB/s)")
        ax.set_title(f"NCCL {op} bus bandwidth ({r.config.num_gpus} GPU){tag}")
        ax.grid(True, alpha=0.3)
        fp = out / f"nccl-busbw-{op}.png"
        fig.tight_layout(); fig.savefig(fp, dpi=120); plt.close(fig)
        written.append(str(fp))

    return written
