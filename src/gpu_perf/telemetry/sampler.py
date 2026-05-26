"""Background telemetry samplers used during a training run.

A sampler is started just before the measured loop and stopped just after; it
returns one aggregated ``GpuTelemetry`` per GPU. Three implementations:

  * DcgmDmonSampler : spawns `dcgmi dmon` and parses its stream (primary).
  * PynvmlSampler   : polls pynvml on a timer (fallback / Colab).
  * NoopSampler     : returns nothing (CPU-only laptop; keeps code path uniform).

``make_sampler`` picks one based on the requested source and what's available,
so the trainer never has to branch on the environment.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Optional

from ..models import GpuTelemetry
from .dcgm import aggregate, parse_dmon

# DCGM field ids to sample: GR active, SM active, DRAM active, power, FB used.
_DCGM_FIELD_IDS = "1001,1002,1005,155,252"


class BaseSampler:
    def start(self) -> None: ...
    def stop(self) -> list[GpuTelemetry]: return []


class NoopSampler(BaseSampler):
    """Used when there is no GPU (laptop) or telemetry is disabled."""

    def stop(self) -> list[GpuTelemetry]:
        return []


class PynvmlSampler(BaseSampler):
    """Polls pynvml on a background thread. Provides util/mem-util/mem/power.

    SM-active isn't exposed cleanly via pynvml, so sm_active_pct is left 0 and
    the classifier relies on gpu_util instead.
    """

    def __init__(self, gpu_indices: list[int], interval_ms: int = 200) -> None:
        self.gpu_indices = gpu_indices
        self.interval = interval_ms / 1000.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: dict[int, list[dict]] = {i: [] for i in gpu_indices}
        self._mem_total: dict[int, float] = {}

    def _loop(self) -> None:
        import pynvml
        pynvml.nvmlInit()
        handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices}
        for i, h in handles.items():
            self._mem_total[i] = pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e6
        while not self._stop.is_set():
            for i, h in handles.items():
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except pynvml.NVMLError:
                    power = 0.0
                self._samples[i].append({
                    "util": float(util.gpu),
                    "mem_util": float(util.memory),
                    "mem_used_mb": mem.used / 1e6,
                    "power": power,
                })
            self._stop.wait(self.interval)
        pynvml.nvmlShutdown()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[GpuTelemetry]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        out: list[GpuTelemetry] = []
        for i in self.gpu_indices:
            rows = self._samples.get(i, [])
            if not rows:
                continue
            n = len(rows)
            out.append(GpuTelemetry(
                gpu_index=i,
                gpu_util_pct=round(sum(r["util"] for r in rows) / n, 2),
                sm_active_pct=0.0,
                mem_util_pct=round(sum(r["mem_util"] for r in rows) / n, 2),
                mem_used_mb=round(max(r["mem_used_mb"] for r in rows), 1),
                mem_total_mb=round(self._mem_total.get(i, 0.0), 1),
                power_w=round(sum(r["power"] for r in rows) / n, 1),
                sample_count=n,
            ))
        return out


class DcgmDmonSampler(BaseSampler):
    """Spawns `dcgmi dmon` for the run window and parses the captured stream."""

    def __init__(self, gpu_indices: list[int], interval_ms: int = 200) -> None:
        self.gpu_indices = gpu_indices
        self.interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._out_lines: list[str] = []
        self._reader: Optional[threading.Thread] = None

    def start(self) -> None:
        gpu_arg = ",".join(str(i) for i in self.gpu_indices) or "-1"
        cmd = ["dcgmi", "dmon", "-e", _DCGM_FIELD_IDS,
               "-d", str(self.interval_ms), "-i", gpu_arg]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)

        def _read() -> None:
            assert self._proc and self._proc.stdout
            for line in self._proc.stdout:
                self._out_lines.append(line)

        self._reader = threading.Thread(target=_read, daemon=True)
        self._reader.start()

    def stop(self) -> list[GpuTelemetry]:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._reader:
            self._reader.join(timeout=5)
        samples = parse_dmon("".join(self._out_lines))
        return aggregate(samples)


def make_sampler(source: str, gpu_indices: list[int], interval_ms: int = 200) -> BaseSampler:
    """Choose a sampler.

    source: "dcgm" | "pynvml" | "none" | "auto".
    Falls back gracefully: dcgm -> pynvml -> noop depending on availability.
    """
    if source == "none" or not gpu_indices:
        return NoopSampler()

    dcgm_ok = shutil.which("dcgmi") is not None
    try:
        import pynvml  # noqa: F401
        pynvml_ok = True
    except ImportError:
        pynvml_ok = False

    if source in ("dcgm", "auto") and dcgm_ok:
        return DcgmDmonSampler(gpu_indices, interval_ms)
    if source in ("pynvml", "auto", "dcgm") and pynvml_ok:
        return PynvmlSampler(gpu_indices, interval_ms)
    return NoopSampler()
