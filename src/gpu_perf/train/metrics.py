"""Timing helpers for the training loop.

Timing GPU work correctly is subtle: CUDA kernels are launched *asynchronously*,
so a plain ``time.perf_counter()`` around ``loss.backward()`` measures how long
the Python call took to *enqueue* work, not how long the GPU actually ran. The
correct tool is CUDA events, which are recorded on the stream and measured after
a synchronize.

``DeviceTimer`` wraps that: on CUDA it uses ``torch.cuda.Event``; on CPU it
falls back to ``perf_counter`` so the exact same code runs (slowly) on a laptop
for validation.
"""

from __future__ import annotations

import time
from typing import Optional


class DeviceTimer:
    """Context manager timing a region in milliseconds, GPU-accurate on CUDA.

        with DeviceTimer(device) as t:
            out = model(x); loss.backward(); ...
        ms = t.ms

    On CUDA it records start/stop events and synchronizes on exit (so the
    number reflects real kernel time). On CPU it uses a wall-clock timer.
    """

    def __init__(self, device) -> None:
        self.device = device
        self.is_cuda = getattr(device, "type", str(device)) == "cuda"
        self.ms: float = 0.0
        self._t0: float = 0.0
        self._start = None
        self._stop = None
        if self.is_cuda:
            import torch
            self._start = torch.cuda.Event(enable_timing=True)
            self._stop = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "DeviceTimer":
        if self.is_cuda:
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.is_cuda:
            import torch
            self._stop.record()
            torch.cuda.synchronize(self.device)
            self.ms = self._start.elapsed_time(self._stop)  # already milliseconds
        else:
            self.ms = (time.perf_counter() - self._t0) * 1000.0


class HostTimer:
    """Plain wall-clock timer in ms, for host-side waits (e.g. dataloader).

    We time the dataloader on the HOST because a stall there means the GPU is
    sitting idle waiting for input — that's the input-pipeline signal.
    """

    def __init__(self) -> None:
        self.ms: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "HostTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
