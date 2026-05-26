"""Loader for experiments.yaml.

Kept intentionally thin: it reads the YAML into a small typed wrapper so the
rest of the code accesses config via attributes/helpers instead of poking at
raw dicts. The FIXED model/dataset and the experiment matrix (GPU counts,
batch sizes, scaling modes) live in YAML — not in code — so you can re-run a
different comparison without editing Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import Backend, RunConfig, ScalingMode

_DEFAULT_NAMES = ("experiments.yaml", "experiments.yml")


@dataclass
class Config:
    """Typed-ish wrapper around the parsed YAML document."""

    raw: dict[str, Any]
    source_path: Path

    # --- fixed model / dataset ------------------------------------------------
    @property
    def model(self) -> str:
        return (self.raw.get("model") or {}).get("name", "resnet50")

    @property
    def dataset(self) -> str:
        return (self.raw.get("dataset") or {}).get("name", "synthetic-imagenet")

    @property
    def precision(self) -> str:
        return self.raw.get("precision", "amp")

    # --- assumed hardware (drives SIMULATED sizing until real re-baseline) ----
    @property
    def hardware(self) -> dict[str, Any]:
        return self.raw.get("hardware", {}) or {}

    @property
    def gpu_model(self) -> str:
        return self.hardware.get("gpu_model", "UNKNOWN GPU")

    # --- telemetry ------------------------------------------------------------
    @property
    def telemetry(self) -> dict[str, Any]:
        return self.raw.get("telemetry", {}) or {}

    # --- nccl -----------------------------------------------------------------
    @property
    def nccl(self) -> dict[str, Any]:
        return self.raw.get("nccl", {}) or {}

    # --- bottleneck-classifier thresholds (provisional) -----------------------
    @property
    def thresholds(self) -> dict[str, Any]:
        return self.raw.get("thresholds", {}) or {}

    # --- run bookkeeping ------------------------------------------------------
    @property
    def steps(self) -> int:
        return int(self.raw.get("run", {}).get("steps", 50))

    @property
    def warmup_steps(self) -> int:
        return int(self.raw.get("run", {}).get("warmup_steps", 10))

    def experiments(self) -> list[RunConfig]:
        """Materialize the experiment matrix into concrete RunConfig objects.

        Each experiment entry names a scaling_mode + num_gpus + one batch knob.
        We derive the other batch value so both global and per-GPU are always
        populated on the RunConfig:
            strong: global fixed  -> per_gpu = global / num_gpus
            weak:   per_gpu fixed  -> global  = per_gpu * num_gpus
            single: whichever is given
        """
        out: list[RunConfig] = []
        for e in (self.raw.get("experiments", []) or []):
            mode = ScalingMode(e.get("scaling_mode", "single"))
            n = int(e.get("num_gpus", 1))
            gbs = e.get("global_batch_size")
            pbs = e.get("per_gpu_batch_size")

            if mode is ScalingMode.STRONG:
                if gbs is None:
                    raise ValueError(f"strong experiment '{e.get('label')}' needs global_batch_size")
                pbs = int(gbs) // n
            elif mode is ScalingMode.WEAK:
                if pbs is None:
                    raise ValueError(f"weak experiment '{e.get('label')}' needs per_gpu_batch_size")
                gbs = int(pbs) * n
            else:  # single
                if gbs is None and pbs is not None:
                    gbs = int(pbs) * n
                elif pbs is None and gbs is not None:
                    pbs = int(gbs) // n

            out.append(RunConfig(
                model=self.model,
                dataset=self.dataset,
                num_gpus=n,
                global_batch_size=int(gbs or 0),
                per_gpu_batch_size=int(pbs or 0),
                scaling_mode=mode,
                precision=self.precision,
                steps=self.steps,
                warmup_steps=self.warmup_steps,
                backend=Backend.UNKNOWN,
                label=e.get("label", f"{mode.value}-{n}gpu"),
            ))
        return out


def find_config(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve the config path.

    Order: explicit arg -> $GPU_PERF_CONFIG -> ./config/experiments.yaml ->
    ./experiments.yaml. Raises if nothing is found.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Config not found: {p}")
        return p

    env = os.environ.get("GPU_PERF_CONFIG")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"$GPU_PERF_CONFIG points to missing file: {p}")
        return p

    for d in (Path.cwd() / "config", Path.cwd()):
        for name in _DEFAULT_NAMES:
            candidate = d / name
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "Could not locate experiments.yaml. Pass --config, set $GPU_PERF_CONFIG, "
        "or run from the project root."
    )


def load_config(explicit: str | os.PathLike | None = None) -> Config:
    path = find_config(explicit)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}: {path}")
    return Config(raw=raw, source_path=path)
