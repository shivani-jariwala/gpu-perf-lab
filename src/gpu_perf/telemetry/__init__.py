"""Telemetry sampling during a run.

Primary source is DCGM (`dcgmi dmon`) — the same telemetry backbone as the
first two projects in the series — with a pynvml/nvidia-smi fallback for boxes
(or Colab) where DCGM isn't running. Sampling happens WHILE training runs, so
the aggregated per-GPU utilization / memory / power lines up with the timing
breakdown and drives the bottleneck classification in Phase 4.
"""
