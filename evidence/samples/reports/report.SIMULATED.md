# GPU Performance Report

> **SIMULATED DATA** — generated from synthetic fixtures, not a real hardware capture. Numbers are provisional pending a lab-box run.

## Runs and bottleneck classification

| label | gpus | global_bs | throughput (smp/s) | step (ms) | compute/data/comm | bottleneck |
|---|---:|---:|---:|---:|:--:|---|
| baseline-1gpu | 1 | 128 | 750.0 | 170.67 | 97%/3%/0% | **compute** |
| strong-1gpu | 1 | 256 | 748.0 | 342.25 | 98%/2%/0% | **compute** |
| strong-2gpu | 2 | 256 | 1370.0 | 186.86 | 88%/3%/9% | **compute** |
| weak-1gpu | 1 | 128 | 750.0 | 170.67 | 97%/3%/0% | **compute** |
| weak-2gpu | 2 | 256 | 1400.0 | 182.86 | 90%/3%/7% | **compute** |

- `baseline-1gpu` — GPU util 96% (>= 90%), comm 0% / input 3% low; compute-bound
- `strong-1gpu` — GPU util 96% (>= 90%), comm 0% / input 2% low; compute-bound
- `strong-2gpu` — GPU util 93% (>= 90%), comm 9% / input 3% low; compute-bound
- `weak-1gpu` — GPU util 96% (>= 90%), comm 0% / input 3% low; compute-bound
- `weak-2gpu` — GPU util 94% (>= 90%), comm 7% / input 3% low; compute-bound

## Strong scaling (model+dataset fixed)

| gpus | global_bs | per_gpu | throughput | speedup | efficiency | bottleneck |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 256 | 256 | 748.0 | 1.00 | 1.00 | compute |
| 2 | 256 | 128 | 1370.0 | 1.83 | 0.92 | compute |

## Weak scaling (model+dataset fixed)

| gpus | global_bs | per_gpu | throughput | speedup | efficiency | bottleneck |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 128 | 128 | 750.0 | 1.00 | 1.00 | compute |
| 2 | 256 | 128 | 1400.0 | 1.87 | 0.93 | compute |

## NCCL microbenchmarks (context)

| collective | gpus | peak busbw (GB/s) | sizes | #wrong |
|---|---:|---:|---:|---:|
| all_reduce | 2 | 11.60 | 27 | 0 |
