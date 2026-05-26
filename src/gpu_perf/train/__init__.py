"""Phase 2 — the instrumented DDP training harness.

ResNet-50 on synthetic ImageNet-shaped data (fixed model + dataset). The loop
separately times where each step's wall-clock goes — dataloader wait, GPU
compute, and DDP communication — because that breakdown is what lets Phase 4
classify a run as compute-/memory-/input-pipeline-/comm-bound.
"""
