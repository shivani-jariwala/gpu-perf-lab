"""Phase 1 — the launcher.

Prefers Slurm (`sbatch`) when it is installed on the node, and falls back to
`torchrun` (torch.distributed elastic launcher) when it is not. The chosen path
is always reported so a run never silently implies Slurm when torchrun was used.
"""
