"""The FIXED dataset: synthetic ImageNet-shaped data.

Why synthetic:
  * No multi-GB download; deterministic and reproducible anywhere.
  * We can DELIBERATELY control the input pipeline. A real "input-pipeline-bound"
    run is one where the GPU starves waiting for data. With ``input_delay_ms`` we
    can inject per-sample host-side work to reproduce that regime on demand —
    which is what makes the "identify input-pipeline-bound runs" claim true and
    demonstrable, not hypothetical.

Each sample is a random CHW float tensor plus a random class label. Generation
itself is light CPU work (a stand-in for decode/augment); ``input_delay_ms``
adds a tunable stall on top.
"""

from __future__ import annotations

import time


def build_dataset(
    *,
    image_size: int = 224,
    channels: int = 3,
    num_classes: int = 1000,
    length: int = 100_000,
    input_delay_ms: float = 0.0,
):
    """Construct the synthetic dataset (imported lazily so the module loads without torch)."""
    import torch
    from torch.utils.data import Dataset

    class SyntheticImageNet(Dataset):
        def __init__(self) -> None:
            self.image_size = image_size
            self.channels = channels
            self.num_classes = num_classes
            self.length = length
            self.input_delay_ms = input_delay_ms

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int):
            # Deterministic per-index so runs are reproducible across ranks/epochs.
            g = torch.Generator().manual_seed(idx)
            x = torch.randn(self.channels, self.image_size, self.image_size, generator=g)
            y = torch.randint(0, self.num_classes, (1,), generator=g).item()
            if self.input_delay_ms > 0:
                # Simulate a slow input pipeline (decode/augment/IO) to create an
                # input-pipeline-bound run on demand.
                time.sleep(self.input_delay_ms / 1000.0)
            return x, y

    return SyntheticImageNet()


def build_dataloader(
    *,
    per_gpu_batch_size: int,
    num_workers: int = 4,
    image_size: int = 224,
    channels: int = 3,
    num_classes: int = 1000,
    input_delay_ms: float = 0.0,
    distributed: bool = False,
):
    """Build a DataLoader (+ DistributedSampler when running multi-rank).

    With DDP, each rank must see a DISJOINT slice of the data so the effective
    global batch is per_gpu_batch * world_size. DistributedSampler handles that
    partitioning.
    """
    import torch
    from torch.utils.data import DataLoader

    # pin_memory only helps (and is only supported) for CUDA H2D copies.
    pin = torch.cuda.is_available()

    ds = build_dataset(
        image_size=image_size, channels=channels, num_classes=num_classes,
        input_delay_ms=input_delay_ms,
    )

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, shuffle=True)

    loader = DataLoader(
        ds,
        batch_size=per_gpu_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,                # faster H2D copy when CUDA is present
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    return loader, sampler
