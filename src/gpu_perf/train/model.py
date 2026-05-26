"""The FIXED model: ResNet-50.

Held constant across every run so the only variables are GPU count and batch
size. ResNet-50 is a recognizable standard benchmark and heavy enough that DDP
gradient all-reduce (~25.5M parameters => ~97 MB of gradients in fp32) is a
real, measurable cost — which is exactly what makes the "communication
overhead" story concrete.
"""

from __future__ import annotations


def build_model(name: str = "resnet50", num_classes: int = 1000):
    """Return an (uninitialized-weights) model on CPU; caller moves it to device.

    We use torchvision's reference implementation so the architecture is exactly
    the standard one an interviewer would expect. Weights are random — we're
    measuring throughput, not training to accuracy.
    """
    try:
        import torchvision.models as tvm
    except ImportError as exc:  # pragma: no cover - only on GPU/Colab boxes
        raise ImportError(
            "torchvision is required for the model. Install with "
            "`pip install -e \".[gpu]\"` on the GPU box."
        ) from exc

    if name != "resnet50":
        # Fixed by design; guard against silent drift in the config.
        factory = getattr(tvm, name, None)
        if factory is None:
            raise ValueError(f"unknown model '{name}'")
        return factory(num_classes=num_classes)

    return tvm.resnet50(weights=None, num_classes=num_classes)


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())
