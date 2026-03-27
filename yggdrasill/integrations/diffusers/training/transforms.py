"""Image transforms for the minimal diffusion training dataset contract."""
from __future__ import annotations

from typing import Any, Callable


def build_image_transform(*, resolution: int) -> Callable[[Any], Any]:
    """Build a deterministic image transform for SD1.5 training."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for diffusion training image transforms. "
            "Install the diffusion extra dependencies."
        ) from exc

    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
