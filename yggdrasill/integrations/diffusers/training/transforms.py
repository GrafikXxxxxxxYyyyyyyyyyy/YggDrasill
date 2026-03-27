"""Image transforms for diffusion training dataset contracts."""
from __future__ import annotations

from typing import Any, Callable


def _require_torchvision() -> Any:
    try:
        from torchvision import transforms
        return transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for diffusion training image transforms. "
            "Install the diffusion extra dependencies."
        ) from exc


def build_image_transform(*, resolution: int) -> Callable[[Any], Any]:
    """Build a deterministic image transform for RGB images."""
    transforms = _require_torchvision()

    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def build_mask_transform(*, resolution: int) -> Callable[[Any], Any]:
    """Build a deterministic mask transform with values in [0, 1]."""
    transforms = _require_torchvision()

    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
