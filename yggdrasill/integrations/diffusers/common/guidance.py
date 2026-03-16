"""Classifier-free guidance utilities."""
from __future__ import annotations

from typing import Any


def apply_cfg(
    noise_pred_uncond: Any,
    noise_pred_cond: Any,
    guidance_scale: float,
    guidance_rescale: float = 0.0,
) -> Any:
    """Apply classifier-free guidance with optional rescale (SDXL)."""

    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    if guidance_rescale > 0.0:
        std_pos = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
        std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
        factor = std_pos / (std_cfg + 1e-8)
        noise_pred = guidance_rescale * (noise_pred * factor) + (1.0 - guidance_rescale) * noise_pred

    return noise_pred
