"""Optional AnimateDiff helpers (FreeNoise-style latents) aligned with diffusers conventions."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def apply_free_noise_latents(
    *,
    batch_size: int,
    num_channels: int,
    num_frames: int,
    height_latent: int,
    width_latent: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator],
    config: Any,
) -> torch.Tensor:
    """Build initial 5D latents ``(B,C,F,H,W)`` when ``animatediff_free_noise`` is set in *config*.

    Mirrors diffusers ``AnimateDiffFreeNoiseMixin._prepare_latents_free_noise`` for latent-space
    shapes (``height_latent`` / ``width_latent`` are already VAE-scaled).
    """
    try:
        from diffusers.utils.torch_utils import randn_tensor
    except ImportError:  # pragma: no cover
        randn_tensor = None

    noise_type = str(config.get("animatediff_free_noise_noise_type", "random")).lower()
    ctx_raw = config.get("animatediff_free_noise_context_length", num_frames)
    stride = int(config.get("animatediff_free_noise_context_stride", 4))

    shape = (batch_size, num_channels, num_frames, height_latent, width_latent)

    if randn_tensor is not None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    if noise_type == "random":
        return latents

    if noise_type == "repeat_context":
        ctx_len = int(ctx_raw) if isinstance(ctx_raw, (int, float)) else 4
        ctx_len = max(1, ctx_len)
        small_shape = (batch_size, num_channels, ctx_len, height_latent, width_latent)
        if randn_tensor is not None:
            base = randn_tensor(small_shape, generator=generator, device=device, dtype=dtype)
        else:
            base = torch.randn(small_shape, generator=generator, device=device, dtype=dtype)
        num_repeats = (num_frames + ctx_len - 1) // ctx_len
        latents = torch.cat([base] * num_repeats, dim=2)
        return latents[:, :, :num_frames]

    if noise_type != "shuffle_context":
        return latents

    ctx_len = int(ctx_raw) if isinstance(ctx_raw, (int, float)) else 4
    ctx_len = max(1, min(ctx_len, num_frames))
    stride = max(1, stride)

    for i in range(ctx_len, num_frames, stride):
        window_start = max(0, i - ctx_len)
        window_end = min(num_frames, window_start + stride)
        window_length = window_end - window_start
        if window_length == 0:
            break
        indices = torch.arange(window_start, window_end, device=device, dtype=torch.long)
        shuffled_indices = indices[torch.randperm(window_length, generator=generator, device=device)]
        current_start = i
        current_end = min(num_frames, current_start + window_length)
        if current_end == current_start + window_length:
            latents[:, :, current_start:current_end] = latents[:, :, shuffled_indices]
        else:
            prefix_length = current_end - current_start
            shuffled_indices = shuffled_indices[:prefix_length]
            latents[:, :, current_start:current_end] = latents[:, :, shuffled_indices]

    return latents


def free_init_trim_timesteps(
    scheduler: Any,
    num_inference_steps: int,
    *,
    num_iters: int,
) -> Tuple[Any, int]:
    """Placeholder for FreeInit outer-loop bookkeeping (re-run denoise with fresh scheduler)."""
    _ = num_iters
    return scheduler, num_inference_steps
