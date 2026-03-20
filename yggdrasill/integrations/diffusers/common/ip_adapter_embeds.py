"""Align IP-Adapter image embeddings with diffusers UNet expectations.

See ``prepare_ip_adapter_image_embeds`` in
``diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion`` and
``MultiIPAdapterImageProjection.forward`` in ``diffusers.models.embeddings``:
each tensor must be shaped ``(batch_size, num_images, embed_dim)`` (or 4D with a
sequence dim); with classifier-free guidance, batch is ``2 *`` the conditioning
batch, ordered ``[unconditional, conditional]``.
"""
from __future__ import annotations

from typing import Any, List, Optional, Union

import torch


def unet_requires_image_embeds_in_added_cond(unet: Any) -> bool:
    """True when diffusers UNet expects ``added_cond_kwargs[\"image_embeds\"]`` (IP / Kandinsky-style)."""
    cfg = getattr(unet, "config", None)
    if cfg is None or getattr(unet, "encoder_hid_proj", None) is None:
        return False
    t = getattr(cfg, "encoder_hid_dim_type", None)
    return t in ("text_image_proj", "image_proj", "ip_image_proj")


def _first_linear_in_features(module: Any) -> int | None:
    import torch.nn as nn

    for m in module.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    return None


def raw_zero_ip_adapter_image_embeds_for_unet(
    unet: Any,
    cond_batch_size: int,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> List[torch.Tensor]:
    """Raw (B, D) zeros per IP-Adapter slot; pass through :func:`format_ip_adapter_image_embeds`.

    Zeros use **float32** by default so they match typical CLIP vision ``image_embeds``;
    diffusers ``prepare_ip_adapter_image_embeds`` ends with ``.to(device)`` and does not
    cast IP tensors to the UNet weight dtype.
    """
    proj = getattr(unet, "encoder_hid_proj", None)
    if proj is None:
        return []
    layers = getattr(proj, "image_projection_layers", None)
    if layers is not None and len(layers) > 0:
        n_adapters = len(layers)
        dim = _first_linear_in_features(layers[0]) or 1024
    else:
        n_adapters = 1
        dim = _first_linear_in_features(proj) or 1024
    zdt = dtype if dtype is not None else torch.float32
    return [
        torch.zeros(cond_batch_size, dim, device=device, dtype=zdt)
        for _ in range(n_adapters)
    ]


def format_ip_adapter_image_embeds(
    image_embeds: Union[torch.Tensor, List[torch.Tensor]],
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    do_classifier_free_guidance: bool,
) -> List[torch.Tensor]:
    """Return ``image_embeds`` as a list (one entry per IP-Adapter projection layer).

    If *dtype* is ``None`` (default), tensors are moved with ``.to(device)`` only — same
    idea as diffusers ``prepare_ip_adapter_image_embeds`` (vision outputs often stay
    float32 while the UNet runs in fp16). Forcing fp16 here can destabilize IP-Adapter.
    """

    def _one(single: torch.Tensor) -> torch.Tensor:
        e = single.to(device=device)
        if dtype is not None:
            e = e.to(dtype=dtype)
        if e.ndim == 2:
            e = e.unsqueeze(1)
        elif e.ndim not in (3, 4):
            raise ValueError(
                f"IP-Adapter image_embeds must be 2D–4D, got shape {tuple(e.shape)}"
            )
        if do_classifier_free_guidance:
            e = torch.cat([torch.zeros_like(e), e], dim=0)
        return e

    if isinstance(image_embeds, list):
        return [_one(x) for x in image_embeds]
    return [_one(image_embeds)]
