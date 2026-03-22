"""Preprocess binary masks for IP-Adapter spatial conditioning (Diffusers IPAdapterMaskProcessor).

With :class:`~yggdrasill.integrations.diffusers.builder.DiffusionGraphBuilder`, adding an
IP-Adapter component auto-inserts ``ip_mask_prep`` wired to the UNet; pass
``ip_adapter_mask_images=...`` only when you want spatial masks (see
:func:`~yggdrasill.integrations.diffusers.run.run`).

Manual wiring (two reference faces, two spatial masks):

1. Add ``IPAdapterMaskPrepNode`` (``common/ip_adapter_mask_prep``). ``height`` / ``width`` should match
   the generation canvas; :func:`~yggdrasill.integrations.diffusers.run.run` copies them from
   ``latent_init`` when you omit ``width``/``height`` on ``run()`` so masks are not left at PNG size.
2. ``add_edge`` from mask prep ``ip_adapter_masks`` → UNet ``ip_adapter_masks``.
3. Expose mask images on the prep node (``expose_input(..., ip_adapter_mask_images, ...)``) or pass
   ``ip_adapter_mask_images=[m1, m2]`` to :func:`~yggdrasill.integrations.diffusers.run.run`.

Alternatively precompute with :func:`prepare_ip_adapter_masks_tensor` and pass
``ip_adapter_masks=tensor`` to ``graph.run`` / ``run_diffusion`` (routed to the UNet via ``pin_data``
when the port is not exposed).

Semantics match Hugging Face `IP-Adapter masking <https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter#masking>`_.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.common.image_utils import load_mask_image
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConverter


def _resolve_mask_image_sources(items: List[Any]) -> List[Any]:
    """Turn URL/path strings into PIL ``L`` masks; leave ndarray/tensor unchanged; normalizes PIL."""
    out: List[Any] = []
    for im in items:
        if isinstance(im, str):
            out.append(load_mask_image(im))
        elif isinstance(im, Path):
            out.append(load_mask_image(str(im)))
        elif hasattr(im, "save") and callable(getattr(im, "save", None)):
            out.append(load_mask_image(im))
        else:
            out.append(im)
    return out


def prepare_ip_adapter_masks_tensor(
    mask_images: Union[Any, Sequence[Any]],
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    """Run :class:`diffusers.image_processor.IPAdapterMaskProcessor` and pack for ``cross_attention_kwargs``.

    Returns a tensor of shape ``[1, num_masks, H, W]`` — the form expected inside
    ``cross_attention_kwargs["ip_adapter_masks"]`` as ``[tensor]`` for a single
    loaded IP-Adapter slot with multiple reference images (see Diffusers IP-Adapter masking guide).

    Args:
        mask_images: One mask or a sequence of masks (PIL, path, URL, ndarray, tensor).
        height: Output image height for resizing (recommended if mask aspect ratio ≠ canvas).
        width: Output image width for resizing.
    """
    try:
        from diffusers.image_processor import IPAdapterMaskProcessor
    except ImportError as exc:
        raise ImportError(
            "diffusers is required for IP-Adapter mask preprocessing. "
            "Install with: pip install 'yggdrasill[diffusion]'"
        ) from exc

    if isinstance(mask_images, (str, bytes)) or (
        not isinstance(mask_images, (list, tuple))
        and hasattr(mask_images, "save")
    ):
        imgs: List[Any] = [mask_images]
    elif isinstance(mask_images, (list, tuple)):
        imgs = list(mask_images)
    else:
        imgs = [mask_images]

    imgs = _resolve_mask_image_sources(imgs)

    # Match Hugging Face IP-Adapter masking examples: default ``IPAdapterMaskProcessor()`` (binary masks).
    processor = IPAdapterMaskProcessor()
    if height is not None and width is not None:
        masks = processor.preprocess(imgs, height=int(height), width=int(width))
    else:
        masks = processor.preprocess(imgs)

    if not isinstance(masks, torch.Tensor):
        masks = torch.as_tensor(masks)
    if masks.dim() != 4:
        raise ValueError(
            f"IPAdapterMaskProcessor.preprocess expected a 4D tensor [N, C, H, W], got shape {tuple(masks.shape)}"
        )
    # Match HF docs: [N, C, H, W] -> [1, N, H, W] (channel dim dropped like official example).
    packed = masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])
    return packed


class IPAdapterMaskPrepNode(AbstractConverter):
    """Converter: binary mask images → packed tensor for UNet ``ip_adapter_masks``."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=dict(config or {}))

    @property
    def block_type(self) -> str:
        return "common/ip_adapter_mask_prep"

    def declare_ports(self) -> List[Port]:
        return [
            Port(
                C.PORT_IP_ADAPTER_MASK_IMAGES,
                PortDirection.IN,
                PortType.IMAGE,
                optional=True,
            ),
            Port(C.PORT_IP_ADAPTER_MASKS, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raw = inputs.get(C.PORT_IP_ADAPTER_MASK_IMAGES)
        if raw is None:
            return {C.PORT_IP_ADAPTER_MASKS: None}
        h = self._config.get("height")
        w = self._config.get("width")
        packed = prepare_ip_adapter_masks_tensor(
            raw,
            height=int(h) if h is not None else None,
            width=int(w) if w is not None else None,
        )
        return {C.PORT_IP_ADAPTER_MASKS: packed}
