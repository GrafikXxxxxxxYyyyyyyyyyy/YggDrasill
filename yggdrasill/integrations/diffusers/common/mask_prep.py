"""Inpainting mask preparation — shared by SD1.5, SDXL, and FLUX."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConverter


class InpaintMaskPrepNode(AbstractConverter):
    """Prepares mask and masked-image latents for inpainting.

    Supports both 9-channel (concat) and 4-channel (blend) UNet variants.
    Shared by SD1.5, SDXL, and FLUX.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        vae: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._vae = vae

    @property
    def block_type(self) -> str:
        return "common/mask_prep"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_INIT_IMAGE, PortDirection.IN, PortType.IMAGE, optional=True),
            Port(C.PORT_MASK_IMAGE, PortDirection.IN, PortType.IMAGE, optional=True),
            Port(C.PORT_MASK_LATENTS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_MASKED_IMAGE_LATENTS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_CLEAN_IMAGE_LATENTS, PortDirection.OUT, PortType.TENSOR, optional=True),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from yggdrasill.integrations.diffusers.common.image_utils import (
            preprocess_image,
            preprocess_mask,
        )

        image = inputs.get(C.PORT_INIT_IMAGE)
        mask = inputs.get(C.PORT_MASK_IMAGE)
        if image is None:
            raise RuntimeError(
                "common/mask_prep ran without init_image; executor should skip this node "
                "for text2img (universal SD1.5 graph)."
            )
        height = self._config.get("height", 512)
        width = self._config.get("width", 512)
        device = self._config.get("device", "cpu")
        dtype = getattr(self._vae, "dtype", None) if self._vae else None

        image_tensor = preprocess_image(image, height=height, width=width, dtype=dtype, device=device)
        if mask is None:
            mask_tensor = torch.ones(
                image_tensor.shape[0],
                1,
                image_tensor.shape[2],
                image_tensor.shape[3],
                device=image_tensor.device,
                dtype=image_tensor.dtype,
            )
        else:
            mask_tensor = preprocess_mask(mask, height=height, width=width, dtype=dtype, device=device)

        masked_image = image_tensor * (mask_tensor < 0.5)

        if self._vae is not None:
            with torch.no_grad():
                clean_latents = self._vae.encode(image_tensor).latent_dist.sample()
                masked_latents = self._vae.encode(masked_image).latent_dist.sample()
            scaling = getattr(self._vae.config, "scaling_factor", 0.18215)
            shift = getattr(self._vae.config, "shift_factor", 0.0) or 0.0
            clean_latents = (clean_latents - shift) * scaling
            masked_latents = (masked_latents - shift) * scaling
        else:
            clean_latents = image_tensor
            masked_latents = masked_image

        latent_h, latent_w = height // 8, width // 8
        if mask_tensor.shape[-2:] != (latent_h, latent_w):
            import torch.nn.functional as F
            mask_tensor = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="nearest")

        # Strict 0/1 at latent resolution avoids soft-edge blending (halos / ghost outlines) in the
        # 4-channel UNet path, which composites with (1 - mask) * ref + mask * denoised.
        mask_tensor = (mask_tensor >= 0.5).to(dtype=mask_tensor.dtype)

        if self._config.get("pack_latents_2x2"):
            from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode

            num_latent_channels = int(
                getattr(clean_latents, "shape", [1, self._config.get("num_latent_channels", 16)])[1]
            )
            mask_tensor = mask_tensor.repeat(1, num_latent_channels, 1, 1)
            clean_latents = FluxLatentInitNode._pack_latents(clean_latents)
            masked_latents = FluxLatentInitNode._pack_latents(masked_latents)
            mask_tensor = FluxLatentInitNode._pack_latents(mask_tensor)

        return {
            C.PORT_MASK_LATENTS: mask_tensor,
            C.PORT_MASKED_IMAGE_LATENTS: masked_latents,
            C.PORT_CLEAN_IMAGE_LATENTS: clean_latents,
        }

    def to(self, device: Any) -> "InpaintMaskPrepNode":
        if self._vae is not None:
            self._vae.to(device)
        return self
