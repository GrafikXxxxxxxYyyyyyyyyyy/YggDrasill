"""SDXL VAE encode/decode nodes (handles SDXL-specific scaling)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConverter


class SDXLVAEEncodeNode(AbstractConverter):
    """Encodes images to SDXL latent space."""

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
        return "sdxl/vae_encode"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_INIT_IMAGE, PortDirection.IN, PortType.IMAGE),
            Port(C.PORT_LATENTS, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from yggdrasill.integrations.diffusers.common.image_utils import preprocess_image

        image = inputs[C.PORT_INIT_IMAGE]
        dtype = getattr(self._vae, "dtype", None)
        device = self._config.get("device", "cpu")

        pixel_values = preprocess_image(
            image,
            height=self._config.get("height", 1024),
            width=self._config.get("width", 1024),
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            latents = self._vae.encode(pixel_values).latent_dist.sample()

        scaling = getattr(self._vae.config, "scaling_factor", 0.13025)
        shift = getattr(self._vae.config, "shift_factor", None)
        if shift is not None:
            latents = (latents - shift) * scaling
        else:
            latents = latents * scaling

        return {C.PORT_LATENTS: latents}

    def to(self, device: Any) -> "SDXLVAEEncodeNode":
        if self._vae is not None:
            self._vae.to(device)
        return self


class SDXLVAEDecodeNode(AbstractConverter):
    """Decodes SDXL latents back to pixel space."""

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
        return "sdxl/vae_decode"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_DECODED_IMAGE, PortDirection.OUT, PortType.IMAGE),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from yggdrasill.integrations.diffusers.common.image_utils import postprocess_image

        latents = inputs[C.PORT_LATENTS]
        output_type = self._config.get("output_type", "pil")

        if output_type == "latent":
            return {C.PORT_DECODED_IMAGE: latents}

        scaling = getattr(self._vae.config, "scaling_factor", 0.13025)
        shift = getattr(self._vae.config, "shift_factor", None)
        if shift is not None:
            latents = latents / scaling + shift
        else:
            latents = latents / scaling

        with torch.no_grad():
            image = self._vae.decode(latents, return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        return {C.PORT_DECODED_IMAGE: postprocess_image(image, output_type=output_type)}

    def to(self, device: Any) -> "SDXLVAEDecodeNode":
        if self._vae is not None:
            self._vae.to(device)
        return self
