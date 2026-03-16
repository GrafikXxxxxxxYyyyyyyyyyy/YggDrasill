"""FLUX VAE encode/decode nodes (16 latent channels, 2x2 patch un/packing)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConverter


class FluxVAEEncodeNode(AbstractConverter):
    """Encodes images to FLUX latent space (16 channels)."""

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
        return "flux/vae_encode"

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

        scaling = getattr(self._vae.config, "scaling_factor", 0.3611)
        shift = getattr(self._vae.config, "shift_factor", 0.1159)
        latents = (latents - shift) * scaling

        return {C.PORT_LATENTS: latents}

    def to(self, device: Any) -> "FluxVAEEncodeNode":
        if self._vae is not None:
            self._vae.to(device)
        return self


class FluxVAEDecodeNode(AbstractConverter):
    """Decodes FLUX packed latents back to pixel space.

    Unpacks latents from [B, num_patches, C*4] -> [B, C, H, W] before
    VAE decoding, applying FLUX-specific scaling/shift inversion.
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
        return "flux/vae_decode"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PACKED_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_DECODED_IMAGE, PortDirection.OUT, PortType.IMAGE),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from yggdrasill.integrations.diffusers.common.image_utils import postprocess_image

        packed = inputs[C.PORT_PACKED_LATENTS]
        output_type = self._config.get("output_type", "pil")

        if output_type == "latent":
            return {C.PORT_DECODED_IMAGE: packed}

        height = self._config.get("height", 1024)
        width = self._config.get("width", 1024)
        latent_h = height // 8
        latent_w = width // 8
        vae_channels = self._config.get("num_latent_channels", 16)

        latents = self._unpack_latents(packed, latent_h, latent_w, vae_channels)

        scaling = getattr(self._vae.config, "scaling_factor", 0.3611)
        shift = getattr(self._vae.config, "shift_factor", 0.1159)
        latents = latents / scaling + shift

        with torch.no_grad():
            image = self._vae.decode(latents, return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        return {C.PORT_DECODED_IMAGE: postprocess_image(image, output_type=output_type)}

    @staticmethod
    def _unpack_latents(packed: Any, h: int, w: int, channels: int) -> Any:
        """Unpack [B, (H/2)*(W/2), C*4] -> [B, C, H, W]."""
        b = packed.shape[0]
        packed = packed.reshape(b, h // 2, w // 2, channels, 2, 2)
        packed = packed.permute(0, 3, 1, 4, 2, 5)
        latents = packed.reshape(b, channels, h, w)
        return latents

    def to(self, device: Any) -> "FluxVAEDecodeNode":
        if self._vae is not None:
            self._vae.to(device)
        return self
