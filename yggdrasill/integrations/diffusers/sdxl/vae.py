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
        device = self._config.get("device", "cpu")
        p = next(self._vae.parameters(), None)
        vae_dtype = getattr(self._vae, "dtype", None)
        if p is not None and vae_dtype is None:
            vae_dtype = p.dtype

        # Match StableDiffusionXLImg2ImgPipeline.prepare_latents: fp16 VAE encode overflows;
        # temporarily run encoder weights and pixels in float32 when config.force_upcast.
        force_upcast = bool(getattr(self._vae.config, "force_upcast", False))
        prep_dtype = torch.float32 if force_upcast else vae_dtype

        pixel_values = preprocess_image(
            image,
            height=self._config.get("height", 1024),
            width=self._config.get("width", 1024),
            dtype=prep_dtype,
            device=device,
        )

        orig_vae_dtype = vae_dtype
        did_fp32_encode = False
        if force_upcast and p is not None and vae_dtype == torch.float16:
            self._vae.to(dtype=torch.float32)
            pixel_values = pixel_values.float()
            did_fp32_encode = True
        elif force_upcast:
            pixel_values = pixel_values.float()

        with torch.no_grad():
            latents = self._vae.encode(pixel_values).latent_dist.sample()

        if did_fp32_encode and orig_vae_dtype is not None:
            self._vae.to(dtype=orig_vae_dtype)

        scaling = getattr(self._vae.config, "scaling_factor", 0.13025)
        shift = getattr(self._vae.config, "shift_factor", None)
        if shift is not None:
            latents = (latents - shift) * scaling
        else:
            latents = latents * scaling

        if orig_vae_dtype is not None:
            latents = latents.to(dtype=orig_vae_dtype)

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
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        self._vae = resolve_if_lazy(self._vae)

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

        p = next(self._vae.parameters(), None)
        device = p.device if p is not None else latents.device
        vae_dtype = getattr(self._vae, "dtype", None)
        if p is not None and vae_dtype is None:
            vae_dtype = p.dtype
        orig_vae_dtype = vae_dtype
        needs_fp32_decode = (
            p is not None
            and vae_dtype == torch.float16
            and bool(getattr(self._vae.config, "force_upcast", False))
        )
        if needs_fp32_decode:
            self._vae.to(dtype=torch.float32)
            try:
                pq = next(iter(self._vae.post_quant_conv.parameters()))
                decode_dtype = pq.dtype
            except (StopIteration, AttributeError):
                decode_dtype = torch.float32
            latents = latents.to(device=device, dtype=decode_dtype)
        elif p is not None:
            latents = latents.to(device=device, dtype=p.dtype)
        else:
            latents = latents.to(device=device)

        try:
            with torch.no_grad():
                image = self._vae.decode(latents, return_dict=False)[0]
        finally:
            if needs_fp32_decode and orig_vae_dtype is not None:
                self._vae.to(dtype=orig_vae_dtype)

        image = (image / 2 + 0.5).clamp(0, 1)
        return {C.PORT_DECODED_IMAGE: postprocess_image(image, output_type=output_type)}

    def to(self, device: Any) -> "SDXLVAEDecodeNode":
        if self._vae is not None:
            self._vae.to(device)
        return self
