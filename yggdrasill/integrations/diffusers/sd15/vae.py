"""SD1.5 VAE encode/decode nodes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConverter


class SD15VAEEncodeNode(AbstractConverter):
    """Encodes images to latent space using AutoencoderKL."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        vae: Any = None,
    ) -> None:
        cfg = dict(config or {})
        self._vae = vae or cfg.pop("vae", None)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)

    @property
    def block_type(self) -> str:
        return "sd15/vae_encode"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_INIT_IMAGE, PortDirection.IN, PortType.IMAGE, optional=True),
            Port(C.PORT_LATENTS, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._vae = resolve_if_lazy(self._vae)

        import torch
        from yggdrasill.integrations.diffusers.common.image_utils import preprocess_image

        image = inputs.get(C.PORT_INIT_IMAGE)
        if image is None:
            raise RuntimeError(
                "sd15/vae_encode ran without init_image; executor should skip this node "
                "for text2img (universal SD1.5 graph)."
            )
        device = self._config.get("device", "cpu")
        dtype = getattr(self._vae, "dtype", None)

        pixel_values = preprocess_image(
            image,
            height=self._config.get("height", 512),
            width=self._config.get("width", 512),
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            latent_dist = self._vae.encode(pixel_values).latent_dist
            latents = latent_dist.sample()

        scaling_factor = getattr(self._vae.config, "scaling_factor", 0.18215)
        latents = latents * scaling_factor

        return {C.PORT_LATENTS: latents}

    def to(self, device: Any) -> "SD15VAEEncodeNode":
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._vae = resolve_if_lazy(self._vae)
        if self._vae is not None:
            self._vae.to(device)
        return self


class SD15VAEDecodeNode(AbstractConverter):
    """Decodes latents back to pixel space and postprocesses."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        vae: Any = None,
    ) -> None:
        cfg = dict(config or {})
        self._vae = vae or cfg.pop("vae", None)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)

    @property
    def block_type(self) -> str:
        return "sd15/vae_decode"

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

        scaling = getattr(self._vae.config, "scaling_factor", 0.18215)
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
        result = postprocess_image(image, output_type=output_type)

        return {C.PORT_DECODED_IMAGE: result}

    def to(self, device: Any) -> "SD15VAEDecodeNode":
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._vae = resolve_if_lazy(self._vae)
        if self._vae is not None:
            self._vae.to(device)
        return self
