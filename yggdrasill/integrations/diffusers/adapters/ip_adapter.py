"""IP-Adapter node for image-conditioned generation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInjector


class IPAdapterNode(AbstractInjector):
    """Processes reference images through an IP-Adapter image encoder.

    Produces image embeddings that are injected into the UNet via
    ``added_cond_kwargs["image_embeds"]``.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        image_encoder: Any = None,
        feature_extractor: Any = None,
    ) -> None:
        cfg = dict(config or {})
        image_encoder = image_encoder or cfg.pop("image_encoder", None)
        feature_extractor = feature_extractor or cfg.pop("feature_extractor", None)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)
        self._image_encoder = image_encoder
        self._feature_extractor = feature_extractor
        self._cached_zero_image_embeds: Any = None

    @property
    def block_type(self) -> str:
        return "adapter/ip_adapter"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_IP_ADAPTER_IMAGE, PortDirection.IN, PortType.IMAGE, optional=True),
            Port(C.PORT_IMAGE_EMBEDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def _inactive_image_embeds(self) -> Any:
        """Zeros with the same shape as a real encoding so multi-IP-Adapter UNets stay aligned."""
        import torch

        if self._feature_extractor is not None and self._image_encoder is not None:
            if self._cached_zero_image_embeds is None:
                from PIL import Image

                img = Image.new("RGB", (64, 64), (0, 0, 0))
                pixel_values = self._feature_extractor(
                    images=[img], return_tensors="pt"
                ).pixel_values
                dev = next(self._image_encoder.parameters()).device
                dt = next(self._image_encoder.parameters()).dtype
                pixel_values = pixel_values.to(device=dev, dtype=dt)
                with torch.inference_mode():
                    ref = self._image_encoder(pixel_values).image_embeds
                self._cached_zero_image_embeds = torch.zeros_like(ref)
            return self._cached_zero_image_embeds
        dim = int(self._config.get("ip_adapter_embed_dim", 1024))
        return torch.zeros(1, dim)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        ip_image = inputs.get(C.PORT_IP_ADAPTER_IMAGE)
        if ip_image is None:
            return {C.PORT_IMAGE_EMBEDS: self._inactive_image_embeds()}

        from yggdrasill.integrations.diffusers.common.image_utils import load_image as _load_image
        ip_image = _load_image(ip_image)

        if self._feature_extractor is not None:
            pixel_values = self._feature_extractor(
                images=ip_image if isinstance(ip_image, list) else [ip_image],
                return_tensors="pt",
            ).pixel_values

            if self._image_encoder is not None:
                pixel_values = pixel_values.to(
                    device=self._image_encoder.device,
                    dtype=self._image_encoder.dtype,
                )
                with torch.no_grad():
                    image_embeds = self._image_encoder(pixel_values).image_embeds
            else:
                image_embeds = pixel_values
        elif isinstance(ip_image, torch.Tensor):
            image_embeds = ip_image
        else:
            raise ValueError(
                "IP-Adapter requires either a feature_extractor+image_encoder "
                "or pre-computed tensor embeddings."
            )

        return {C.PORT_IMAGE_EMBEDS: image_embeds}

    def to(self, device: Any) -> "IPAdapterNode":
        self._cached_zero_image_embeds = None
        if self._image_encoder is not None:
            self._image_encoder.to(device)
        return self
