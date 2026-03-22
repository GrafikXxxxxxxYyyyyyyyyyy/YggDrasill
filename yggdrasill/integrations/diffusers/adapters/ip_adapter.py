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

    When ``ip_adapter_image_embeds`` is wired (or passed via ``run(..., ip_adapter_image_embeds=...)``),
    the node forwards those tensors and **does not** run the image encoder — use for cached /
    pipeline-``prepare_ip_adapter_image_embeds`` workflows. Tensors should be **conditional-only**
    (see :func:`~yggdrasill.integrations.diffusers.common.ip_adapter_embeds.pipeline_ip_adapter_embeds_cond_only`
    if you saved Diffusers CFG-packed tensors).
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
            Port(C.PORT_IP_ADAPTER_IMAGE_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_IMAGE_EMBEDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def encode_ip_adapter_image(
        self,
        ip_adapter_image: Any,
        *,
        device: Optional[Any] = None,
    ) -> Any:
        """Run CLIP preprocessor + ``image_encoder`` (same as the image branch of :meth:`forward`).

        Returns **conditional** image embeddings (no classifier-free doubling). Suitable for caching
        and for :func:`~yggdrasill.integrations.diffusers.common.ip_adapter_embeds.prepare_ip_adapter_image_embeds`.

        Args:
            ip_adapter_image: URL, path, PIL image, or tensor (tensor path only when encoder is absent).
            device: Optional device for the image encoder and output tensors (mutates encoder placement).
        """
        import torch

        from yggdrasill.integrations.diffusers.common.image_utils import load_image as _load_image
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        self._image_encoder = resolve_if_lazy(self._image_encoder)
        fe_raw = self._feature_extractor
        fe = resolve_if_lazy(fe_raw) if fe_raw is not None else None

        ip_image = _load_image(ip_adapter_image)

        if fe is not None:
            enc = self._image_encoder
            if enc is None:
                raise ValueError("IP-Adapter node has feature_extractor but no image_encoder")
            if device is not None and hasattr(enc, "to"):
                self._image_encoder = enc.to(device)
                enc = self._image_encoder
            pixel_values = fe(
                images=ip_image if isinstance(ip_image, list) else [ip_image],
                return_tensors="pt",
            ).pixel_values
            pixel_values = pixel_values.to(device=enc.device, dtype=enc.dtype)
            with torch.no_grad():
                return enc(pixel_values).image_embeds

        if isinstance(ip_image, torch.Tensor):
            t = ip_image
            if device is not None:
                t = t.to(device=device)
            return t

        raise ValueError(
            "IP-Adapter encode_ip_adapter_image requires feature_extractor+image_encoder "
            "or a pre-computed torch.Tensor."
        )

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

        precomputed = inputs.get(C.PORT_IP_ADAPTER_IMAGE_EMBEDS)
        if precomputed is not None:
            if isinstance(precomputed, (list, tuple)):
                tensors = [x for x in precomputed if x is not None]
                if not tensors:
                    return {C.PORT_IMAGE_EMBEDS: self._inactive_image_embeds()}
                if len(tensors) == 1:
                    return {C.PORT_IMAGE_EMBEDS: tensors[0]}
                return {C.PORT_IMAGE_EMBEDS: list(tensors)}
            return {C.PORT_IMAGE_EMBEDS: precomputed}

        ip_image = inputs.get(C.PORT_IP_ADAPTER_IMAGE)
        if ip_image is None:
            return {C.PORT_IMAGE_EMBEDS: self._inactive_image_embeds()}

        image_embeds = self.encode_ip_adapter_image(ip_image, device=None)
        return {C.PORT_IMAGE_EMBEDS: image_embeds}

    def to(self, device: Any) -> "IPAdapterNode":
        self._cached_zero_image_embeds = None
        if self._image_encoder is not None:
            self._image_encoder.to(device)
        return self
