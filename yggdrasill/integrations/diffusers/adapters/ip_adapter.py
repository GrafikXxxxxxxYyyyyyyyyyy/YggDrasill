"""IP-Adapter node for image-conditioned generation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConjector


def align_vision_hidden_states_for_ip_adapter_plus(
    enc: Any,
    h: Any,
    *,
    token_embed_dim: Optional[int],
) -> Any:
    """Reshape ViT penultimate features to the per-token width expected by loaded Plus weights.

    *token_embed_dim* should match ``encoder_hid_proj.image_projection_layers[0].proj_in.in_features``
    after ``_load_ip_adapter_weights_into_graph`` (config key
    :data:`~yggdrasill.integrations.diffusers.contracts.CFG_IP_ADAPTER_PLUS_TOKEN_EMBED_DIM`).

    When *token_embed_dim* is ``None`` (tests, graphs built without loading Plus weights into the
    UNet), *h* is returned unchanged — set the config key manually if you use Plus with a wide ViT
    (e.g. h94 SDXL ``hidden_size=1664``) and see a ``proj_in`` shape mismatch.
    """
    if token_embed_dim is None or not hasattr(h, "shape") or h.ndim < 2:
        return h
    target = int(token_embed_dim)
    d = int(h.shape[-1])
    if d == target:
        return h
    cfg = getattr(enc, "config", None)
    hs = int(getattr(cfg, "hidden_size", 0) or 0)
    pdim = int(getattr(cfg, "projection_dim", 0) or 0)
    proj = getattr(enc, "visual_projection", None)
    if (
        proj is not None
        and d == hs
        and int(getattr(proj, "in_features", 0) or 0) == hs
        and int(getattr(proj, "out_features", 0) or 0) == pdim
        and target == pdim
    ):
        return proj(h)
    return h


def _unwrap_ip_adapter_loaded_image(loaded: Any) -> Any:
    """Normalize ``[[url]]`` / ``[[PIL]]`` (extra nesting) to ``[url]`` / ``[PIL]``.

    One reference is often passed like Diffusers ``ip_adapter_image=[[img]]``; :func:`load_image`
    then yields ``[[PIL]]``, and the feature extractor gets a batch dimension wrong → garbage
    latents. ``[img_a, img_b]`` is left unchanged.
    """
    cur: Any = loaded
    while isinstance(cur, list) and len(cur) == 1:
        inner = cur[0]
        if isinstance(inner, (list, tuple)):
            cur = list(inner)
            continue
        break
    return cur


class IPAdapterNode(AbstractConjector):
    """Reference-image conditioning encoder (Conjector): CLIP vision → UNet ``image_embeds``.

    Same role as text-side conjectors: supplies conditioning **without** changing backbone
    weights or internal layout (contrast :class:`~yggdrasill.task_nodes.abstract.AbstractInjector`
    for LoRA-style adaptation).

    When ``ip_adapter_image_embeds`` is wired (or passed via ``run(..., ip_adapter_image_embeds=...)``),
    the node forwards those tensors and **does not** run the image encoder — use for cached /
    pipeline-``prepare_ip_adapter_image_embeds`` workflows. **Pooled** tensors should be
    **conditional-only**; **Plus / Plus-Face** tensors must stay **CFG-packed**
    ``[2, num_images, seq, dim]`` (see :func:`~yggdrasill.integrations.diffusers.common.ip_adapter_embeds.prepare_ip_adapter_image_embeds`).
    For pooled Diffusers outputs that are ``cat([neg, pos], dim=0)``, use
    :func:`~yggdrasill.integrations.diffusers.common.ip_adapter_embeds.pipeline_ip_adapter_embeds_cond_only`
    only on pooled tensors — never strip the Plus uncond row.
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

    def _plus_token_embed_dim(self) -> Optional[int]:
        v = self._config.get(C.CFG_IP_ADAPTER_PLUS_TOKEN_EMBED_DIM)
        if v is None:
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    def encode_ip_adapter_image(
        self,
        ip_adapter_image: Any,
        *,
        device: Optional[Any] = None,
    ) -> Any:
        """Run CLIP preprocessor + ``image_encoder`` (same as the image branch of :meth:`forward`).

        Returns image embeddings for the UNet. **Pooled** IP-Adapter returns conditional tensors
        only (CFG doubling is applied in :func:`~yggdrasill.integrations.diffusers.common.ip_adapter_embeds.format_ip_adapter_image_embeds`).
        **Plus / Plus-Face** (``ip_adapter_use_hidden_states``) returns ``[2, N, seq, dim]`` with
        dim 0 ordered ``[encoder(zeros_like(pixel_values)), encoder(image)]``, matching Diffusers
        ``encode_image`` — :func:`~yggdrasill.integrations.diffusers.common.ip_adapter_embeds.prepare_ip_adapter_image_embeds`
        preserves this packing so the UNet does not replace the uncond row with ``zeros_like`` in
        embedding space.

        Args:
            ip_adapter_image: URL, path, PIL image, or tensor (tensor path only when encoder is absent).
            device: Optional device for the image encoder and output tensors (mutates encoder placement).

        Node config:
            ``ip_adapter_vision_float32`` (bool): If True, run the vision encoder in float32 for this
            encode (then restore weight dtype), and cast embeddings back to the encoder weight dtype
            for ``encoder_hid_proj``. Off by default; enable only if you see clear benefit, as the
            fp16 round-trip can sometimes hurt more than it helps.
        """
        import torch

        from yggdrasill.integrations.diffusers.common.image_utils import load_image as _load_image
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        self._image_encoder = resolve_if_lazy(self._image_encoder)
        fe_raw = self._feature_extractor
        fe = resolve_if_lazy(fe_raw) if fe_raw is not None else None

        ip_image = _unwrap_ip_adapter_loaded_image(_load_image(ip_adapter_image))

        if fe is not None:
            enc = self._image_encoder
            if enc is None:
                raise ValueError("IP-Adapter node has feature_extractor but no image_encoder")
            if device is not None and hasattr(enc, "to"):
                self._image_encoder = enc.to(device)
                enc = self._image_encoder
            # Match inference pipelines: train-mode dropout (if any) would randomize embeddings
            # every denoising step and destroy IP-Adapter conditioning.
            _eval = getattr(enc, "eval", None)
            if callable(_eval):
                _eval()
            pixel_values = fe(
                images=ip_image if isinstance(ip_image, list) else [ip_image],
                return_tensors="pt",
            ).pixel_values
            pixel_values = pixel_values.to(device=enc.device)
            vision_fp32 = bool(self._config.get("ip_adapter_vision_float32", False))
            orig_enc_dtype = None
            if vision_fp32:
                try:
                    orig_enc_dtype = next(enc.parameters()).dtype
                except (StopIteration, TypeError):
                    orig_enc_dtype = None
                if orig_enc_dtype is not None and orig_enc_dtype != torch.float32:
                    enc = enc.to(torch.float32)
                    self._image_encoder = enc
                pixel_values = pixel_values.to(dtype=torch.float32)
            else:
                pixel_values = pixel_values.to(dtype=enc.dtype)
            use_hidden = bool(self._config.get("ip_adapter_use_hidden_states", False))
            try:
                with torch.inference_mode():
                    if use_hidden:
                        out = enc(pixel_values, output_hidden_states=True)
                        h = align_vision_hidden_states_for_ip_adapter_plus(
                            enc,
                            out.hidden_states[-2],
                            token_embed_dim=self._plus_token_embed_dim(),
                        )
                        # Match Diffusers ``encode_image(..., output_hidden_states=True)``: the CFG
                        # "unconditional" branch uses the vision encoder on black pixels, not literal
                        # zero tensors (see SDXL ``prepare_ip_adapter_image_embeds``). Zero embeddings
                        # break Plus / Plus-Face spatial alignment and yield mangled faces.
                        try:
                            zeros_pv = torch.zeros_like(pixel_values)
                        except TypeError:
                            sh = tuple(getattr(pixel_values, "shape", ()))
                            dev = getattr(pixel_values, "device", "cpu")
                            dt = getattr(pixel_values, "dtype", torch.float32)
                            if not isinstance(dt, torch.dtype):
                                dt = torch.float32
                            zeros_pv = torch.zeros(sh, dtype=dt, device=dev)
                        out_u = enc(zeros_pv, output_hidden_states=True)
                        h_u = align_vision_hidden_states_for_ip_adapter_plus(
                            enc,
                            out_u.hidden_states[-2],
                            token_embed_dim=self._plus_token_embed_dim(),
                        )

                        def _t(x: Any) -> "torch.Tensor":
                            if isinstance(x, torch.Tensor):
                                return x
                            return torch.zeros(tuple(getattr(x, "shape", ())), dtype=torch.float32)

                        e_c = _t(h).unsqueeze(0)
                        e_u = _t(h_u).unsqueeze(0)
                        encoded = torch.cat([e_u, e_c], dim=0)
                    else:
                        encoded = enc(pixel_values).image_embeds
            finally:
                if vision_fp32 and orig_enc_dtype is not None and orig_enc_dtype != torch.float32:
                    self._image_encoder = self._image_encoder.to(orig_enc_dtype)
            if (
                vision_fp32
                and orig_enc_dtype is not None
                and torch.is_tensor(encoded)
                and encoded.dtype != orig_enc_dtype
            ):
                # Vision ran in fp32; UNet ``encoder_hid_proj`` / ``proj_in`` stay fp16 → match dtypes.
                encoded = encoded.to(dtype=orig_enc_dtype)
            return encoded

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
                use_hidden = bool(self._config.get("ip_adapter_use_hidden_states", False))
                _eval = getattr(self._image_encoder, "eval", None)
                if callable(_eval):
                    _eval()
                with torch.inference_mode():
                    if use_hidden:
                        enc = self._image_encoder
                        out = enc(pixel_values, output_hidden_states=True)
                        h = align_vision_hidden_states_for_ip_adapter_plus(
                            enc,
                            out.hidden_states[-2],
                            token_embed_dim=self._plus_token_embed_dim(),
                        )
                        ref = h.unsqueeze(0)
                    else:
                        ref = self._image_encoder(pixel_values).image_embeds
                self._cached_zero_image_embeds = torch.zeros_like(ref)
            return self._cached_zero_image_embeds
        dim = int(self._config.get("ip_adapter_embed_dim", 1024))
        return torch.zeros(1, dim)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
