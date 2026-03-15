"""FLUX transformer (MMDiT) backbone node."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortAggregation, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractBackbone


class FluxTransformerNode(AbstractBackbone):
    """Wraps FluxTransformer2DModel for FLUX denoising.

    Unlike SD/SDXL UNet, FLUX uses a DiT (Diffusion Transformer) backbone
    with joint attention blocks (MMDiT). Key differences:

    - No classifier-free guidance (no double forward pass)
    - guidance_scale is embedded via CombinedTimestepGuidanceTextProjEmbeddings
    - Latents are 2x2-packed: [B, num_patches, C*4]
    - Positional encoding via RoPE with img_ids and txt_ids
    - Timestep is normalized to [0, 1] before passing to transformer
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        transformer: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._transformer = transformer

    @property
    def block_type(self) -> str:
        return "flux/transformer"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PACKED_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_POOLED_PROJECTIONS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_IMG_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TXT_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_GUIDANCE, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_CONTROLNET_BLOCK_SAMPLES, PortDirection.IN, PortType.ANY, optional=True, aggregation=PortAggregation.CONCAT),
            Port(C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES, PortDirection.IN, PortType.ANY, optional=True, aggregation=PortAggregation.CONCAT),
            Port(C.PORT_NOISE_PRED, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        latents = inputs[C.PORT_PACKED_LATENTS]
        timestep = inputs[C.PORT_TIMESTEP]
        prompt_embeds = inputs[C.PORT_PROMPT_EMBEDS]
        pooled_projections = inputs[C.PORT_POOLED_PROJECTIONS]
        img_ids = inputs[C.PORT_IMG_IDS]
        txt_ids = inputs[C.PORT_TXT_IDS]

        timestep_normalized = timestep / 1000

        kwargs: Dict[str, Any] = {
            "hidden_states": latents,
            "timestep": timestep_normalized,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_projections,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "return_dict": False,
        }

        has_guidance_embeds = (
            hasattr(self._transformer, "config")
            and getattr(self._transformer.config, "guidance_embeds", False)
        )
        guidance = inputs.get(C.PORT_GUIDANCE)
        if has_guidance_embeds and guidance is not None:
            kwargs["guidance"] = guidance

        joint_attention_kwargs = self._config.get("joint_attention_kwargs")
        if joint_attention_kwargs is not None:
            kwargs["joint_attention_kwargs"] = joint_attention_kwargs

        from yggdrasill.integrations.diffusers.sd15.unet import _merge_residuals

        cn_block = inputs.get(C.PORT_CONTROLNET_BLOCK_SAMPLES)
        cn_single = inputs.get(C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES)
        if cn_block is not None:
            kwargs["controlnet_block_samples"] = _merge_residuals(cn_block)
        if cn_single is not None:
            kwargs["controlnet_single_block_samples"] = _merge_residuals(cn_single)

        output = self._transformer(**kwargs)
        noise_pred = output[0] if isinstance(output, (tuple, list)) else output

        return {C.PORT_NOISE_PRED: noise_pred}

    def to(self, device: Any) -> "FluxTransformerNode":
        if self._transformer is not None:
            self._transformer.to(device)
        return self
