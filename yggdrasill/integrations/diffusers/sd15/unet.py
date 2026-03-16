"""SD1.5 UNet denoiser node."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortAggregation, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractBackbone


def _merge_residuals(value: Any) -> Any:
    """Merge residuals from multiple adapters.

    When ``CONCAT`` aggregation is used, the executor delivers a list of
    residual tuples (one per adapter).  This helper element-wise sums
    them into a single residual tuple suitable for the UNet.  If the
    value is not a list (single adapter), it is returned as-is.
    """
    if not isinstance(value, list):
        return value
    if len(value) == 1:
        return value[0]
    import torch
    merged = value[0]
    if isinstance(merged, (list, tuple)):
        merged = list(merged)
        for extra in value[1:]:
            for i, t in enumerate(extra):
                merged[i] = merged[i] + t
        return tuple(merged)
    for extra in value[1:]:
        merged = merged + extra
    return merged


class SD15UNetNode(AbstractBackbone):
    """Wraps UNet2DConditionModel for a single denoising forward pass.

    Handles CFG expansion internally: if both positive and negative embeds
    are provided, concatenates input latents, runs UNet once on the doubled
    batch, splits, and applies guidance.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        unet: Any = None,
    ) -> None:
        cfg = dict(config or {})
        self._unet = unet or cfg.pop("unet", None)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)

    @property
    def block_type(self) -> str:
        return "sd15/unet"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_SCHEDULER_STATE, PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_DOWN_BLOCK_RESIDUALS, PortDirection.IN, PortType.ANY, optional=True, aggregation=PortAggregation.CONCAT),
            Port(C.PORT_MID_BLOCK_RESIDUAL, PortDirection.IN, PortType.ANY, optional=True, aggregation=PortAggregation.CONCAT),
            Port(C.PORT_NOISE_PRED, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._unet = resolve_if_lazy(self._unet)

        import torch
        from yggdrasill.integrations.diffusers.common.guidance import apply_cfg

        latents = inputs[C.PORT_LATENTS]
        timestep = inputs[C.PORT_TIMESTEP]
        prompt_embeds = inputs[C.PORT_PROMPT_EMBEDS]
        neg_embeds = inputs.get(C.PORT_NEGATIVE_PROMPT_EMBEDS)
        sched_state = inputs.get(C.PORT_SCHEDULER_STATE)

        guidance_scale = self._config.get("guidance_scale", 7.5)
        do_cfg = guidance_scale > 1.0 and neg_embeds is not None

        if do_cfg:
            latent_input = torch.cat([latents] * 2)
            encoder_states = torch.cat([neg_embeds, prompt_embeds])
        else:
            latent_input = latents
            encoder_states = prompt_embeds

        if sched_state and isinstance(sched_state, dict):
            sched = sched_state.get("scheduler")
            if sched is not None and hasattr(sched, "scale_model_input"):
                latent_input = sched.scale_model_input(latent_input, timestep)

        kwargs: Dict[str, Any] = {
            "encoder_hidden_states": encoder_states,
        }
        down_residuals = inputs.get(C.PORT_DOWN_BLOCK_RESIDUALS)
        mid_residual = inputs.get(C.PORT_MID_BLOCK_RESIDUAL)
        if down_residuals is not None:
            down_residuals = _merge_residuals(down_residuals)
            kwargs["down_block_additional_residuals"] = down_residuals
        if mid_residual is not None:
            mid_residual = _merge_residuals(mid_residual)
            kwargs["mid_block_additional_residual"] = mid_residual

        noise_pred = self._unet(
            latent_input,
            timestep,
            **kwargs,
        ).sample

        if do_cfg:
            pred_uncond, pred_cond = noise_pred.chunk(2)
            noise_pred = apply_cfg(
                pred_uncond, pred_cond,
                guidance_scale,
                guidance_rescale=self._config.get("guidance_rescale", 0.0),
            )

        return {C.PORT_NOISE_PRED: noise_pred}

    def to(self, device: Any) -> "SD15UNetNode":
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._unet = resolve_if_lazy(self._unet)
        if self._unet is not None and hasattr(self._unet, "to"):
            self._unet.to(device)
        return self
