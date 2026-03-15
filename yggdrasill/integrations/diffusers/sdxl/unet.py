"""SDXL UNet denoiser node with added_cond_kwargs support."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortAggregation, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractBackbone


class SDXLUNetNode(AbstractBackbone):
    """Wraps UNet2DConditionModel for SDXL with full conditioning.

    Handles CFG, added_cond_kwargs (text_embeds + time_ids),
    and optional ControlNet/IP-Adapter residuals.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        unet: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._unet = unet

    @property
    def block_type(self) -> str:
        return "sdxl/unet"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_ADD_TEXT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_ADD_TIME_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_ADD_TIME_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_IMAGE_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True, aggregation=PortAggregation.CONCAT),
            Port(C.PORT_DOWN_BLOCK_RESIDUALS, PortDirection.IN, PortType.ANY, optional=True, aggregation=PortAggregation.CONCAT),
            Port(C.PORT_MID_BLOCK_RESIDUAL, PortDirection.IN, PortType.ANY, optional=True, aggregation=PortAggregation.CONCAT),
            Port(C.PORT_NOISE_PRED, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from yggdrasill.integrations.diffusers.common.guidance import apply_cfg

        latents = inputs[C.PORT_LATENTS]
        timestep = inputs[C.PORT_TIMESTEP]
        prompt_embeds = inputs[C.PORT_PROMPT_EMBEDS]
        neg_embeds = inputs.get(C.PORT_NEGATIVE_PROMPT_EMBEDS)
        add_text_embeds = inputs[C.PORT_ADD_TEXT_EMBEDS]
        add_time_ids = inputs[C.PORT_ADD_TIME_IDS]
        neg_time_ids = inputs.get(C.PORT_NEGATIVE_ADD_TIME_IDS, add_time_ids)

        guidance_scale = self._config.get("guidance_scale", 7.5)
        guidance_rescale = self._config.get("guidance_rescale", 0.0)
        do_cfg = guidance_scale > 1.0 and neg_embeds is not None

        if do_cfg:
            latent_input = torch.cat([latents] * 2)
            encoder_states = torch.cat([neg_embeds, prompt_embeds])
            neg_add_text_embeds = inputs.get("negative_add_text_embeds", torch.zeros_like(add_text_embeds))
            text_embeds_cat = torch.cat([neg_add_text_embeds, add_text_embeds])
            time_ids_cat = torch.cat([neg_time_ids, add_time_ids])
        else:
            latent_input = latents
            encoder_states = prompt_embeds
            text_embeds_cat = add_text_embeds
            time_ids_cat = add_time_ids

        added_cond_kwargs: Dict[str, Any] = {
            "text_embeds": text_embeds_cat,
            "time_ids": time_ids_cat,
        }

        image_embeds = inputs.get(C.PORT_IMAGE_EMBEDS)
        if image_embeds is not None:
            if isinstance(image_embeds, list):
                import torch
                image_embeds = torch.cat(image_embeds, dim=0)
            added_cond_kwargs["image_embeds"] = image_embeds

        unet_kwargs: Dict[str, Any] = {
            "encoder_hidden_states": encoder_states,
            "added_cond_kwargs": added_cond_kwargs,
        }

        from yggdrasill.integrations.diffusers.sd15.unet import _merge_residuals

        down_residuals = inputs.get(C.PORT_DOWN_BLOCK_RESIDUALS)
        mid_residual = inputs.get(C.PORT_MID_BLOCK_RESIDUAL)
        if down_residuals is not None:
            unet_kwargs["down_block_additional_residuals"] = _merge_residuals(down_residuals)
        if mid_residual is not None:
            unet_kwargs["mid_block_additional_residual"] = _merge_residuals(mid_residual)

        timestep_cond = None
        if hasattr(self._unet, "config") and getattr(self._unet.config, "time_cond_proj_dim", None):
            timestep_cond = self._get_guidance_scale_embedding(guidance_scale)
            unet_kwargs["timestep_cond"] = timestep_cond

        noise_pred = self._unet(latent_input, timestep, **unet_kwargs).sample

        if do_cfg:
            pred_uncond, pred_cond = noise_pred.chunk(2)
            noise_pred = apply_cfg(pred_uncond, pred_cond, guidance_scale, guidance_rescale)

        return {C.PORT_NOISE_PRED: noise_pred}

    def _get_guidance_scale_embedding(self, guidance_scale: float) -> Any:
        import torch

        w = torch.tensor([guidance_scale * 1000.0])
        dim = getattr(self._unet.config, "time_cond_proj_dim", 256)
        half = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half) * -emb)
        emb = w[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb.to(device=self._unet.device, dtype=self._unet.dtype)

    def to(self, device: Any) -> "SDXLUNetNode":
        if self._unet is not None:
            self._unet.to(device)
        return self
