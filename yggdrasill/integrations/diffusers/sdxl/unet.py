"""SDXL UNet denoiser node with added_cond_kwargs support."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortAggregation, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractBackbone


class SDXLUNetNode(AbstractBackbone):
    """Wraps UNet2DConditionModel for SDXL with full conditioning.

    Handles CFG, added_cond_kwargs (text_embeds + time_ids),
    optional ControlNet/IP-Adapter residuals, and 9-channel inpaint concat.
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
        return "sdxl/unet"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(
                C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS,
                PortDirection.IN,
                PortType.TENSOR,
                optional=True,
            ),
            Port(C.PORT_ADD_TEXT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_ADD_TIME_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_ADD_TIME_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_MASK_LATENTS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(
                C.PORT_MASKED_IMAGE_LATENTS,
                PortDirection.IN,
                PortType.TENSOR,
                optional=True,
            ),
            Port(C.PORT_IMAGE_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True, aggregation=PortAggregation.CONCAT),
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
        from yggdrasill.integrations.diffusers.common.scheduler_step import (
            scheduler_uses_float_timestep_in_step,
        )

        latents = inputs[C.PORT_LATENTS]
        timestep_in = inputs[C.PORT_TIMESTEP]
        prompt_embeds = inputs[C.PORT_PROMPT_EMBEDS]
        neg_embeds = inputs.get(C.PORT_NEGATIVE_PROMPT_EMBEDS)
        add_text_embeds = inputs[C.PORT_ADD_TEXT_EMBEDS]
        add_time_ids = inputs[C.PORT_ADD_TIME_IDS]
        neg_time_ids = inputs.get(C.PORT_NEGATIVE_ADD_TIME_IDS, add_time_ids)

        dtype = next(self._unet.parameters()).dtype
        device = next(self._unet.parameters()).device
        latents = latents.to(device=device, dtype=dtype)

        sched_state = inputs.get(C.PORT_SCHEDULER_STATE)
        sched = sched_state.get("scheduler") if isinstance(sched_state, dict) else None
        use_float_t = scheduler_uses_float_timestep_in_step(sched)

        if isinstance(timestep_in, torch.Tensor):
            timestep = timestep_in.to(device=device)
            if timestep.ndim > 0:
                timestep = timestep.reshape(-1)[0]
            if not use_float_t and timestep.dtype in (
                torch.float16, torch.bfloat16, torch.float32, torch.float64,
            ):
                timestep = timestep.long()
            elif use_float_t:
                timestep = timestep.to(dtype=dtype)
        else:
            if hasattr(timestep_in, "item") and callable(getattr(timestep_in, "item")):
                try:
                    t_val = timestep_in.item()
                except (TypeError, ValueError):
                    t_val = timestep_in
            else:
                t_val = timestep_in
            if use_float_t:
                timestep = torch.tensor(float(t_val), device=device, dtype=dtype)
            else:
                timestep = torch.tensor(int(t_val), device=device, dtype=torch.long)

        guidance_scale = self._config.get("guidance_scale", 7.5)
        guidance_rescale = self._config.get("guidance_rescale", 0.0)
        do_cfg = guidance_scale > 1.0 and neg_embeds is not None

        if do_cfg:
            latent_input = torch.cat([latents] * 2)
            encoder_states = torch.cat([
                neg_embeds.to(device=device, dtype=dtype),
                prompt_embeds.to(device=device, dtype=dtype),
            ])
            neg_add_text_embeds = inputs.get(C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS)
            if neg_add_text_embeds is None:
                neg_add_text_embeds = torch.zeros_like(add_text_embeds)
            text_embeds_cat = torch.cat([
                neg_add_text_embeds.to(device=device, dtype=dtype),
                add_text_embeds.to(device=device, dtype=dtype),
            ])
            time_ids_cat = torch.cat([
                neg_time_ids.to(device=device, dtype=dtype),
                add_time_ids.to(device=device, dtype=dtype),
            ])
        else:
            latent_input = latents
            encoder_states = prompt_embeds.to(device=device, dtype=dtype)
            text_embeds_cat = add_text_embeds.to(device=device, dtype=dtype)
            time_ids_cat = add_time_ids.to(device=device, dtype=dtype)

        added_cond_kwargs: Dict[str, Any] = {
            "text_embeds": text_embeds_cat,
            "time_ids": time_ids_cat,
        }

        if sched is not None and hasattr(sched, "scale_model_input"):
            latent_input = sched.scale_model_input(latent_input, timestep)

        mask_latents = inputs.get(C.PORT_MASK_LATENTS)
        masked_latents = inputs.get(C.PORT_MASKED_IMAGE_LATENTS)
        in_ch = getattr(getattr(self._unet, "config", None), "in_channels", 4)
        if (
            mask_latents is not None
            and masked_latents is not None
            and in_ch == 9
        ):
            mask_latents = mask_latents.to(device=device, dtype=dtype)
            masked_latents = masked_latents.to(device=device, dtype=dtype)
            if do_cfg:
                mask_latents = torch.cat([mask_latents, mask_latents], dim=0)
                masked_latents = torch.cat([masked_latents, masked_latents], dim=0)
            latent_input = torch.cat(
                [latent_input, mask_latents, masked_latents],
                dim=1,
            )

        b_cond = latent_input.shape[0] // 2 if do_cfg else latent_input.shape[0]
        image_embeds = inputs.get(C.PORT_IMAGE_EMBEDS)
        from yggdrasill.integrations.diffusers.common.ip_adapter_embeds import (
            format_ip_adapter_image_embeds,
            raw_zero_ip_adapter_image_embeds_for_unet,
            unet_requires_image_embeds_in_added_cond,
        )

        if image_embeds is not None:
            added_cond_kwargs["image_embeds"] = format_ip_adapter_image_embeds(
                image_embeds,
                device=device,
                dtype=None,
                do_classifier_free_guidance=do_cfg,
            )
        elif unet_requires_image_embeds_in_added_cond(self._unet):
            added_cond_kwargs["image_embeds"] = format_ip_adapter_image_embeds(
                raw_zero_ip_adapter_image_embeds_for_unet(
                    self._unet, b_cond, device=device,
                ),
                device=device,
                dtype=None,
                do_classifier_free_guidance=do_cfg,
            )

        unet_kwargs: Dict[str, Any] = {
            "encoder_hidden_states": encoder_states,
            "added_cond_kwargs": added_cond_kwargs,
        }

        from yggdrasill.integrations.diffusers.common.merge import merge_residuals

        down_residuals = inputs.get(C.PORT_DOWN_BLOCK_RESIDUALS)
        mid_residual = inputs.get(C.PORT_MID_BLOCK_RESIDUAL)
        if down_residuals is not None:
            unet_kwargs["down_block_additional_residuals"] = merge_residuals(down_residuals)
        if mid_residual is not None:
            unet_kwargs["mid_block_additional_residual"] = merge_residuals(mid_residual)

        if hasattr(self._unet, "config") and getattr(self._unet.config, "time_cond_proj_dim", None):
            unet_kwargs["timestep_cond"] = self._get_guidance_scale_embedding(guidance_scale)

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
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        self._unet = resolve_if_lazy(self._unet)
        if self._unet is not None and hasattr(self._unet, "to"):
            self._unet.to(device)
        return self
