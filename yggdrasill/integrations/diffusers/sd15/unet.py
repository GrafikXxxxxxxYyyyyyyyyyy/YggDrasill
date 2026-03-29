"""SD1.5 UNet denoiser node."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.common.merge import merge_residuals
from yggdrasill.foundation.port import Port, PortAggregation, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractBackbone


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

    def _resolve_unet(self) -> Any:
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._unet = resolve_if_lazy(self._unet)
        return self._unet

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
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
            Port(C.PORT_DOWN_INTRABLOCK_RESIDUALS, PortDirection.IN, PortType.ANY, optional=True, aggregation=PortAggregation.CONCAT),
            # Allow multi-edge wiring: one ip_mask_prep per loaded IP-Adapter slot.
            # Forward already supports list/tuple and packages it into cross_attention_kwargs.
            Port(C.PORT_IP_ADAPTER_MASKS, PortDirection.IN, PortType.TENSOR, optional=True, aggregation=PortAggregation.CONCAT),
            Port(C.PORT_NOISE_PRED, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._resolve_unet()

        import torch
        from yggdrasill.integrations.diffusers.common.guidance import apply_cfg

        latents = inputs[C.PORT_LATENTS]
        timestep = inputs[C.PORT_TIMESTEP]
        prompt_embeds = inputs[C.PORT_PROMPT_EMBEDS]
        lat_ndim = (
            int(latents.dim())
            if hasattr(latents, "dim") and callable(getattr(latents, "dim", None))
            else int(getattr(latents, "ndim", 4))
        )
        is_video = lat_ndim == 5
        num_frames = int(latents.shape[2]) if is_video else 1

        from yggdrasill.integrations.diffusers.common.scheduler_step import (
            scheduler_reference_timestep_dtype,
            scheduler_uses_float_timestep_in_step,
        )

        # Ensure dtype match with UNet (avoids "Half and Float" in time_embedding)
        dtype = next(self._unet.parameters()).dtype
        device = next(self._unet.parameters()).device
        latents = latents.to(device=device, dtype=dtype)
        sched_state = inputs.get(C.PORT_SCHEDULER_STATE)
        sched = sched_state.get("scheduler") if isinstance(sched_state, dict) else None
        use_float_t = scheduler_uses_float_timestep_in_step(sched)

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(device=device)
            if timestep.ndim > 0:
                timestep = timestep.reshape(-1)[0]
            if not use_float_t and timestep.dtype in (
                torch.float16, torch.bfloat16, torch.float32, torch.float64,
            ):
                timestep = timestep.long()
            elif use_float_t:
                tref = scheduler_reference_timestep_dtype(sched)
                timestep = timestep.to(dtype=tref)
        else:
            if hasattr(timestep, "item") and callable(getattr(timestep, "item")):
                try:
                    t_val = timestep.item()
                except (TypeError, ValueError):
                    t_val = timestep
            else:
                t_val = timestep
            if use_float_t:
                tref = scheduler_reference_timestep_dtype(sched)
                timestep = torch.tensor(float(t_val), device=device, dtype=tref)
            else:
                timestep = torch.tensor(int(t_val), device=device, dtype=torch.long)
        neg_embeds = inputs.get(C.PORT_NEGATIVE_PROMPT_EMBEDS)

        guidance_scale = self._config.get("guidance_scale", 7.5)
        do_cfg = guidance_scale > 1.0 and neg_embeds is not None

        if do_cfg:
            latent_input = torch.cat([latents] * 2)
            encoder_states = torch.cat([
                neg_embeds.to(device=device, dtype=dtype),
                prompt_embeds.to(device=device, dtype=dtype),
            ])
        else:
            latent_input = latents
            encoder_states = prompt_embeds.to(device=device, dtype=dtype)

        # AnimateDiff / UNetMotionModel: match diffusers AnimateDiffPipeline (repeat per frame on batch dim).
        if is_video and num_frames > 1:
            encoder_states = encoder_states.repeat_interleave(num_frames, dim=0)

        if sched_state and isinstance(sched_state, dict):
            sched = sched_state.get("scheduler")
            if sched is not None and hasattr(sched, "scale_model_input"):
                latent_input = sched.scale_model_input(latent_input, timestep)

        mask_latents = inputs.get(C.PORT_MASK_LATENTS)
        masked_latents = inputs.get(C.PORT_MASKED_IMAGE_LATENTS)
        in_ch = getattr(getattr(self._unet, "config", None), "in_channels", 4)

        def _inpaint_aux_to_video(aux: Any) -> Any:
            """Broadcast 4D mask/masked latents to (B, C_aux, F, H, W) (before CFG doubling)."""
            aux = aux.to(device=device, dtype=dtype)
            if not is_video or num_frames <= 1:
                return aux
            if aux.dim() != 4:
                if aux.dim() == 5 and aux.shape[2] == num_frames:
                    return aux
                raise ValueError(
                    f"inpaint aux for video expects 4D (B,C,H,W) or 5D with F={num_frames}; got {tuple(aux.shape)}"
                )
            b_a, c_a, h_a, w_a = aux.shape
            b_lat = int(latents.shape[0])
            if b_a != b_lat:
                raise ValueError(f"inpaint aux batch {b_a} != latent batch {b_lat}")
            return aux.unsqueeze(2).expand(b_lat, c_a, num_frames, h_a, w_a).contiguous()

        if (
            mask_latents is not None
            and masked_latents is not None
            and in_ch == 9
        ):
            mask_latents = _inpaint_aux_to_video(mask_latents)
            masked_latents = _inpaint_aux_to_video(masked_latents)
            if do_cfg:
                mask_latents = torch.cat([mask_latents, mask_latents], dim=0)
                masked_latents = torch.cat([masked_latents, masked_latents], dim=0)
            latent_input = torch.cat(
                [latent_input, mask_latents, masked_latents],
                dim=1,
            )

        kwargs: Dict[str, Any] = {
            "encoder_hidden_states": encoder_states,
        }
        b_cond = latent_input.shape[0] // 2 if do_cfg else latent_input.shape[0]
        image_embeds = inputs.get(C.PORT_IMAGE_EMBEDS)
        from yggdrasill.integrations.diffusers.common.ip_adapter_embeds import (
            format_ip_adapter_image_embeds,
            raw_zero_ip_adapter_image_embeds_for_unet,
            unet_requires_image_embeds_in_added_cond,
        )

        if image_embeds is not None:
            kwargs["added_cond_kwargs"] = {
                "image_embeds": format_ip_adapter_image_embeds(
                    image_embeds,
                    device=device,
                    dtype=dtype,
                    do_classifier_free_guidance=do_cfg,
                )
            }
        elif unet_requires_image_embeds_in_added_cond(self._unet):
            # Diffusers runs ``\"image_embeds\" not in added_cond_kwargs`` without a None-guard;
            # default ``added_cond_kwargs=None`` then raises TypeError.
            kwargs["added_cond_kwargs"] = {
                "image_embeds": format_ip_adapter_image_embeds(
                    raw_zero_ip_adapter_image_embeds_for_unet(
                        self._unet, b_cond, device=device, dtype=dtype,
                    ),
                    device=device,
                    dtype=dtype,
                    do_classifier_free_guidance=do_cfg,
                )
            }
        def _residuals_to_unet_dtype(x: Any) -> Any:
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return type(x)(_residuals_to_unet_dtype(t) for t in x)
            if hasattr(x, "to"):
                return x.to(device=device, dtype=dtype)
            return x

        down_residuals = inputs.get(C.PORT_DOWN_BLOCK_RESIDUALS)
        mid_residual = inputs.get(C.PORT_MID_BLOCK_RESIDUAL)
        if down_residuals is not None:
            down_residuals = _residuals_to_unet_dtype(merge_residuals(down_residuals))
            kwargs["down_block_additional_residuals"] = down_residuals
        if mid_residual is not None:
            mid_residual = _residuals_to_unet_dtype(merge_residuals(mid_residual))
            kwargs["mid_block_additional_residual"] = mid_residual
        intrablock = inputs.get(C.PORT_DOWN_INTRABLOCK_RESIDUALS)
        if intrablock is not None:
            merged_intra = merge_residuals(intrablock)
            merged_intra = _residuals_to_unet_dtype(merged_intra)
            if isinstance(merged_intra, tuple):
                merged_intra = list(merged_intra)
            kwargs["down_intrablock_additional_residuals"] = merged_intra

        ip_masks = inputs.get(C.PORT_IP_ADAPTER_MASKS)
        if ip_masks is not None:
            cross_kw: Dict[str, Any] = {}
            if isinstance(ip_masks, (list, tuple)):
                cast_masks: List[Any] = []
                for t in ip_masks:
                    if hasattr(t, "to"):
                        # Keep masks dtype (usually float32) to match diffusers preprocessing.
                        # The attention processor will cast downsampled masks to query dtype later.
                        cast_masks.append(t.to(device=device))
                    else:
                        cast_masks.append(t)
                cross_kw["ip_adapter_masks"] = cast_masks
            else:
                m = ip_masks
                if hasattr(m, "to"):
                    m = m.to(device=device)
                cross_kw["ip_adapter_masks"] = [m]
            kwargs["cross_attention_kwargs"] = cross_kw

        # Match diffusers pipelines: no autocast here. fp16/bf16 weights already run in
        # matching dtype; autocast with ControlNet residuals can destabilize the denoiser.
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
        self._resolve_unet()
        if self._unet is not None and hasattr(self._unet, "to"):
            self._unet.to(device)
        return self

    def train(self, mode: bool = True) -> "SD15UNetNode":
        super().train(mode)
        self._resolve_unet()
        if self._unet is not None and hasattr(self._unet, "train"):
            self._unet.train(mode)
        return self

    def trainable_parameters(self):
        self._resolve_unet()
        if self._unet is None or not hasattr(self._unet, "parameters"):
            return iter(())
        return (parameter for parameter in self._unet.parameters() if getattr(parameter, "requires_grad", False))
