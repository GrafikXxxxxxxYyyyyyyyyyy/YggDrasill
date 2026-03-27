"""4-channel UNet inpaint: latent compositing after each scheduler step.

Shared by SD1.5 and SDXL when ``unet.config.in_channels == 4`` (Diffusers inpaint
pipelines): preserved regions follow noised clean latents; masked regions use the
denoised sample from ``scheduler.step``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.common.scheduler_step import coerce_timestep_for_add_noise
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInnerModule


class InpaintLatentBlendNode(AbstractInnerModule):
    """Runs inside the denoising loop after the scheduler step node."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)

    @property
    def block_type(self) -> str:
        return "common/inpaint_latent_blend"

    def declare_ports(self) -> List[Port]:
        return [
            Port("latents_post_step", PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_CLEAN_IMAGE_LATENTS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_MASK_LATENTS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_SCHEDULER_STATE, PortDirection.IN, PortType.ANY, optional=True),
            Port("next_latent", PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        latents = inputs["latents_post_step"]
        clean = inputs.get(C.PORT_CLEAN_IMAGE_LATENTS)
        mask = inputs.get(C.PORT_MASK_LATENTS)
        sched_state = inputs.get(C.PORT_SCHEDULER_STATE) or {}
        if not isinstance(sched_state, dict):
            sched_state = {}

        noise = sched_state.get("_inpaint_blend_noise")
        scheduler = sched_state.get("scheduler")
        if noise is None or scheduler is None or not hasattr(scheduler, "timesteps"):
            return {"next_latent": latents}

        timesteps = scheduler.timesteps
        n = len(timesteps) if timesteps is not None else 0
        if n == 0:
            return {"next_latent": latents}

        i = int(sched_state.get("_inpaint_blend_i", 0))
        device, dtype = latents.device, latents.dtype
        if mask is None:
            mask = torch.ones_like(latents, device=device, dtype=dtype)
        else:
            mask = mask.to(device=device, dtype=dtype)
        if clean is None:
            clean = torch.zeros_like(latents)
        else:
            clean = clean.to(device=device, dtype=dtype)
        noise = noise.to(device=device, dtype=dtype)

        init_latents_proper = clean
        if i < n - 1:
            t_next = coerce_timestep_for_add_noise(
                timesteps[i + 1], scheduler=scheduler, device=device
            )
            if hasattr(scheduler, "scale_noise"):
                init_latents_proper = scheduler.scale_noise(clean, t_next, noise)
            else:
                init_latents_proper = scheduler.add_noise(clean, noise, t_next)

        out = (1.0 - mask) * init_latents_proper + mask * latents
        sched_state["_inpaint_blend_i"] = i + 1
        return {"next_latent": out}
