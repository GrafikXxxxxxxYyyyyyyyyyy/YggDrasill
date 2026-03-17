"""FLUX ControlNet adapter node."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInnerModule


class FluxControlNetNode(AbstractInnerModule):
    """Inner Module: wraps FluxControlNetModel to produce block samples for the
    transformer. Executed inside the denoising loop on each iteration.

    FLUX ControlNet produces two sets of residuals:
    - controlnet_block_samples: for joint (MMDiT) transformer blocks
    - controlnet_single_block_samples: for single transformer blocks

    Supports FluxControlNetModel and FluxMultiControlNetModel.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        controlnet: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._controlnet = controlnet

    @property
    def block_type(self) -> str:
        return "adapter/controlnet_flux"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PACKED_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_POOLED_PROJECTIONS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_IMG_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TXT_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_GUIDANCE, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_CONTROL_IMAGE, PortDirection.IN, PortType.IMAGE),
            Port(C.PORT_CONTROLNET_BLOCK_SAMPLES, PortDirection.OUT, PortType.ANY),
            Port(C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES, PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        latents = inputs[C.PORT_PACKED_LATENTS]
        timestep = inputs[C.PORT_TIMESTEP]
        prompt_embeds = inputs[C.PORT_PROMPT_EMBEDS]
        pooled_projections = inputs[C.PORT_POOLED_PROJECTIONS]
        img_ids = inputs[C.PORT_IMG_IDS]
        txt_ids = inputs[C.PORT_TXT_IDS]
        control_image = inputs[C.PORT_CONTROL_IMAGE]

        conditioning_scale = self._config.get("controlnet_conditioning_scale", 1.0)

        kwargs: Dict[str, Any] = {
            "hidden_states": latents,
            "timestep": timestep / 1000,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_projections,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "controlnet_cond": control_image,
            "conditioning_scale": conditioning_scale,
            "return_dict": False,
        }

        guidance = inputs.get(C.PORT_GUIDANCE)
        if guidance is not None:
            kwargs["guidance"] = guidance

        output = self._controlnet(**kwargs)
        block_samples = output[0] if isinstance(output, (tuple, list)) else output.controlnet_block_samples
        single_samples = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else getattr(output, "controlnet_single_block_samples", None)

        return {
            C.PORT_CONTROLNET_BLOCK_SAMPLES: block_samples,
            C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES: single_samples,
        }

    def to(self, device: Any) -> "FluxControlNetNode":
        if self._controlnet is not None:
            self._controlnet.to(device)
        return self
