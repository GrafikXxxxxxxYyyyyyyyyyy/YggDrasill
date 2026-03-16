"""ControlNet adapter node for SD1.5/SDXL."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInjector


class ControlNetNode(AbstractInjector):
    """Runs ControlNet to produce down/mid block residuals for the UNet.

    The residuals are passed via edges to the UNet node's optional
    control residual input ports.
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
        return "adapter/controlnet"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_CONTROL_IMAGE, PortDirection.IN, PortType.IMAGE),
            Port(C.PORT_ADD_TEXT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_ADD_TIME_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_DOWN_BLOCK_RESIDUALS, PortDirection.OUT, PortType.ANY),
            Port(C.PORT_MID_BLOCK_RESIDUAL, PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from yggdrasill.integrations.diffusers.common.image_utils import preprocess_image

        latents = inputs[C.PORT_LATENTS]
        timestep = inputs[C.PORT_TIMESTEP]
        encoder_states = inputs[C.PORT_PROMPT_EMBEDS]
        control_image = inputs[C.PORT_CONTROL_IMAGE]

        conditioning_scale = self._config.get("conditioning_scale", 1.0)
        guess_mode = self._config.get("guess_mode", False)

        if not isinstance(control_image, torch.Tensor):
            height = self._config.get("height", latents.shape[-2] * 8)
            width = self._config.get("width", latents.shape[-1] * 8)
            dtype = latents.dtype
            control_image = preprocess_image(
                control_image, height=height, width=width, dtype=dtype,
                device=str(latents.device),
            )

        kwargs: Dict[str, Any] = {
            "return_dict": False,
            "guess_mode": guess_mode,
        }

        added_cond = {}
        if C.PORT_ADD_TEXT_EMBEDS in inputs:
            added_cond["text_embeds"] = inputs[C.PORT_ADD_TEXT_EMBEDS]
        if C.PORT_ADD_TIME_IDS in inputs:
            added_cond["time_ids"] = inputs[C.PORT_ADD_TIME_IDS]
        if added_cond:
            kwargs["added_cond_kwargs"] = added_cond

        down_residuals, mid_residual = self._controlnet(
            latents,
            timestep,
            encoder_hidden_states=encoder_states,
            controlnet_cond=control_image,
            conditioning_scale=conditioning_scale,
            **kwargs,
        )

        return {
            C.PORT_DOWN_BLOCK_RESIDUALS: down_residuals,
            C.PORT_MID_BLOCK_RESIDUAL: mid_residual,
        }

    def to(self, device: Any) -> "ControlNetNode":
        if self._controlnet is not None:
            self._controlnet.to(device)
        return self
