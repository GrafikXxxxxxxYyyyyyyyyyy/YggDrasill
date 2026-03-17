"""ControlNet adapter node for SD1.5/SDXL."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInnerModule


class ControlNetNode(AbstractInnerModule):
    """Inner Module: runs ControlNet inside the denoising loop to produce down/mid
    block residuals for the UNet. Executed on each iteration; residuals are passed
    via edges to the UNet node's optional control residual input ports.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        controlnet: Any = None,
    ) -> None:
        cfg = dict(config or {})
        controlnet = controlnet or cfg.pop("controlnet", None)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)
        self._controlnet = controlnet
        self._control_image_cache: Dict[Any, Any] = {}  # (input_id, h, w) -> tensor

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
            dev = str(latents.device)
            # Cache key: use string for URLs/paths (stable), id() for PIL/ndarray
            cache_key = (
                control_image if isinstance(control_image, str) else id(control_image),
                height,
                width,
            )
            if cache_key not in self._control_image_cache:
                self._control_image_cache[cache_key] = preprocess_image(
                    control_image, height=height, width=width, dtype=dtype,
                    device=dev,
                )
            control_image = self._control_image_cache[cache_key].to(
                device=latents.device, dtype=dtype
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

        if self._controlnet is None:
            raise RuntimeError(
                f"{type(self).__name__}(node_id={self._node_id!r}): controlnet module is None. "
                "Pass controlnet=... when constructing the node or via config. "
                "When using DiffusionGraphBuilder.add_component, ensure pretrained= points to a valid repo."
            )

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
