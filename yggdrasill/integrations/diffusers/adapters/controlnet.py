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
        cfg.setdefault("guidance_scale", 7.5)
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
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(
                C.PORT_CONTROL_IMAGE,
                PortDirection.IN,
                PortType.IMAGE,
                optional=True,
            ),
            Port(C.PORT_ADD_TEXT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_ADD_TIME_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_SCHEDULER_STATE, PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_DOWN_BLOCK_RESIDUALS, PortDirection.OUT, PortType.ANY),
            Port(C.PORT_MID_BLOCK_RESIDUAL, PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        from yggdrasill.integrations.diffusers.common.image_utils import preprocess_image

        self._controlnet = resolve_if_lazy(self._controlnet)
        if self._controlnet is None:
            raise RuntimeError(
                f"{type(self).__name__}(node_id={self._node_id!r}): controlnet module is None. "
                "Pass controlnet=... when constructing the node or via config. "
                "When using DiffusionGraphBuilder.add_component, ensure pretrained= points to a valid repo."
            )

        latents = inputs[C.PORT_LATENTS]
        p = None
        if hasattr(self._controlnet, "parameters"):
            p = next(self._controlnet.parameters(), None)
        if p is not None:
            model_dtype, model_device = p.dtype, p.device
        elif isinstance(latents, torch.Tensor):
            model_dtype, model_device = latents.dtype, latents.device
        else:
            # Non-torch stubs in tests (e.g. FakeTensor with string dtype/device).
            model_dtype, model_device = torch.float32, torch.device("cpu")

        timestep = inputs[C.PORT_TIMESTEP]
        prompt_embeds = inputs[C.PORT_PROMPT_EMBEDS]
        neg_embeds = inputs.get(C.PORT_NEGATIVE_PROMPT_EMBEDS)
        control_image = inputs.get(C.PORT_CONTROL_IMAGE)
        if control_image is None:
            return {}

        # Match ControlNet weights (often fp16) — graph buffers may still hold fp32 latents.
        latents = latents.to(device=model_device, dtype=model_dtype)
        prompt_embeds = prompt_embeds.to(device=model_device, dtype=model_dtype)
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(device=model_device)
            if timestep.ndim > 0:
                timestep = timestep.reshape(-1)[0]
            if timestep.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                timestep = timestep.long()
        else:
            if hasattr(timestep, "item") and callable(getattr(timestep, "item")):
                try:
                    t_val = int(timestep.item())
                except (TypeError, ValueError):
                    t_val = int(timestep)
            else:
                t_val = int(timestep)
            timestep = torch.tensor(t_val, device=model_device, dtype=torch.long)

        # Match pipeline_controlnet __call__ denoising loop: same latent scaling as UNet
        # (scale_model_input after CFG doubling), else residuals do not match noise_pred batch.
        gs = float(self._config.get("guidance_scale", 7.5))
        do_cfg = gs > 1.0 and neg_embeds is not None
        # pipeline_controlnet: guess_mode = guess_mode or controlnet.config.global_pool_conditions
        guess_mode = bool(self._config.get("guess_mode", False))
        if hasattr(self._controlnet, "config"):
            guess_mode = guess_mode or bool(
                getattr(self._controlnet.config, "global_pool_conditions", False) or False
            )

        # guess_mode + CFG: run ControlNet on conditional batch only; pad residuals with zeros
        # for the unconditional half (see diffusers pipeline_controlnet.py).
        if guess_mode and do_cfg:
            latents_in = latents
            encoder_states = prompt_embeds
        elif do_cfg:
            neg_embeds = neg_embeds.to(device=model_device, dtype=model_dtype)
            latents_in = torch.cat([latents, latents], dim=0)
            encoder_states = torch.cat([neg_embeds, prompt_embeds], dim=0)
        else:
            latents_in = latents
            encoder_states = prompt_embeds

        sched_state = inputs.get(C.PORT_SCHEDULER_STATE)
        if sched_state and isinstance(sched_state, dict):
            sched = sched_state.get("scheduler")
            if sched is not None and hasattr(sched, "scale_model_input"):
                latents_in = sched.scale_model_input(latents_in, timestep)

        conditioning_scale = self._config.get("conditioning_scale", 1.0)

        def _resolved_conditioning_mode() -> str:
            """Canny / depth / … checkpoints expect different conditioning; 'canny' needs edge maps."""
            mode = self._config.get("controlnet_conditioning_mode", "auto")
            if mode != "auto":
                return str(mode)
            hint = str(self._config.get("pretrained", "")).lower()
            if "canny" in hint:
                return "canny"
            return "none"

        cond_mode = _resolved_conditioning_mode()

        if not isinstance(control_image, torch.Tensor):
            height = self._config.get("height", latents_in.shape[-2] * 8)
            width = self._config.get("width", latents_in.shape[-1] * 8)
            dev = str(model_device)
            # Cache key includes model dtype so fp16/fp32 preprocess caches do not clash.
            cache_key = (
                control_image if isinstance(control_image, str) else id(control_image),
                height,
                width,
                str(model_dtype),
                cond_mode,
            )
            if cache_key not in self._control_image_cache:
                # Canny + VAE preprocess must run only once per run, not every denoising step.
                preprocess_src: Any = control_image
                if cond_mode == "canny":
                    from yggdrasill.integrations.diffusers.common.image_utils import (
                        apply_canny_for_controlnet_conditioning,
                    )

                    preprocess_src = apply_canny_for_controlnet_conditioning(
                        control_image,
                        height=height,
                        width=width,
                    )
                # Match pipeline_controlnet: control_image_processor uses do_normalize=False, do_convert_rgb=True
                # (Vae input uses [-1,1]; ControlNet conditioning must stay in [0,1] — wrong range breaks residuals.)
                self._control_image_cache[cache_key] = preprocess_image(
                    preprocess_src,
                    height=height,
                    width=width,
                    dtype=torch.float32,
                    device=dev,
                    do_normalize=False,
                    do_convert_rgb=True,
                )
            control_image = self._control_image_cache[cache_key].to(
                device=model_device, dtype=model_dtype
            )
        else:
            control_image = control_image.to(device=model_device, dtype=model_dtype)

        # prepare_image() in pipeline_controlnet: if CFG and not guess_mode,
        # image = torch.cat([image] * 2) so controlnet_cond batch matches latents_in.
        if (
            do_cfg
            and not guess_mode
            and isinstance(control_image, torch.Tensor)
            and control_image.shape[0] * 2 == latents_in.shape[0]
        ):
            control_image = torch.cat([control_image, control_image], dim=0)

        if isinstance(control_image, torch.Tensor) and control_image.shape[0] != latents_in.shape[0]:
            raise ValueError(
                f"ControlNet conditioning batch ({control_image.shape[0]}) must match "
                f"latent batch ({latents_in.shape[0]}). Under CFG, expected "
                f"{latents_in.shape[0]} (e.g. duplicate the conditioning image like pipeline_controlnet). "
                f"do_cfg={do_cfg}, guess_mode={guess_mode}."
            )

        kwargs: Dict[str, Any] = {
            "return_dict": False,
            "guess_mode": guess_mode,
        }

        added_cond = {}
        if C.PORT_ADD_TEXT_EMBEDS in inputs:
            te = inputs[C.PORT_ADD_TEXT_EMBEDS]
            if isinstance(te, torch.Tensor):
                te = te.to(device=model_device, dtype=model_dtype)
            added_cond["text_embeds"] = te
        if C.PORT_ADD_TIME_IDS in inputs:
            tid = inputs[C.PORT_ADD_TIME_IDS]
            if isinstance(tid, torch.Tensor):
                tid = tid.to(device=model_device)
            added_cond["time_ids"] = tid
        if added_cond:
            kwargs["added_cond_kwargs"] = added_cond

        down_residuals, mid_residual = self._controlnet(
            latents_in,
            timestep,
            encoder_hidden_states=encoder_states,
            controlnet_cond=control_image,
            conditioning_scale=conditioning_scale,
            **kwargs,
        )

        if guess_mode and do_cfg:
            seq = list(down_residuals) if isinstance(down_residuals, (list, tuple)) else [down_residuals]
            down_residuals = tuple(
                torch.cat([torch.zeros_like(d), d], dim=0) for d in seq
            )
            mid_residual = torch.cat(
                [torch.zeros_like(mid_residual), mid_residual], dim=0
            )
        elif isinstance(down_residuals, list):
            # Diffusers returns a list of per-level tensors; merge_residuals must not treat
            # that as multi-adapter CONCAT (which would sum incompatible spatial sizes).
            down_residuals = tuple(down_residuals)

        return {
            C.PORT_DOWN_BLOCK_RESIDUALS: down_residuals,
            C.PORT_MID_BLOCK_RESIDUAL: mid_residual,
        }

    def to(self, device: Any) -> "ControlNetNode":
        if self._controlnet is not None:
            self._controlnet.to(device)
        return self
