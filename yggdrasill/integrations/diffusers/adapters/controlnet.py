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
            Port(
                C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS,
                PortDirection.IN,
                PortType.TENSOR,
                optional=True,
            ),
            Port(
                C.PORT_NEGATIVE_ADD_TIME_IDS,
                PortDirection.IN,
                PortType.TENSOR,
                optional=True,
            ),
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
        is_video_latents = isinstance(latents, torch.Tensor) and latents.dim() == 5
        num_frames_video = int(latents.shape[2]) if is_video_latents else 1
        video_base_batch = int(latents.shape[0]) if isinstance(latents, torch.Tensor) else 1

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

        # AnimateDiff (diffusers pipeline_animatediff_controlnet): 5D → repeat prompts per frame, flatten batch.
        if is_video_latents:
            if latents_in.dim() != 5:
                raise ValueError(
                    f"expected 5D latents (B,C,F,H,W) for video ControlNet, got {tuple(latents_in.shape)}"
                )
            b5, c5, f5, h5, w5 = latents_in.shape
            encoder_states = encoder_states.repeat_interleave(f5, dim=0)
            latents_in = latents_in.transpose(1, 2).reshape(b5 * f5, c5, h5, w5)

        conditioning_scale = self._config.get("conditioning_scale", 1.0)
        conditioning_scale = self._apply_control_guidance_window(sched_state, conditioning_scale)

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

        height = self._config.get("height", latents_in.shape[-2] * 8)
        width = self._config.get("width", latents_in.shape[-1] * 8)
        dev = str(model_device)

        if not isinstance(control_image, torch.Tensor):
            # Video: list/tuple of per-frame images (length F), matching AnimateDiff ControlNet pipeline.
            if (
                is_video_latents
                and isinstance(control_image, (list, tuple))
                and not isinstance(control_image, str)
            ):
                cache_key = (
                    "video_frames",
                    len(control_image),
                    height,
                    width,
                    str(model_dtype),
                    cond_mode,
                    tuple(id(x) for x in control_image),
                )
                if cache_key not in self._control_image_cache:
                    frames_t: List[Any] = []
                    for fr in control_image:
                        preprocess_src: Any = fr
                        if cond_mode == "canny":
                            from yggdrasill.integrations.diffusers.common.image_utils import (
                                apply_canny_for_controlnet_conditioning,
                            )

                            preprocess_src = apply_canny_for_controlnet_conditioning(
                                fr, height=height, width=width,
                            )
                        frames_t.append(
                            preprocess_image(
                                preprocess_src,
                                height=height,
                                width=width,
                                dtype=torch.float32,
                                device=dev,
                                do_normalize=False,
                                do_convert_rgb=True,
                            )
                        )
                    self._control_image_cache[cache_key] = torch.cat(frames_t, dim=0)
                control_image = self._control_image_cache[cache_key].to(
                    device=model_device, dtype=model_dtype
                )
            else:
                # Cache key includes model dtype so fp16/fp32 preprocess caches do not clash.
                cache_key = (
                    control_image if isinstance(control_image, str) else id(control_image),
                    height,
                    width,
                    str(model_dtype),
                    cond_mode,
                )
                if cache_key not in self._control_image_cache:
                    preprocess_src = control_image
                    if cond_mode == "canny":
                        from yggdrasill.integrations.diffusers.common.image_utils import (
                            apply_canny_for_controlnet_conditioning,
                        )

                        preprocess_src = apply_canny_for_controlnet_conditioning(
                            control_image,
                            height=height,
                            width=width,
                        )
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

        control_image = self._match_controlnet_cond_batch(
            control_image,
            target_batch=int(latents_in.shape[0]),
            num_frames=num_frames_video,
            video_base_batch=video_base_batch,
            is_video_latents=is_video_latents,
            do_cfg=do_cfg,
            guess_mode=guess_mode,
        )

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

        # SDXL ControlNet must receive the same ``added_cond_kwargs`` layout as the UNet under CFG:
        # ``torch.cat([negative, positive], dim=0)`` for ``text_embeds`` and ``time_ids``
        # (see diffusers ``pipeline_controlnet_sd_xl``). Passing only the positive tensors
        # with doubled latents / encoder states breaks residuals (often black output).
        added_cond: Dict[str, Any] = {}
        te_pos = inputs.get(C.PORT_ADD_TEXT_EMBEDS)
        tid_pos = inputs.get(C.PORT_ADD_TIME_IDS)
        te_neg = inputs.get(C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS)
        tid_neg = inputs.get(C.PORT_NEGATIVE_ADD_TIME_IDS)

        if isinstance(te_pos, torch.Tensor):
            te_p = te_pos.to(device=model_device, dtype=model_dtype)
            tid_p = (
                tid_pos.to(device=model_device, dtype=model_dtype)
                if isinstance(tid_pos, torch.Tensor)
                else None
            )
            if guess_mode and do_cfg:
                added_cond["text_embeds"] = te_p
                if tid_p is not None:
                    added_cond["time_ids"] = tid_p
            elif do_cfg:
                if isinstance(te_neg, torch.Tensor):
                    te_n = te_neg.to(device=model_device, dtype=model_dtype)
                else:
                    te_n = torch.zeros_like(te_p)
                added_cond["text_embeds"] = torch.cat([te_n, te_p], dim=0)
                if tid_p is not None:
                    if isinstance(tid_neg, torch.Tensor):
                        tid_n = tid_neg.to(device=model_device, dtype=model_dtype)
                    else:
                        tid_n = torch.zeros_like(tid_p)
                    added_cond["time_ids"] = torch.cat([tid_n, tid_p], dim=0)
            else:
                added_cond["text_embeds"] = te_p
                if tid_p is not None:
                    added_cond["time_ids"] = tid_p
        elif isinstance(tid_pos, torch.Tensor):
            tid_p = tid_pos.to(device=model_device, dtype=model_dtype)
            if do_cfg and not guess_mode:
                if isinstance(tid_neg, torch.Tensor):
                    tid_n = tid_neg.to(device=model_device, dtype=model_dtype)
                else:
                    tid_n = torch.zeros_like(tid_p)
                added_cond["time_ids"] = torch.cat([tid_n, tid_p], dim=0)
            else:
                added_cond["time_ids"] = tid_p

        if added_cond and is_video_latents:
            f_rep = num_frames_video
            for k, v in list(added_cond.items()):
                if isinstance(v, torch.Tensor):
                    added_cond[k] = v.repeat_interleave(f_rep, dim=0)

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

    def _match_controlnet_cond_batch(
        self,
        control_image: Any,
        *,
        target_batch: int,
        num_frames: int,
        video_base_batch: int,
        is_video_latents: bool,
        do_cfg: bool,
        guess_mode: bool,
    ) -> Any:
        """Match ``controlnet_cond.shape[0]`` to flattened video/control latent batch (AnimateDiff + CFG)."""
        import torch

        if not isinstance(control_image, torch.Tensor):
            return control_image
        ci = control_image
        if ci.dim() != 4:
            raise ValueError(
                f"ControlNet conditioning must be 4D [N,C,H,W]; got dim={ci.dim()} shape={tuple(ci.shape)}"
            )
        n, c, h, w = ci.shape
        if n == target_batch:
            return ci
        if n == 1:
            return ci.expand(target_batch, c, h, w).contiguous()

        if not is_video_latents:
            if do_cfg and not guess_mode and n * 2 == target_batch:
                return torch.cat([ci, ci], dim=0)
        else:
            if num_frames <= 1:
                if do_cfg and not guess_mode and n * 2 == target_batch:
                    return torch.cat([ci, ci], dim=0)
            else:
                if n == num_frames and video_base_batch == 1:
                    if target_batch == num_frames:
                        return ci
                    if target_batch == 2 * num_frames and do_cfg and not guess_mode:
                        return torch.cat([ci, ci], dim=0)
                strip = video_base_batch * num_frames
                if n == strip and target_batch == strip:
                    return ci
                if n == strip and target_batch == 2 * strip and do_cfg and not guess_mode:
                    return torch.cat([ci, ci], dim=0)
                if n == num_frames and video_base_batch > 1 and target_batch == strip:
                    return (
                        ci.unsqueeze(0)
                        .expand(video_base_batch, num_frames, c, h, w)
                        .reshape(-1, c, h, w)
                        .contiguous()
                    )

        raise ValueError(
            f"ControlNet conditioning batch {n} cannot be aligned to target_batch={target_batch} "
            f"(video={is_video_latents}, num_frames={num_frames}, video_base_batch={video_base_batch}, "
            f"do_cfg={do_cfg}, guess_mode={guess_mode}, shape={tuple(ci.shape)})."
        )

    def _apply_control_guidance_window(self, sched_state: Any, scale: Any) -> Any:
        """Match diffusers' control_guidance_start/end via controlnet_keep.

        keep = 1.0 - float(i/L < start or (i+1)/L > end)
        where L = len(timesteps), i = current index in timesteps.
        """
        if scale is None:
            return scale
        start = self._config.get("control_guidance_start", 0.0)
        end = self._config.get("control_guidance_end", 1.0)
        try:
            s = float(start)
            e = float(end)
        except (TypeError, ValueError):
            return scale

        scheduler = None
        if isinstance(sched_state, dict):
            scheduler = sched_state.get("scheduler")
        ts = getattr(scheduler, "timesteps", None) if scheduler is not None else None
        try:
            L = int(len(ts)) if ts is not None else 0
        except Exception:
            L = 0
        if L <= 0:
            return scale
        try:
            i = int(getattr(scheduler, "_yggdrasill_step_idx", 0))
        except Exception:
            i = 0

        keep = 1.0 - float((i / L) < s or ((i + 1) / L) > e)
        try:
            return float(scale) * keep
        except Exception:
            return scale

    def to(self, device: Any) -> "ControlNetNode":
        if self._controlnet is not None:
            self._controlnet.to(device)
        return self
