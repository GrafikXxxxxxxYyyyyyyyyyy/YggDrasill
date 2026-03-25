"""T2I-Adapter node for image-conditioned generation (Diffusers T2IAdapter)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInnerModule


class T2IAdapterNode(AbstractInnerModule):
    """Runs a Diffusers :class:`~diffusers.T2IAdapter` and emits intrablock residuals for the UNet.

    Mirrors diffusers' `StableDiffusion(XL)AdapterPipeline` behavior:
    - preprocess control image to `[0, 1]` tensor BCHW
    - multiply features by `conditioning_scale`
    - under CFG, duplicate features to match doubled UNet batch
    - optionally disable after a fraction of steps (`conditioning_factor`)
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        adapter: Any = None,
    ) -> None:
        cfg = dict(config or {})
        adapter = adapter or cfg.pop("adapter", None)
        cfg.setdefault("guidance_scale", 7.5)
        cfg.setdefault("conditioning_scale", 1.0)
        cfg.setdefault("conditioning_factor", 1.0)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)
        self._adapter = adapter
        self._image_cache: Dict[Any, Any] = {}  # (input_id, h, w, dtype) -> tensor
        self._state_cache: Dict[Any, Any] = {}  # (image_cache_key, dtype, device) -> adapter_state tuple

    @property
    def block_type(self) -> str:
        return "adapter/t2i_adapter"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_T2I_ADAPTER_IMAGE, PortDirection.IN, PortType.IMAGE, optional=True),
            Port(C.PORT_SCHEDULER_STATE, PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_DOWN_INTRABLOCK_RESIDUALS, PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        from yggdrasill.integrations.diffusers.common.image_utils import preprocess_image

        self._adapter = resolve_if_lazy(self._adapter)
        if self._adapter is None:
            raise RuntimeError(
                f"{type(self).__name__}(node_id={self._node_id!r}): adapter module is None. "
                "Pass adapter=... when constructing the node or via DiffusionGraphBuilder.add_component."
            )

        control_image = inputs.get(C.PORT_T2I_ADAPTER_IMAGE)
        if control_image is None:
            return {}

        latents = inputs[C.PORT_LATENTS]
        neg_embeds = inputs.get(C.PORT_NEGATIVE_PROMPT_EMBEDS)

        # Determine dtype/device from adapter parameters (preferred).
        p = None
        if hasattr(self._adapter, "parameters"):
            p = next(self._adapter.parameters(), None)
        if p is not None:
            model_dtype, model_device = p.dtype, p.device
        elif isinstance(latents, torch.Tensor):
            model_dtype, model_device = latents.dtype, latents.device
        else:
            model_dtype, model_device = torch.float32, torch.device("cpu")

        # CFG alignment: T2I-Adapter features must match UNet batch (doubled under CFG).
        gs = float(self._config.get("guidance_scale", 7.5))
        do_cfg = gs > 1.0 and neg_embeds is not None

        # Preprocess adapter image to [0,1], BCHW, float32 like diffusers.
        # IMPORTANT: many SD1.5 T2I-Adapters are trained with in_channels=1 (grayscale).
        # In that case PixelUnshuffle expects 1 channel (→ 64 channels for factor 8), not RGB (→ 192).
        in_ch = int(getattr(getattr(self._adapter, "config", None), "in_channels", 3) or 3)
        height = self._config.get("height")
        width = self._config.get("width")
        if height is None or width is None:
            # Best-effort infer from latents (8x VAE scale).
            if hasattr(latents, "shape") and len(getattr(latents, "shape", ())) >= 4:
                height = int(latents.shape[-2]) * 8
                width = int(latents.shape[-1]) * 8
            else:
                height = 1024
                width = 1024

        cache_key = (
            control_image if isinstance(control_image, str) else id(control_image),
            int(height),
            int(width),
            "float32",
            in_ch,
        )
        if cache_key not in self._image_cache:
            self._image_cache[cache_key] = preprocess_image(
                control_image,
                height=int(height),
                width=int(width),
                dtype=torch.float32,
                device=str(model_device),
                do_normalize=False,
                do_convert_rgb=(in_ch != 1),
            )
        adapter_input = self._image_cache[cache_key]
        if isinstance(adapter_input, torch.Tensor):
            # If adapter expects grayscale but we received 3ch tensor, convert to 1ch.
            if in_ch == 1 and adapter_input.ndim == 4 and adapter_input.shape[1] == 3:
                adapter_input = adapter_input.mean(dim=1, keepdim=True)
            # If adapter expects RGB but we received 1ch tensor, repeat.
            if in_ch == 3 and adapter_input.ndim == 4 and adapter_input.shape[1] == 1:
                adapter_input = adapter_input.repeat(1, 3, 1, 1)
        adapter_input = adapter_input.to(device=model_device, dtype=getattr(self._adapter, "dtype", model_dtype))

        # Cache adapter_state (timestep-independent) to avoid recomputing every denoise step.
        state_key = (cache_key, str(model_dtype), str(model_device))
        if state_key not in self._state_cache:
            state = self._adapter(adapter_input)
            # Diffusers returns list; we keep an immutable tuple for safe caching.
            if isinstance(state, list):
                state = tuple(state)
            self._state_cache[state_key] = state
        adapter_state = self._state_cache[state_key]

        # Apply conditioning scale.
        cond_scale = float(self._config.get("conditioning_scale", 1.0))
        scaled: List[Any] = []
        for v in adapter_state:
            if isinstance(v, torch.Tensor):
                scaled.append(v * cond_scale)
            else:
                try:
                    scaled.append(v * cond_scale)
                except Exception:
                    scaled.append(v)

        # Gating window: only apply for early fraction of steps (diffusers adapter_conditioning_factor).
        factor = self._config.get("conditioning_factor", 1.0)
        try:
            f = float(factor)
        except (TypeError, ValueError):
            f = 1.0
        sched_state = inputs.get(C.PORT_SCHEDULER_STATE)
        scheduler = sched_state.get("scheduler") if isinstance(sched_state, dict) else None
        timesteps = getattr(scheduler, "timesteps", None) if scheduler is not None else None
        L = int(len(timesteps)) if timesteps is not None else 0
        i = int(getattr(scheduler, "_yggdrasill_step_idx", 0)) if scheduler is not None else 0
        if L > 0 and i >= int(L * f):
            return {}

        # Match diffusers: clone per step (UNet may modify in-place).
        out_state: List[Any] = []
        for v in scaled:
            if isinstance(v, torch.Tensor):
                out_state.append(v.clone())
            else:
                out_state.append(v)

        # num_images_per_prompt: in graph world, batch size should already match;
        # CFG duplication is handled here to align with UNet's internal doubling.
        if do_cfg:
            out_state = [
                torch.cat([v, v], dim=0) if isinstance(v, torch.Tensor) else v
                for v in out_state
            ]

        return {C.PORT_DOWN_INTRABLOCK_RESIDUALS: tuple(out_state)}

    def to(self, device: Any) -> "T2IAdapterNode":
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        self._adapter = resolve_if_lazy(self._adapter)
        if self._adapter is not None and hasattr(self._adapter, "to"):
            self._adapter.to(device)
        return self

