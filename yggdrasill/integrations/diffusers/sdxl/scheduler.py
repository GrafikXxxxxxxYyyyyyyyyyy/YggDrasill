"""SDXL scheduler nodes: setup + per-step transition.

Implementation is intentionally duplicated from sd15/scheduler.py (not imported) so
the sdxl package stays independent of sd15. Keep algorithmic changes in sync manually
if the classic LDM scheduler contract evolves.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInnerModule, AbstractOuterModule


class SDXLSchedulerSetupNode(AbstractOuterModule):
    """Configures scheduler timesteps and optional denoising_start/end slicing (SDXL)."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        scheduler: Any = None,
    ) -> None:
        cfg = dict(config or {})
        self._scheduler = scheduler or cfg.pop("scheduler", None)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)

    @property
    def block_type(self) -> str:
        return "sdxl/scheduler_setup"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_TIMESTEPS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_SCHEDULER_STATE, PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._scheduler = resolve_if_lazy(self._scheduler)

        num_steps = self._config.get("num_inference_steps", 50)
        custom_timesteps = self._config.get("timesteps")
        custom_sigmas = self._config.get("sigmas")
        device = self._config.get("device", "cpu")

        kwargs: Dict[str, Any] = {}
        if custom_timesteps is not None:
            kwargs["timesteps"] = custom_timesteps
        elif custom_sigmas is not None:
            kwargs["sigmas"] = custom_sigmas

        self._scheduler.set_timesteps(num_steps, device=device, **kwargs)

        timesteps = self._scheduler.timesteps
        setattr(self._scheduler, "_yggdrasill_step_idx", 0)

        result: Dict[str, Any] = {
            C.PORT_TIMESTEPS: timesteps,
            C.PORT_SCHEDULER_STATE: {
                "scheduler": self._scheduler,
                "init_noise_sigma": getattr(self._scheduler, "init_noise_sigma", 1.0),
                "order": getattr(self._scheduler, "order", 1),
                "num_loop_steps": len(timesteps) if timesteps is not None else num_steps,
                "_inpaint_blend_i": 0,
            },
        }

        denoising_start = self._config.get("denoising_start")
        denoising_end = self._config.get("denoising_end")

        if denoising_start is not None or denoising_end is not None:
            ts = result[C.PORT_TIMESTEPS]
            total = len(ts)
            start_idx = int(total * (denoising_start or 0.0))
            end_idx = int(total * (denoising_end or 1.0))
            result[C.PORT_TIMESTEPS] = ts[start_idx:end_idx]

        return result

    def to(self, device: Any) -> "SDXLSchedulerSetupNode":
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        self._scheduler = resolve_if_lazy(self._scheduler)
        if self._scheduler is not None and hasattr(self._scheduler, "to"):
            self._scheduler.to(device)
        return self


class SDXLSchedulerStepNode(AbstractInnerModule):
    """Performs one scheduler step (same LDM contract as SD1.5; standalone copy)."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        scheduler: Any = None,
    ) -> None:
        cfg = dict(config or {})
        self._scheduler = scheduler or cfg.pop("scheduler", None)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)

    @property
    def block_type(self) -> str:
        return "sdxl/scheduler_step"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NOISE_PRED, PortDirection.IN, PortType.TENSOR),
            Port("next_latent", PortDirection.OUT, PortType.TENSOR),
            Port("next_timestep", PortDirection.OUT, PortType.TENSOR, optional=True),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._scheduler = resolve_if_lazy(self._scheduler)

        latents = inputs[C.PORT_LATENTS]
        timestep_in = inputs[C.PORT_TIMESTEP]
        noise_pred = inputs[C.PORT_NOISE_PRED]

        from yggdrasill.integrations.diffusers.common.scheduler_step import (
            coerce_timestep_for_scheduler_step,
            scheduler_uses_float_timestep_in_step,
        )

        if scheduler_uses_float_timestep_in_step(self._scheduler):
            t = coerce_timestep_for_scheduler_step(timestep_in, latents, self._scheduler)
        else:
            timestep = self._clamp_timestep(timestep_in)
            t = timestep
            if hasattr(t, "item"):
                t = int(t.item())
            elif t is not None:
                t = int(t)

        eta = self._config.get("eta", 0.0)
        generator = self._config.get("generator")

        step_kwargs: Dict[str, Any] = {}
        if eta > 0:
            step_kwargs["eta"] = eta
        if generator is not None:
            step_kwargs["generator"] = generator

        result = self._scheduler.step(
            noise_pred, t, latents,
            return_dict=False,
            **step_kwargs,
        )
        next_latents = result[0] if isinstance(result, (tuple, list)) else result.prev_sample
        if scheduler_uses_float_timestep_in_step(self._scheduler):
            next_timestep = self._get_next_timestep(timestep_in)
        else:
            next_timestep = self._clamp_timestep(self._get_next_timestep(timestep))

        return {
            "next_latent": next_latents,
            "next_timestep": next_timestep,
        }

    def to(self, device: Any) -> "SDXLSchedulerStepNode":
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        self._scheduler = resolve_if_lazy(self._scheduler)
        if self._scheduler is not None and hasattr(self._scheduler, "to"):
            self._scheduler.to(device)
        return self

    def _get_next_timestep(self, timestep: Any) -> Any:
        if not hasattr(self._scheduler, "timesteps") or self._scheduler.timesteps is None:
            return timestep
        ts = self._scheduler.timesteps
        if len(ts) == 0:
            return timestep
        i = int(getattr(self._scheduler, "_yggdrasill_step_idx", 0))
        setattr(self._scheduler, "_yggdrasill_step_idx", i + 1)
        if i + 1 < len(ts):
            return ts[i + 1]
        return timestep

    def _clamp_timestep(self, t: Any) -> Any:
        import torch
        if t is None:
            return None
        if hasattr(t, "clamp"):
            t = t.clamp(0, 999)
            ndim = t.ndim if hasattr(t, "ndim") else (t.dim() if hasattr(t, "dim") else 0)
            if ndim > 0 and hasattr(t, "flatten"):
                t = t.flatten()[0]
            return t
        try:
            return max(0, min(999, int(t)))
        except (TypeError, ValueError):
            return t

    def scale_model_input(self, latents: Any, timestep: Any) -> Any:
        return self._scheduler.scale_model_input(latents, timestep)
