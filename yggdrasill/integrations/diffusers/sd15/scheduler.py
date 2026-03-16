"""SD1.5 scheduler nodes: setup + per-step transition."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractOuterModule, AbstractInnerModule


class SD15SchedulerSetupNode(AbstractOuterModule):
    """Configures scheduler timesteps and provides initial scheduler state.

    This is an outer module that runs once before the denoising loop.
    """

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
        return "sd15/scheduler_setup"

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
        return {
            C.PORT_TIMESTEPS: timesteps,
            C.PORT_SCHEDULER_STATE: {
                "scheduler": self._scheduler,
                "init_noise_sigma": getattr(self._scheduler, "init_noise_sigma", 1.0),
                "order": getattr(self._scheduler, "order", 1),
                "num_loop_steps": len(timesteps) if timesteps is not None else num_steps,
            },
        }


class SD15SchedulerStepNode(AbstractInnerModule):
    """Performs one scheduler step: scale_model_input + step.

    This is an inner module inside the denoising loop.
    """

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
        return "sd15/scheduler_step"

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
        timestep = self._clamp_timestep(inputs[C.PORT_TIMESTEP])
        noise_pred = inputs[C.PORT_NOISE_PRED]

        # PNDM expects int for indexing; diffusers passes scalar from timesteps tensor
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
        next_timestep = self._clamp_timestep(self._get_next_timestep(timestep))

        return {
            "next_latent": next_latents,
            "next_timestep": next_timestep,
        }

    def _get_next_timestep(self, timestep: Any) -> Any:
        """Return the next timestep in the scheduler sequence, or current if last.

        PNDM with skip_prk_steps produces duplicates (e.g. [981, 961, 961, 941, ...]).
        Use scheduler.counter (incremented by step()) as the index - matches diffusers
        ``for i, t in enumerate(timesteps)`` semantics.
        """
        if not hasattr(self._scheduler, "timesteps") or self._scheduler.timesteps is None:
            return timestep
        ts = self._scheduler.timesteps
        if len(ts) == 0:
            return timestep
        # After step(), counter is 1-based (number of steps completed)
        idx = getattr(self._scheduler, "counter", 0)
        if idx < len(ts):
            return ts[idx]
        return timestep

    def _clamp_timestep(self, t: Any) -> Any:
        """Clamp timestep to valid range [0, 999] for schedulers with 1000 steps."""
        import torch
        if t is None:
            return None
        if hasattr(t, "clamp"):  # Tensor-like (incl. FakeTensor)
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
        """Public helper for nodes that need to scale before UNet."""
        return self._scheduler.scale_model_input(latents, timestep)
