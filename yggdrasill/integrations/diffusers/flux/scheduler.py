"""FLUX scheduler nodes: FlowMatchEulerDiscreteScheduler setup + step."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInnerModule, AbstractOuterModule


class FluxSchedulerSetupNode(AbstractOuterModule):
    """Configures FlowMatchEulerDiscreteScheduler with optional dynamic shifting.

    Supports the ``mu`` parameter for resolution-dependent timestep shifting:
    when ``use_dynamic_shifting`` is True, ``mu`` is linearly interpolated
    from ``base_shift`` to ``max_shift`` based on image sequence length.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        scheduler: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._scheduler = scheduler

    @property
    def block_type(self) -> str:
        return "flux/scheduler_setup"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_TIMESTEPS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_SCHEDULER_STATE, PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        num_steps = self._config.get("num_inference_steps", 28)
        device = self._config.get("device", "cpu")

        set_timesteps_kwargs: Dict[str, Any] = {}

        mu = self._config.get("mu")
        if mu is not None:
            set_timesteps_kwargs["mu"] = mu
        elif self._config.get("use_dynamic_shifting", False):
            height = self._config.get("height", 1024)
            width = self._config.get("width", 1024)
            image_seq_len = (height // 8 // 2) * (width // 8 // 2)
            mu = self._calculate_shift(image_seq_len)
            set_timesteps_kwargs["mu"] = mu

        self._scheduler.set_timesteps(num_steps, device=device, **set_timesteps_kwargs)

        return {
            C.PORT_TIMESTEPS: self._scheduler.timesteps,
            C.PORT_SCHEDULER_STATE: {
                "scheduler": self._scheduler,
                "init_noise_sigma": getattr(self._scheduler, "init_noise_sigma", 1.0),
                "order": getattr(self._scheduler, "order", 1),
            },
        }

    def _calculate_shift(self, image_seq_len: int) -> float:
        sched_cfg = self._scheduler.config if hasattr(self._scheduler, "config") else {}
        base_seq_len = getattr(sched_cfg, "base_image_seq_len", 256)
        max_seq_len = getattr(sched_cfg, "max_image_seq_len", 4096)
        base_shift = getattr(sched_cfg, "base_shift", 0.5)
        max_shift = getattr(sched_cfg, "max_shift", 1.15)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return m * image_seq_len + b

    def to(self, device: Any) -> "FluxSchedulerSetupNode":
        if self._scheduler is not None and hasattr(self._scheduler, "to"):
            self._scheduler.to(device)
        return self


class FluxSchedulerStepNode(AbstractInnerModule):
    """Performs one FlowMatchEulerDiscrete step.

    Flow-matching ODE: prev_sample = sample + (sigma_next - sigma) * model_output.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        scheduler: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._scheduler = scheduler

    @property
    def block_type(self) -> str:
        return "flux/scheduler_step"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PACKED_LATENTS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NOISE_PRED, PortDirection.IN, PortType.TENSOR),
            Port("next_latent", PortDirection.OUT, PortType.TENSOR),
            Port("next_timestep", PortDirection.OUT, PortType.TENSOR, optional=True),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        latents = inputs[C.PORT_PACKED_LATENTS]
        timestep = inputs[C.PORT_TIMESTEP]
        noise_pred = inputs[C.PORT_NOISE_PRED]

        result = self._scheduler.step(noise_pred, timestep, latents, return_dict=False)
        next_latents = result[0] if isinstance(result, (tuple, list)) else result.prev_sample

        return {
            "next_latent": next_latents,
            "next_timestep": timestep,
        }

    def to(self, device: Any) -> "FluxSchedulerStepNode":
        if self._scheduler is not None and hasattr(self._scheduler, "to"):
            self._scheduler.to(device)
        return self
