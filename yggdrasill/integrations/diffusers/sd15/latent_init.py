"""SD1.5 latent initialization nodes for text2img, img2img, inpaint."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractOuterModule


class SD15LatentInitNode(AbstractOuterModule):
    """Initializes latents for text2img (random noise) or img2img (encoded + noise).

    For text2img: generates random noise scaled by scheduler.init_noise_sigma.
    For img2img: receives pre-encoded latents on ``init_latents``, adds noise.
    """

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
        return "sd15/latent_init"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_SCHEDULER_STATE, PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_INIT_LATENTS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_LATENTS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_TIMESTEP, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        existing_latents = inputs.get(C.PORT_INIT_LATENTS)
        sched_state = inputs.get(C.PORT_SCHEDULER_STATE, {})
        init_noise_sigma = sched_state.get("init_noise_sigma", 1.0) if isinstance(sched_state, dict) else 1.0

        if existing_latents is not None:
            latents = existing_latents * init_noise_sigma
            timestep = self._clamp_timestep(self._get_first_timestep(sched_state))
            return {C.PORT_LATENTS: latents, C.PORT_TIMESTEP: timestep}

        height = self._config.get("height", 512)
        width = self._config.get("width", 512)
        batch_size = self._config.get("batch_size", 1)
        num_channels = self._config.get("num_latent_channels", 4)
        device = self._config.get("device", "cpu")
        dtype_str = self._config.get("dtype", "float16")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, torch.float16)

        shape = (batch_size, num_channels, height // 8, width // 8)
        target_device = device if isinstance(device, (str, torch.device)) else str(device)

        generator = None
        seed = self._config.get("seed")
        if seed is not None:
            generator = torch.Generator(device=target_device).manual_seed(int(seed))

        latents = torch.randn(shape, generator=generator, device=target_device, dtype=dtype) * init_noise_sigma
        timestep = self._clamp_timestep(self._get_first_timestep(sched_state))

        return {C.PORT_LATENTS: latents, C.PORT_TIMESTEP: timestep}

    def _get_first_timestep(self, sched_state: Any):
        """Extract first timestep from scheduler state for the denoising loop."""
        import torch
        if isinstance(sched_state, dict):
            scheduler = sched_state.get("scheduler")
            if scheduler is not None and hasattr(scheduler, "timesteps"):
                timesteps = scheduler.timesteps
                if timesteps is not None and len(timesteps) > 0:
                    return timesteps[0]
        device = self._config.get("device", "cpu")
        return torch.tensor(999, device=device, dtype=torch.long)

    def _clamp_timestep(self, t: Any) -> Any:
        """Clamp timestep to valid range [0, 999] for schedulers with 1000 steps."""
        import torch
        if t is None:
            return torch.tensor(999, device=self._config.get("device", "cpu"), dtype=torch.long)
        if isinstance(t, torch.Tensor):
            return t.clamp(0, 999)
        return max(0, min(999, int(t)))
