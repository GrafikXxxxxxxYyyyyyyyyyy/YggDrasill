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
        if not isinstance(sched_state, dict):
            sched_state = {}
        init_noise_sigma = sched_state.get("init_noise_sigma", 1.0)

        if existing_latents is not None:
            return self._forward_from_encoded_latents(
                existing_latents, sched_state, init_noise_sigma,
            )

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

        noise = torch.randn(shape, generator=generator, device=target_device, dtype=dtype)
        sig = init_noise_sigma
        if isinstance(sig, torch.Tensor):
            latents = noise * sig.to(device=target_device, dtype=dtype)
        else:
            latents = noise * float(sig)
        timestep = self._clamp_timestep(self._get_first_timestep(sched_state))

        if self._config.get("inpaint_4ch_composite"):
            sched_state["_inpaint_blend_noise"] = noise

        return {C.PORT_LATENTS: latents, C.PORT_TIMESTEP: timestep}

    def _forward_from_encoded_latents(
        self,
        existing_latents: Any,
        sched_state: Dict[str, Any],
        init_noise_sigma: float,
    ) -> Dict[str, Any]:
        """img2img path: match Diffusers ``get_timesteps`` + ``add_noise`` (see SD img2img pipeline)."""
        import torch

        scheduler = sched_state.get("scheduler")
        try:
            use_torch = torch.is_tensor(existing_latents)
        except Exception:
            use_torch = False

        # Mocks / tests without real tensors: keep legacy scale-only behaviour.
        if not use_torch or scheduler is None or not hasattr(scheduler, "add_noise"):
            latents = existing_latents * init_noise_sigma
            timestep = self._clamp_timestep(self._get_first_timestep(sched_state))
            return {C.PORT_LATENTS: latents, C.PORT_TIMESTEP: timestep}

        strength = float(self._config.get("strength", 0.8))
        if strength <= 0 or strength > 1.0:
            raise ValueError(f"strength must be in (0, 1], got {strength}")

        timesteps = getattr(scheduler, "timesteps", None)
        if timesteps is None:
            latents = existing_latents * init_noise_sigma
            timestep = self._clamp_timestep(self._get_first_timestep(sched_state))
            return {C.PORT_LATENTS: latents, C.PORT_TIMESTEP: timestep}

        n = len(timesteps)
        if n == 0:
            latents = existing_latents * init_noise_sigma
            timestep = self._clamp_timestep(self._get_first_timestep(sched_state))
            return {C.PORT_LATENTS: latents, C.PORT_TIMESTEP: timestep}

        # StableDiffusionImg2ImgPipeline.get_timesteps
        init_timestep = min(int(n * strength), n)
        if strength > 0 and init_timestep == 0:
            init_timestep = 1
        t_start = max(n - init_timestep, 0)
        order = int(getattr(scheduler, "order", 1))
        start_idx = t_start * order

        if isinstance(timesteps, torch.Tensor):
            new_ts = timesteps[start_idx:].clone()
            scheduler.timesteps = new_ts
        else:
            new_ts = timesteps[start_idx:]
            scheduler.timesteps = new_ts

        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(start_idx)

        sched_state["num_loop_steps"] = len(scheduler.timesteps)

        device = existing_latents.device
        dtype = existing_latents.dtype
        generator = None
        seed = self._config.get("seed")
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        # EulerDiscreteScheduler.add_noise reads timesteps.shape[0]; indexing
        # scheduler.timesteps[0] yields a 0-d tensor, and shape[0] raises IndexError.
        ts_line = scheduler.timesteps
        first_t = ts_line[0]
        batch = int(existing_latents.shape[0])
        if isinstance(ts_line, torch.Tensor):
            timesteps_for_noise = ts_line[0].expand(batch).to(device=device)
        else:
            v0 = ts_line[0]
            tdtype = torch.float32 if isinstance(v0, float) else torch.long
            timesteps_for_noise = torch.full((batch,), v0, device=device, dtype=tdtype)

        # StableDiffusion(Inpaint)Pipeline.prepare_latents: strength == 1.0 → pure noise × init_noise_sigma
        if strength >= 1.0:
            noise = torch.randn(
                existing_latents.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            sig = init_noise_sigma
            if isinstance(sig, torch.Tensor):
                latents = noise * sig.to(device=device, dtype=dtype)
            else:
                latents = noise * float(sig)
            timestep = self._clamp_timestep(first_t)
            if self._config.get("inpaint_4ch_composite"):
                sched_state["_inpaint_blend_noise"] = noise
            return {C.PORT_LATENTS: latents, C.PORT_TIMESTEP: timestep}

        noise = torch.randn(
            existing_latents.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latents = scheduler.add_noise(existing_latents, noise, timesteps_for_noise)
        timestep = self._clamp_timestep(first_t)
        if self._config.get("inpaint_4ch_composite"):
            sched_state["_inpaint_blend_noise"] = noise
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
