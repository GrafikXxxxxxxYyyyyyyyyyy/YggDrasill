"""FLUX latent initialization with 2x2 patch packing and position IDs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractOuterModule


class FluxLatentInitNode(AbstractOuterModule):
    """Initializes packed latents and image position IDs for FLUX.

    FLUX packs latents from [B, C, H, W] into [B, (H/2)*(W/2), C*4]
    using 2x2 spatial patches. Also generates img_ids [num_patches, 3]
    containing (batch_idx, row, col) coordinates for RoPE.

    For text2img: generates random noise then packs.
    For img2img: receives encoded latents, applies flow-matching noise
    scaling via scheduler.scale_noise(), then packs.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = dict(config or {})
        cfg.setdefault("height", 1024)
        cfg.setdefault("width", 1024)
        cfg.setdefault("num_latent_channels", 16)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)

    @property
    def block_type(self) -> str:
        return "flux/latent_init"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_SCHEDULER_STATE, PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_INIT_LATENTS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_PACKED_LATENTS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_IMG_IDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        height = self._config.get("height", 1024)
        width = self._config.get("width", 1024)
        batch_size = self._config.get("batch_size", 1)
        num_channels = self._config.get("num_latent_channels", 16)
        device = self._config.get("device", "cpu")
        dtype_str = self._config.get("dtype", "bfloat16")

        dtype_map = {
            "float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16, "fp32": torch.float32,
            "fp16": torch.float16, "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        latent_h = height // 8
        latent_w = width // 8

        existing = inputs.get(C.PORT_INIT_LATENTS)
        sched_state = inputs.get(C.PORT_SCHEDULER_STATE, {})
        scheduler = sched_state.get("scheduler") if isinstance(sched_state, dict) else None

        if existing is not None:
            latents = existing
            strength = self._config.get("strength", 1.0)
            if scheduler is not None and strength < 1.0 and hasattr(scheduler, "scale_noise"):
                noise = torch.randn_like(latents)
                timesteps = sched_state.get("scheduler", scheduler).timesteps if isinstance(sched_state, dict) else None
                if timesteps is not None and len(timesteps) > 0:
                    t = timesteps[0]
                    latents = scheduler.scale_noise(latents, t, noise)
        else:
            shape = (batch_size, num_channels, latent_h, latent_w)
            generator = None
            seed = self._config.get("seed")
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)
            latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
            latents = latents.to(device)

        packed = self._pack_latents(latents)
        img_ids = self._prepare_img_ids(batch_size, latent_h // 2, latent_w // 2, device, dtype)

        return {
            C.PORT_PACKED_LATENTS: packed,
            C.PORT_IMG_IDS: img_ids,
        }

    @staticmethod
    def _pack_latents(latents: Any) -> Any:
        """Pack [B, C, H, W] -> [B, (H/2)*(W/2), C*4] via 2x2 patches."""
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    @staticmethod
    def _prepare_img_ids(batch_size: int, h: int, w: int, device: Any, dtype: Any) -> Any:
        """Generate image position IDs [h*w, 3] for RoPE."""
        import torch

        img_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        img_ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        img_ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        img_ids = img_ids.reshape(h * w, 3)
        return img_ids
