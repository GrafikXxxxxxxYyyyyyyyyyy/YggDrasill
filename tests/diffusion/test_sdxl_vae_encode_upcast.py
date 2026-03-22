"""SDXL VAE encode must float32-encode when config.force_upcast (matches diffusers img2img prepare_latents)."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from yggdrasill.integrations.diffusers import contracts as C
from PIL import Image

from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEEncodeNode


def test_sdxl_vae_encode_force_upcast_runs_encoder_in_float32() -> None:
    class MiniVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Conv2d(3, 4, 1)

        def encode(self, x: torch.Tensor):
            seen.append(x.dtype)
            assert x.dtype == torch.float32

            class D:
                def sample(self_inner):
                    return torch.zeros(
                        x.shape[0], 4, x.shape[2] // 8, x.shape[3] // 8,
                        device=x.device, dtype=torch.float32,
                    )

            return SimpleNamespace(latent_dist=D())

    seen: list = []
    vae = MiniVAE().half()
    vae.config = SimpleNamespace(
        scaling_factor=0.13025,
        shift_factor=None,
        force_upcast=True,
    )

    node = SDXLVAEEncodeNode("E", vae=vae, config={"device": "cpu", "height": 64, "width": 64})
    out = node.forward({C.PORT_INIT_IMAGE: Image.new("RGB", (64, 64), color=(128, 64, 32))})

    assert seen and seen[0] == torch.float32
    assert out[C.PORT_LATENTS].dtype == torch.float16
    assert next(vae.parameters()).dtype == torch.float16
