"""SD1.5 VAE decode must float32-decode when config.force_upcast (same as diffusers SD pipelines)."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEDecodeNode


def test_sd15_vae_decode_force_upcast_passes_float32_latents_to_decode() -> None:
    class MiniVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.post_quant_conv = nn.Conv2d(4, 4, 1)

        def decode(self, latents: torch.Tensor, return_dict: bool = False):
            raise AssertionError("patched below")

    vae = MiniVAE().half()
    vae.config = SimpleNamespace(scaling_factor=0.18215, shift_factor=None, force_upcast=True)

    seen: list = []

    def decode_patch(latents: torch.Tensor, return_dict: bool = False):
        seen.append(latents.dtype)
        x = torch.zeros(
            latents.shape[0],
            3,
            latents.shape[2],
            latents.shape[3],
            device=latents.device,
            dtype=latents.dtype,
        )
        return (x,)

    vae.decode = decode_patch  # type: ignore[method-assign]

    node = SD15VAEDecodeNode("V", vae=vae)
    node.forward({C.PORT_LATENTS: torch.randn(1, 4, 8, 8, dtype=torch.float16, device="cpu")})

    assert seen, "decode was not called"
    assert seen[0] == torch.float32
    assert next(vae.parameters()).dtype == torch.float16
