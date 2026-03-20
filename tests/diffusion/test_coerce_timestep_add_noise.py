"""Timestep dtype for scheduler.add_noise (4-ch inpaint blend vs Euler / DDIM)."""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

from yggdrasill.integrations.diffusers.common.scheduler_step import coerce_timestep_for_add_noise


def test_coerce_add_noise_float_when_timesteps_tensor_float() -> None:
    sched = MagicMock()
    sched.__class__.__name__ = "EulerDiscreteScheduler"
    sched.timesteps = torch.tensor([999.0, 888.0], dtype=torch.float32)
    t = coerce_timestep_for_add_noise(888.0, scheduler=sched, device=torch.device("cpu"))
    assert t.shape == (1,)
    assert t.dtype == torch.float32


def test_coerce_add_noise_long_when_timesteps_tensor_long() -> None:
    sched = MagicMock()
    sched.__class__.__name__ = "DDIMScheduler"
    sched.timesteps = torch.tensor([50, 40], dtype=torch.long)
    t = coerce_timestep_for_add_noise(40, scheduler=sched, device=torch.device("cpu"))
    assert t.shape == (1,)
    assert t.dtype == torch.long


def test_inpaint_blend_uses_float_for_euler_timesteps() -> None:
    from yggdrasill.integrations.diffusers.sd15.inpaint_blend import SD15InpaintFourChannelBlendNode

    node = SD15InpaintFourChannelBlendNode("b", config={})
    sched = MagicMock()
    sched.__class__.__name__ = "EulerDiscreteScheduler"
    sched.timesteps = torch.tensor([10.0, 5.0], dtype=torch.float32)

    def add_noise(clean, noise, t):
        assert t.dtype == torch.float32, t.dtype
        assert t.shape == (1,)
        return clean

    sched.add_noise = add_noise

    clean = torch.zeros(1, 4, 2, 2)
    latents = torch.ones(1, 4, 2, 2)
    mask = torch.ones(1, 1, 2, 2)
    noise = torch.randn(1, 4, 2, 2)
    state = {
        "scheduler": sched,
        "_inpaint_blend_noise": noise,
        "_inpaint_blend_i": 0,
    }
    out = node.forward(
        {
            "latents_post_step": latents,
            "clean_image_latents": clean,
            "mask_latents": mask,
            "scheduler_state": state,
        }
    )
    assert "next_latent" in out
