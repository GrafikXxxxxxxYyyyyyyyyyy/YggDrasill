"""AnimateDiff (sd15.motionadapter) builder and 5D latent / VAE decode paths."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yggdrasill.integrations.diffusers import contracts as C  # noqa: E402
from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode  # noqa: E402
from yggdrasill.integrations.diffusers.components import resolve_component_type  # noqa: E402
from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode  # noqa: E402
from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEDecodeNode  # noqa: E402


def test_sd15_motionadapter_component_registered() -> None:
    spec = resolve_component_type("sd15.motionadapter")
    assert spec.block_types == []
    assert spec.load_keys == []


def test_latent_init_num_frames_produces_5d() -> None:
    n = SD15LatentInitNode(
        "latent_init",
        config={
            "height": 64,
            "width": 64,
            "device": "cpu",
            "dtype": "float32",
            "num_frames": 5,
            "batch_size": 1,
        },
    )
    out = n.forward({"scheduler_state": {"init_noise_sigma": 1.0, "scheduler": None}})
    assert out["latents"].shape == (1, 4, 5, 8, 8)


def test_vae_decode_5d_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    """5D latents are flattened to (B*F,C,H,W) and decoded in chunks."""

    class _Cfg:
        scaling_factor = 1.0
        shift_factor = None
        force_upcast = False

    class _VAE:
        dtype = torch.float32
        config = _Cfg()

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def decode(self, x, return_dict: bool = True):
            return (x.clone(),)

    node = SD15VAEDecodeNode("vae", vae=_VAE(), config={"decode_chunk_size": 2, "output_type": "pt"})
    lat = torch.randn(1, 4, 6, 8, 8)
    out = node.forward({"latents": lat})
    assert out["decoded_image"].shape[0] == 6


def test_latent_init_5d_encoded_strength() -> None:
    class _Sched:
        order = 1
        timesteps = torch.tensor([800, 700, 600, 500, 400])

        def add_noise(self, sample, noise, t):
            return sample * 0.5 + noise * 0.5

    enc = torch.randn(1, 4, 3, 8, 8)
    n = SD15LatentInitNode(
        "li",
        config={"strength": 0.5, "dtype": "float32", "device": "cpu"},
    )
    out = n.forward(
        {
            "init_latents": enc,
            "scheduler_state": {"scheduler": _Sched(), "init_noise_sigma": 1.0},
        },
    )
    assert tuple(out["latents"].shape) == (1, 4, 3, 8, 8)


def test_controlnet_flattens_5d_latents() -> None:
    class _CN:
        dtype = torch.float32

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def __call__(self, x, t, **kwargs):
            assert x.dim() == 4 and x.shape[0] == 6
            assert kwargs["encoder_hidden_states"].shape[0] == 6
            return ([torch.zeros(1)] * 3, torch.zeros(1))

    lat = torch.randn(1, 4, 3, 8, 8)
    pe = torch.randn(1, 77, 768)
    ne = torch.randn(1, 77, 768)
    ctrl = torch.randn(3, 3, 64, 64)
    node = ControlNetNode("cn", controlnet=_CN(), config={"guidance_scale": 7.5, "height": 64, "width": 64})
    out = node.forward(
        {
            "latents": lat,
            "timestep": torch.tensor(500),
            "prompt_embeds": pe,
            "negative_prompt_embeds": ne,
            "control_image": ctrl,
        },
    )
    assert C.PORT_DOWN_BLOCK_RESIDUALS in out


def test_sdxl_motionadapter_component_registered() -> None:
    spec = resolve_component_type("sdxl.motionadapter")
    assert spec.block_types == []


def test_free_noise_latents_shuffle() -> None:
    from yggdrasill.integrations.diffusers.common.animatediff_extras import apply_free_noise_latents

    cfg = {
        "animatediff_free_noise_noise_type": "shuffle_context",
        "animatediff_free_noise_context_length": 2,
        "animatediff_free_noise_context_stride": 2,
    }
    g = torch.Generator().manual_seed(0)
    z = apply_free_noise_latents(
        batch_size=1,
        num_channels=4,
        num_frames=6,
        height_latent=8,
        width_latent=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=g,
        config=cfg,
    )
    assert z.shape == (1, 4, 6, 8, 8)
