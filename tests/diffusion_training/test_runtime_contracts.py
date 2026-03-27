from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sd15_unet_node_train_propagates_to_underlying_module() -> None:
    import torch

    from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode

    class _TinyUNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.config = type("Cfg", (), {"in_channels": 4, "time_cond_proj_dim": None})()

        def forward(self, latents, timestep, **kwargs):
            return type("Out", (), {"sample": latents})()

    module = _TinyUNet()
    node = SD15UNetNode("unet", unet=module)
    node.train(True)

    assert module.training is True
    assert list(node.trainable_parameters())

    node.train(False)
    assert module.training is False


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sd15_prompt_encoder_train_propagates_to_underlying_module() -> None:
    import torch

    from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode

    class _TextModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.device = torch.device("cpu")
            self.text_model = type("TM", (), {"final_layer_norm": torch.nn.Identity()})()

        def forward(self, input_ids, output_hidden_states=False, **kwargs):
            embeds = self.emb(input_ids)
            if output_hidden_states:
                return type("Out", (), {"hidden_states": [embeds, embeds]})()
            return (embeds,)

    module = _TextModel()
    node = SD15PromptEncoderNode("enc", text_encoder=module)
    node.train(True)

    assert module.training is True
    assert list(node.trainable_parameters())

    node.train(False)
    assert module.training is False


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_training_vae_encode_force_upcast_runs_encoder_in_float32() -> None:
    import torch
    import torch.nn as nn

    from yggdrasill.integrations.diffusers.training.objective_utils import vae_encode_latents

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

    latents = vae_encode_latents(
        vae=vae,
        pixel_values=torch.randn(1, 3, 64, 64, dtype=torch.float16),
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert seen and seen[0] == torch.float32
    assert latents.dtype == torch.float16
    assert next(vae.parameters()).dtype == torch.float16
