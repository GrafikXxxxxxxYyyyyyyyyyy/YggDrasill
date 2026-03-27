from __future__ import annotations

from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.sdxl_inpaint_objective import SDXLInpaintLoRAObjective
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sdxl_inpaint_objective_builds_9ch_unet_input(tmp_path) -> None:
    import torch

    class _Tokenizer:
        model_max_length = 8

        def __call__(self, captions, **kwargs):
            batch = len(captions)
            return SimpleNamespace(input_ids=torch.ones((batch, 8), dtype=torch.long))

    class _TextEncoder(torch.nn.Module):
        def __init__(self, with_projection: bool) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.emb = torch.nn.Embedding(16, 4)
            self.pooled = torch.nn.Parameter(torch.ones(4))
            self.with_projection = with_projection

        def forward(self, input_ids, output_hidden_states=False):
            hidden = self.emb(input_ids)
            hidden_states = [hidden, hidden + 1]
            if self.with_projection:
                return SimpleNamespace(hidden_states=hidden_states, text_embeds=self.pooled.unsqueeze(0).expand(input_ids.shape[0], -1))
            return SimpleNamespace(hidden_states=hidden_states, pooler_output=self.pooled.unsqueeze(0).expand(input_ids.shape[0], -1))

    class _LatentDist:
        def __init__(self, sample):
            self._sample = sample

        def sample(self):
            return self._sample

    class _VAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

        def encode(self, x):
            pooled = torch.nn.functional.avg_pool2d(x, kernel_size=8)
            return SimpleNamespace(latent_dist=_LatentDist(pooled[:, :4]))

    class _UNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(9, 3, kernel_size=1)
            self.config = SimpleNamespace(in_channels=9)
            self.last_latents = None

        def forward(self, latents, timesteps, encoder_hidden_states=None, added_cond_kwargs=None):
            self.last_latents = latents
            return SimpleNamespace(sample=self.conv(latents))

    class _Scheduler:
        config = SimpleNamespace(num_train_timesteps=10, prediction_type="epsilon")

        def add_noise(self, latents, noise, timesteps):
            return latents + noise

    components = TrainingComponents(
        tokenizer=_Tokenizer(),
        tokenizer_2=_Tokenizer(),
        text_encoder=_TextEncoder(with_projection=False),
        text_encoder_2=_TextEncoder(with_projection=True),
        unet=_UNet(),
        vae=_VAE(),
        scheduler=_Scheduler(),
    )
    targets = TrainingTargetSetup(
        unet=components.unet,
        text_encoder=components.text_encoder,
        text_encoder_2=components.text_encoder_2,
        trainable_parameters=list(components.unet.parameters()),
    )
    config = TrainingConfig(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="sdxl",
        task="inpaint",
        resolution=16,
    )
    objective = SDXLInpaintLoRAObjective(
        components=components,
        targets=targets,
        config=config,
        device=torch.device("cpu"),
    )

    objective.compute_loss(
        {
            "pixel_values": torch.randn(1, 3, 16, 16),
            "caption": ["castle"],
            "prompt_2": ["detailed castle"],
            "init_pixel_values": torch.randn(1, 3, 16, 16),
            "mask_values": torch.ones(1, 1, 16, 16),
        }
    )

    assert components.unet.last_latents is not None
    assert components.unet.last_latents.shape[1] == 9
