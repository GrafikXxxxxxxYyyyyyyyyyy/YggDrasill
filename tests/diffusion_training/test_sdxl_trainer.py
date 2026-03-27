from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.trainer import SDXLLoRATrainer
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sdxl_trainer_runs_tiny_step_with_injected_components(monkeypatch, tmp_path) -> None:
    import torch

    pytest.importorskip("PIL.Image")
    from PIL import Image

    image = Image.new("RGB", (16, 16), color="blue")
    image.save(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("blue square", encoding="utf-8")

    class _Tokenizer:
        model_max_length = 8

        def __call__(self, captions, **kwargs):
            batch = len(captions)
            return SimpleNamespace(input_ids=torch.ones((batch, 8), dtype=torch.long))

    class _TextEncoder(torch.nn.Module):
        def __init__(self, hidden_size: int, pooled_size: int, with_projection: bool) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.emb = torch.nn.Embedding(16, hidden_size)
            self.with_projection = with_projection
            self.pooled = torch.nn.Parameter(torch.ones(pooled_size))

        def forward(self, input_ids, output_hidden_states=False):
            hidden = self.emb(input_ids)
            hidden_states = [hidden, hidden + 1]
            if self.with_projection:
                return SimpleNamespace(
                    hidden_states=hidden_states,
                    text_embeds=self.pooled.unsqueeze(0).expand(input_ids.shape[0], -1),
                )
            return SimpleNamespace(
                hidden_states=hidden_states,
                pooler_output=self.pooled.unsqueeze(0).expand(input_ids.shape[0], -1),
            )

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
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

        def forward(self, latents, timesteps, encoder_hidden_states=None, added_cond_kwargs=None):
            return SimpleNamespace(sample=self.conv(latents))

    class _Scheduler:
        config = SimpleNamespace(num_train_timesteps=10, prediction_type="epsilon")

        def add_noise(self, latents, noise, timesteps):
            return latents + noise

    components = TrainingComponents(
        tokenizer=_Tokenizer(),
        tokenizer_2=_Tokenizer(),
        text_encoder=_TextEncoder(hidden_size=4, pooled_size=4, with_projection=False),
        text_encoder_2=_TextEncoder(hidden_size=4, pooled_size=4, with_projection=True),
        unet=_UNet(),
        vae=_VAE(),
        scheduler=_Scheduler(),
    )

    def _fake_attach_sdxl_lora_targets(*, unet, text_encoder, text_encoder_2, config):
        for parameter in unet.parameters():
            parameter.requires_grad = True
        return TrainingTargetSetup(
            unet=unet,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            trainable_parameters=list(unet.parameters()),
            adapter_metadata={"unet": {"adapter_name": "default"}},
        )

    def _fake_export_lora_weights(**kwargs):
        output_path = kwargs["output_path"]
        output_path.write_text("adapter", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.attach_sdxl_lora_targets",
        _fake_attach_sdxl_lora_targets,
    )
    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.export_lora_weights",
        _fake_export_lora_weights,
    )

    config = TrainingConfig(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="sdxl",
        resolution=16,
        batch_size=1,
        max_train_steps=1,
        num_epochs=1,
    )
    result = SDXLLoRATrainer(config, components=components).train()

    assert result.global_step == 1
    assert result.output_path.exists()
    metadata = json.loads(result.output_path.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert metadata["recipe"] == "sdxl_text2img_lora"
