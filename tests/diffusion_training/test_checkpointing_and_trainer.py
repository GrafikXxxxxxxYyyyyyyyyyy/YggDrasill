from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.checkpointing import load_training_state, save_training_state
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.trainer import SD15LoRATrainer
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_save_and_load_training_state_roundtrip(tmp_path) -> None:
    import torch

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
    )

    checkpoint_dir = save_training_state(
        tmp_path / "checkpoint-1",
        global_step=3,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scaler=None,
        config=config,
    )

    optimizer_2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler_2 = torch.optim.lr_scheduler.LambdaLR(optimizer_2, lr_lambda=lambda step: 1.0)
    step = load_training_state(
        checkpoint_dir,
        optimizer=optimizer_2,
        lr_scheduler=scheduler_2,
        scaler=None,
    )

    assert step == 3


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_trainer_runs_tiny_step_with_injected_components(monkeypatch, tmp_path) -> None:
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
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(16, 4)

        def forward(self, input_ids, **kwargs):
            return (self.emb(input_ids),)

    class _LatentDist:
        def __init__(self, sample):
            self._sample = sample

        def sample(self):
            return self._sample

    class _VAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(scaling_factor=1.0)

        def encode(self, x):
            pooled = torch.nn.functional.avg_pool2d(x, kernel_size=8)
            return SimpleNamespace(latent_dist=_LatentDist(pooled[:, :4]))

    class _UNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

        def forward(self, latents, timesteps, encoder_hidden_states=None):
            return SimpleNamespace(sample=self.conv(latents))

    class _Scheduler:
        config = SimpleNamespace(num_train_timesteps=10, prediction_type="epsilon")

        def add_noise(self, latents, noise, timesteps):
            return latents + noise

    components = TrainingComponents(
        tokenizer=_Tokenizer(),
        text_encoder=_TextEncoder(),
        unet=_UNet(),
        vae=_VAE(),
        scheduler=_Scheduler(),
    )

    def _fake_attach_lora_targets(*, unet, text_encoder, config):
        for parameter in unet.parameters():
            parameter.requires_grad = True
        return TrainingTargetSetup(
            unet=unet,
            text_encoder=text_encoder,
            trainable_parameters=list(unet.parameters()),
            adapter_metadata={"unet": {"adapter_name": "default"}},
        )

    def _fake_export_sd15_lora_weights(**kwargs):
        output_path = kwargs["output_path"]
        output_path.write_text("adapter", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.attach_lora_targets",
        _fake_attach_lora_targets,
    )
    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.export_sd15_lora_weights",
        _fake_export_sd15_lora_weights,
    )

    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        resolution=16,
        batch_size=1,
        max_train_steps=1,
        num_epochs=1,
    )
    result = SD15LoRATrainer(config, components=components).train()

    assert result.global_step == 1
    assert result.output_path.exists()
    metadata = json.loads(result.output_path.with_suffix(".training.json").read_text(encoding="utf-8"))
    assert metadata["recipe"] == "sd15_lora"


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_trainer_uses_float32_loading_without_mixed_precision(monkeypatch, tmp_path) -> None:
    import torch

    captured = {}

    def _fake_load_components(load_keys, pretrained, **kwargs):
        captured["torch_dtype"] = kwargs.get("torch_dtype")
        return {
            "tokenizer": object(),
            "text_encoder": torch.nn.Linear(1, 1),
            "unet": torch.nn.Linear(1, 1),
            "vae": torch.nn.Linear(1, 1),
            "scheduler": object(),
        }

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.load_components_from_pretrained",
        _fake_load_components,
    )

    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
    )
    trainer = SD15LoRATrainer(config)
    trainer._load_components(torch.device("cpu"))

    assert captured["torch_dtype"] == torch.float32
