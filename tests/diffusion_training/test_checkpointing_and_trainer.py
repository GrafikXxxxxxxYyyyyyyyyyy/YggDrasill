from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.checkpointing import load_training_state, save_training_state
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.trainer import SD15LoRATrainer, SDXLLoRATrainer
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

    def _fake_attach_targets(self, components):
        unet = components.unet
        text_encoder = components.text_encoder
        for parameter in unet.parameters():
            parameter.requires_grad = True
        return TrainingTargetSetup(
            unet=unet,
            text_encoder=text_encoder,
            trainable_parameters=list(unet.parameters()),
            adapter_metadata={"unet": {"adapter_name": "default"}},
        )

    def _fake_export_weights(self, *, targets):
        output_path = self.config.final_output_path
        output_path.write_text("adapter", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.DiffusionLoRATrainer._attach_targets",
        _fake_attach_targets,
    )
    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.DiffusionLoRATrainer._export_weights",
        _fake_export_weights,
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
def test_max_train_steps_counts_optimizer_updates_with_gradient_accumulation(monkeypatch, tmp_path) -> None:
    import torch

    pytest.importorskip("PIL.Image")

    from PIL import Image

    for index in range(4):
        image = Image.new("RGB", (16, 16), color="blue")
        image.save(tmp_path / f"sample_{index}.png")
        (tmp_path / f"sample_{index}.txt").write_text("blue square", encoding="utf-8")

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
            return SimpleNamespace(latent_dist=_LatentDist(pooled[:, :3]))

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

    def _fake_attach_targets(self, components):
        unet = components.unet
        text_encoder = components.text_encoder
        for parameter in unet.parameters():
            parameter.requires_grad = True
        return TrainingTargetSetup(
            unet=unet,
            text_encoder=text_encoder,
            trainable_parameters=list(unet.parameters()),
            adapter_metadata={"unet": {"adapter_name": "default"}},
        )

    def _fake_export_weights(self, *, targets):
        output_path = self.config.final_output_path
        output_path.write_text("adapter", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.DiffusionLoRATrainer._attach_targets",
        _fake_attach_targets,
    )
    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.DiffusionLoRATrainer._export_weights",
        _fake_export_weights,
    )

    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        resolution=16,
        batch_size=1,
        gradient_accumulation_steps=4,
        max_train_steps=1,
        num_epochs=1,
    )
    result = SD15LoRATrainer(config, components=components).train()

    assert result.global_step == 1


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_trainer_uses_float32_loading_without_mixed_precision(monkeypatch, tmp_path) -> None:
    import torch

    captured = {}

    def _fake_load_training_components(*, config, device, torch_dtype):
        captured["torch_dtype"] = torch_dtype
        return TrainingComponents(
            tokenizer=object(),
            text_encoder=torch.nn.Linear(1, 1),
            unet=torch.nn.Linear(1, 1),
            vae=torch.nn.Linear(1, 1),
            scheduler=object(),
        )

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.load_training_components",
        _fake_load_training_components,
    )

    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
    )
    trainer = SD15LoRATrainer(config)
    trainer._load_components(torch.device("cpu"))

    assert captured["torch_dtype"] == torch.float32


def test_training_component_loading_forces_fresh_instances(monkeypatch, tmp_path) -> None:
    captured = {}

    class _Store:
        def load_components_by_keys(self, family, load_keys, repo_id, **kwargs):
            captured["force_reload"] = kwargs.get("force_reload")
            captured["variant"] = kwargs.get("variant")
            return {}

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.family_registry.ModelStore.default",
        lambda: _Store(),
    )

    from yggdrasill.integrations.diffusers.training.family_registry import load_training_components

    load_training_components(
        config=TrainingConfig(
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
        ),
        device="cpu",
        torch_dtype=None,
    )

    assert captured["force_reload"] is True
    assert captured["variant"] == ""


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_training_component_loading_uses_fp16_variant_for_bf16_sdxl(monkeypatch, tmp_path) -> None:
    import torch

    captured = {}

    class _Store:
        def load_components_by_keys(self, family, load_keys, repo_id, **kwargs):
            captured["family"] = family
            captured["torch_dtype"] = kwargs.get("torch_dtype")
            captured["variant"] = kwargs.get("variant")
            return {}

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.family_registry.ModelStore.default",
        lambda: _Store(),
    )

    from yggdrasill.integrations.diffusers.training.family_registry import load_training_components

    load_training_components(
        config=TrainingConfig(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            family="sdxl",
        ),
        device="cpu",
        torch_dtype=torch.bfloat16,
    )

    assert captured["family"] == "sdxl"
    assert captured["torch_dtype"] == torch.bfloat16
    assert captured["variant"] == "fp16"


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sdxl_trainer_uses_safe_text_encoder_lr_for_fp16(monkeypatch, tmp_path) -> None:
    import torch

    backbone = torch.nn.Linear(2, 2)
    text_encoder = torch.nn.Linear(2, 2)

    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(True)

    trainer = SDXLLoRATrainer(
        TrainingConfig(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            family="sdxl",
            mixed_precision="fp16",
            train_text_encoder=True,
            learning_rate=1e-4,
        )
    )
    optimizer = trainer._build_optimizer(
        TrainingTargetSetup(
            backbone=backbone,
            text_encoder=text_encoder,
            trainable_parameters=[*backbone.parameters(), *text_encoder.parameters()],
            adapter_metadata={"unet": {"adapter_name": "default"}},
        )
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-5)


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sdxl_trainer_promotes_fp16_to_bf16_when_supported(monkeypatch, tmp_path) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    trainer = SDXLLoRATrainer(
        TrainingConfig(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            family="sdxl",
            mixed_precision="fp16",
            train_text_encoder=True,
            learning_rate=1e-4,
        )
    )

    assert trainer._effective_mixed_precision() == "bf16"
    assert trainer._resolve_training_dtype() == torch.bfloat16


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sdxl_trainer_honors_explicit_text_encoder_lr(monkeypatch, tmp_path) -> None:
    import torch

    backbone = torch.nn.Linear(2, 2)
    text_encoder = torch.nn.Linear(2, 2)

    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(True)

    trainer = SDXLLoRATrainer(
        TrainingConfig(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            family="sdxl",
            mixed_precision="fp16",
            train_text_encoder=True,
            learning_rate=1e-4,
            text_encoder_learning_rate=2e-5,
        )
    )
    optimizer = trainer._build_optimizer(
        TrainingTargetSetup(
            backbone=backbone,
            text_encoder=text_encoder,
            trainable_parameters=[*backbone.parameters(), *text_encoder.parameters()],
            adapter_metadata={"unet": {"adapter_name": "default"}},
        )
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(2e-5)


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sdxl_trainer_honors_explicit_backbone_lr(monkeypatch, tmp_path) -> None:
    import torch

    backbone = torch.nn.Linear(2, 2)
    text_encoder = torch.nn.Linear(2, 2)

    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(True)

    trainer = SDXLLoRATrainer(
        TrainingConfig(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            family="sdxl",
            mixed_precision="fp16",
            train_text_encoder=True,
            learning_rate=1e-4,
            backbone_learning_rate=8e-5,
        )
    )
    optimizer = trainer._build_optimizer(
        TrainingTargetSetup(
            backbone=backbone,
            text_encoder=text_encoder,
            trainable_parameters=[*backbone.parameters(), *text_encoder.parameters()],
            adapter_metadata={"unet": {"adapter_name": "default"}},
        )
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(8e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-5)
