from __future__ import annotations

import pytest

from yggdrasill.integrations.diffusers.training.config import TrainingConfig


def test_sdxl_config_defaults_resolution_and_sizes(tmp_path) -> None:
    config = TrainingConfig(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="sdxl",
    )
    assert config.resolution == 1024
    assert config.original_size == (1024, 1024)
    assert config.target_size == (1024, 1024)


def test_sdxl_refiner_enables_aesthetic_conditioning(tmp_path) -> None:
    config = TrainingConfig(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-refiner-1.0",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="sdxl",
        task="refiner",
    )
    assert config.requires_aesthetics_score is True


def test_sd15_rejects_text_encoder_2_training(tmp_path) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            family="sd15",
            train_text_encoder_2=True,
        )
