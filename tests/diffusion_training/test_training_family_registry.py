from __future__ import annotations

import pytest

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.family_registry import (
    get_training_family_spec,
    registered_training_families,
    resolve_training_config,
)


def test_training_family_registry_exposes_sd_and_flux_specs() -> None:
    families = registered_training_families()

    assert "sd15" in families
    assert "sdxl" in families
    assert "flux" in families
    assert get_training_family_spec("flux").backbone_key == "transformer"


def test_training_family_registry_applies_sdxl_defaults(tmp_path) -> None:
    config = TrainingConfig(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="sdxl",
    )

    resolved = resolve_training_config(config)

    assert resolved.resolution == 1024
    assert resolved.original_size == (1024, 1024)
    assert resolved.target_size == (1024, 1024)


def test_flux_training_registry_rejects_unsupported_tasks(tmp_path) -> None:
    config = TrainingConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="flux",
        task="img2img",
    )

    with pytest.raises(ValueError):
        resolve_training_config(config)
