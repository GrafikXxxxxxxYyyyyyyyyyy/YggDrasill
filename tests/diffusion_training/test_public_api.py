from yggdrasill.integrations.diffusers.training import (
    train_diffusion_lora,
    train_flux_lora,
    train_sd15_lora,
    train_sdxl_lora,
)


def test_public_training_api_exists() -> None:
    assert callable(train_diffusion_lora)
    assert callable(train_sd15_lora)
    assert callable(train_sdxl_lora)
    assert callable(train_flux_lora)
