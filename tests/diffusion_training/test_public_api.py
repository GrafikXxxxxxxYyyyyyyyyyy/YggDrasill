from yggdrasill.integrations.diffusers.training import train_sd15_lora


def test_public_training_api_exists() -> None:
    assert callable(train_sd15_lora)
