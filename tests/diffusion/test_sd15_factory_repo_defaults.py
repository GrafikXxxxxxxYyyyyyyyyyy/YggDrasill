"""SD1.5 factory: task-dependent default Hub repo (inpaint vs text2img)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_components():
    from tests.diffusion.conftest import (
        FakeScheduler,
        FakeTextEncoder,
        FakeTokenizer,
        FakeUNet,
        FakeVAE,
    )

    return {
        "tokenizer": FakeTokenizer(),
        "text_encoder": FakeTextEncoder(),
        "unet": FakeUNet(),
        "vae": FakeVAE(),
        "scheduler": FakeScheduler(),
    }


@patch("yggdrasill.integrations.diffusers.factory.ModelStore")
def test_inpaint_resolves_inpainting_repo_when_repo_id_omitted(
    mock_store_cls, fake_components,
) -> None:
    store = MagicMock()
    store.load_components_by_keys.return_value = fake_components
    mock_store_cls.default.return_value = store

    from yggdrasill.integrations.diffusers.factory import build_sd15_pipeline

    build_sd15_pipeline(task="inpaint", device="cpu")

    args, _ = store.load_components_by_keys.call_args
    assert args[2] == "runwayml/stable-diffusion-inpainting"


@patch("yggdrasill.integrations.diffusers.factory.ModelStore")
def test_text2img_resolves_v15_repo_when_repo_id_omitted(
    mock_store_cls, fake_components,
) -> None:
    store = MagicMock()
    store.load_components_by_keys.return_value = fake_components
    mock_store_cls.default.return_value = store

    from yggdrasill.integrations.diffusers.factory import build_sd15_pipeline

    build_sd15_pipeline(task="text2img", device="cpu")

    args, _ = store.load_components_by_keys.call_args
    assert args[2] == "runwayml/stable-diffusion-v1-5"


@patch("yggdrasill.integrations.diffusers.factory.ModelStore")
def test_explicit_repo_id_overrides_default(mock_store_cls, fake_components) -> None:
    store = MagicMock()
    store.load_components_by_keys.return_value = fake_components
    mock_store_cls.default.return_value = store

    from yggdrasill.integrations.diffusers.factory import build_sd15_pipeline

    custom = "user/custom-sd15"
    build_sd15_pipeline(custom, task="inpaint", device="cpu")

    args, _ = store.load_components_by_keys.call_args
    assert args[2] == custom
