from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.lora_targets import attach_lora_targets


class _DummyModule:
    def __init__(self) -> None:
        self._params = [SimpleNamespace(requires_grad=False), SimpleNamespace(requires_grad=False)]
        self.training = False

    def parameters(self):
        return iter(self._params)

    def add_adapter(self, config, adapter_name="default"):
        for param in self._params:
            param.requires_grad = True

    def requires_grad_(self, flag: bool):
        for param in self._params:
            param.requires_grad = flag
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)


def test_sd15_attach_lora_targets_supports_train_text_encoder(monkeypatch, tmp_path) -> None:
    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.target_modules = kwargs["target_modules"]

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=_FakeLoraConfig, get_peft_model=lambda model, config, adapter_name="default": model),
    )
    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="sd15",
        train_text_encoder=True,
    )
    targets = attach_lora_targets(unet=_DummyModule(), text_encoder=_DummyModule(), config=config)

    assert "text_encoder" in targets.adapter_metadata
    assert len(targets.trainable_parameters) == 4


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_sd15_train_text_encoder_fp16_upcasts_encoder_module(monkeypatch, tmp_path) -> None:
    import torch

    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.target_modules = kwargs["target_modules"]

    class _TorchDummyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16), requires_grad=False)

        def add_adapter(self, config, adapter_name="default"):
            self.weight.requires_grad_(True)

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=_FakeLoraConfig, get_peft_model=lambda model, config, adapter_name="default": model),
    )
    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="sd15",
        train_text_encoder=True,
        mixed_precision="fp16",
    )
    text_encoder = _TorchDummyModule()
    attach_lora_targets(unet=_TorchDummyModule(), text_encoder=text_encoder, config=config)

    assert text_encoder.weight.dtype == torch.float32
