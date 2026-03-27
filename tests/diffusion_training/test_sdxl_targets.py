from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.checkpointing import export_lora_weights
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.sdxl_lora_targets import attach_sdxl_lora_targets


class _DummyModule:
    def __init__(self) -> None:
        self._params = [SimpleNamespace(requires_grad=False), SimpleNamespace(requires_grad=False)]
        self.training = False
        self.adapters = []

    def parameters(self):
        return iter(self._params)

    def add_adapter(self, config, adapter_name="default"):
        self.adapters.append((adapter_name, config.target_modules))
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


def test_attach_sdxl_lora_targets_supports_dual_text_encoders(monkeypatch, tmp_path) -> None:
    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.target_modules = kwargs["target_modules"]

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=_FakeLoraConfig, get_peft_model=lambda model, config, adapter_name="default": model),
    )
    config = TrainingConfig(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="sdxl",
        train_text_encoder=True,
        train_text_encoder_2=True,
    )
    targets = attach_sdxl_lora_targets(
        unet=_DummyModule(),
        text_encoder=_DummyModule(),
        text_encoder_2=_DummyModule(),
        config=config,
    )

    assert targets.adapter_metadata["unet"]["adapter_name"] == "default"
    assert targets.adapter_metadata["text_encoder"]["adapter_name"] == "default"
    assert targets.adapter_metadata["text_encoder_2"]["adapter_name"] == "default"
    assert len(targets.trainable_parameters) == 6


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_attach_sdxl_lora_targets_upcasts_trainable_text_encoders_in_fp16(monkeypatch, tmp_path) -> None:
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
    text_encoder = _TorchDummyModule()
    text_encoder_2 = _TorchDummyModule()
    targets = attach_sdxl_lora_targets(
        unet=_TorchDummyModule(),
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        config=TrainingConfig(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            family="sdxl",
            train_text_encoder=True,
            train_text_encoder_2=True,
            mixed_precision="fp16",
        ),
    )

    assert targets.adapter_metadata["text_encoder"]["adapter_name"] == "default"
    assert text_encoder.weight.dtype == torch.float32
    assert text_encoder_2.weight.dtype == torch.float32


def test_export_sdxl_lora_weights_includes_second_text_encoder(monkeypatch, tmp_path) -> None:
    called = {}

    class _StableDiffusionXLPipeline:
        @staticmethod
        def save_lora_weights(**kwargs):
            called.update(kwargs)
            (tmp_path / kwargs["weight_name"]).write_text("ok", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(get_peft_model_state_dict=lambda model: {"base.weight": 1}),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(StableDiffusionXLPipeline=_StableDiffusionXLPipeline),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils",
        SimpleNamespace(convert_state_dict_to_diffusers=lambda state: {"converted": state}),
    )

    target = export_lora_weights(
        output_path=tmp_path / "sdxl.safetensors",
        family="sdxl",
        unet=object(),
        text_encoder=object(),
        text_encoder_2=object(),
        include_text_encoder=True,
        include_text_encoder_2=True,
    )

    assert target == tmp_path / "sdxl.safetensors"
    assert called["text_encoder_lora_layers"] == {"converted": {"base.weight": 1}}
    assert called["text_encoder_2_lora_layers"] == {"converted": {"base.weight": 1}}
