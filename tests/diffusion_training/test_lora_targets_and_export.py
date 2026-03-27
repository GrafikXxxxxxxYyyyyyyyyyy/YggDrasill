from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.checkpointing import export_sd15_lora_weights
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.lora_targets import attach_lora_targets


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


def test_attach_lora_targets_uses_training_first_contract(monkeypatch, tmp_path) -> None:
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
    )
    targets = attach_lora_targets(unet=_DummyModule(), text_encoder=_DummyModule(), config=config)

    assert targets.adapter_metadata["unet"]["adapter_name"] == "default"
    assert targets.trainable_parameters


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_attach_lora_targets_keeps_base_model_frozen(monkeypatch, tmp_path) -> None:
    import torch

    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.target_modules = kwargs["target_modules"]

    class _ModuleWithBaseAndAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.adapter_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def add_adapter(self, config, adapter_name="default"):
            self.adapter_weight.requires_grad_(True)

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=_FakeLoraConfig, get_peft_model=lambda model, config, adapter_name="default": model),
    )

    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
    )
    model = _ModuleWithBaseAndAdapter()
    targets = attach_lora_targets(unet=model, text_encoder=_ModuleWithBaseAndAdapter(), config=config)

    assert model.base_weight.requires_grad is False
    assert model.adapter_weight.requires_grad is True
    assert targets.trainable_parameters == [model.adapter_weight]


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_attach_lora_targets_upcasts_trainable_params_to_float32(monkeypatch, tmp_path) -> None:
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
        mixed_precision="fp16",
    )
    targets = attach_lora_targets(
        unet=_TorchDummyModule(),
        text_encoder=_TorchDummyModule(),
        config=config,
    )

    assert targets.trainable_parameters[0].dtype == torch.float32


def test_export_sd15_lora_weights_delegates_to_diffusers(monkeypatch, tmp_path) -> None:
    called = {}

    class _StableDiffusionPipeline:
        @staticmethod
        def save_lora_weights(save_directory, unet_lora_layers, text_encoder_lora_layers, weight_name, safe_serialization):
            called["save_directory"] = save_directory
            called["weight_name"] = weight_name
            called["unet_lora_layers"] = unet_lora_layers
            called["text_encoder_lora_layers"] = text_encoder_lora_layers
            called["safe_serialization"] = safe_serialization
            (tmp_path / weight_name).write_text("ok", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(get_peft_model_state_dict=lambda model: {"base.weight": 1}),
    )
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(StableDiffusionPipeline=_StableDiffusionPipeline))
    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils",
        SimpleNamespace(convert_state_dict_to_diffusers=lambda state: {"converted": state}),
    )

    target = export_sd15_lora_weights(
        output_path=tmp_path / "weights.safetensors",
        unet=object(),
        metadata={"recipe": "sd15_lora"},
    )

    assert target == tmp_path / "weights.safetensors"
    assert called["weight_name"] == "weights.safetensors"
    assert called["unet_lora_layers"] == {"converted": {"base.weight": 1}}
    assert (tmp_path / "weights.metadata.json").exists()
