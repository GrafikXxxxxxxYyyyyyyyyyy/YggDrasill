from __future__ import annotations

import sys
from types import SimpleNamespace

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.flux_lora_targets import attach_flux_lora_targets


class _DummyModule:
    def __init__(self) -> None:
        self._params = [SimpleNamespace(requires_grad=False), SimpleNamespace(requires_grad=False)]
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
        return self

    def eval(self):
        return self


def test_attach_flux_lora_targets_supports_transformer_backbone(monkeypatch, tmp_path) -> None:
    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.target_modules = kwargs["target_modules"]

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=_FakeLoraConfig, get_peft_model=lambda model, config, adapter_name="default": model),
    )

    config = TrainingConfig(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="flux",
        train_text_encoder_2=True,
    )
    targets = attach_flux_lora_targets(
        transformer=_DummyModule(),
        text_encoder=_DummyModule(),
        text_encoder_2=_DummyModule(),
        config=config,
    )

    assert targets.backbone_key == "transformer"
    assert targets.adapter_metadata["transformer"]["adapter_name"] == "default"
    assert targets.adapter_metadata["text_encoder_2"]["adapter_name"] == "default"
