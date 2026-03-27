from __future__ import annotations

import sys
from types import SimpleNamespace

from yggdrasill.integrations.diffusers.training.checkpointing import export_lora_weights


def test_export_flux_lora_weights_uses_transformer_strategy(monkeypatch, tmp_path) -> None:
    called = {}

    class _FluxPipeline:
        @staticmethod
        def save_lora_weights(
            save_directory,
            transformer_lora_layers,
            text_encoder_lora_layers,
            text_encoder_2_lora_layers,
            weight_name,
            safe_serialization,
        ):
            called["save_directory"] = save_directory
            called["transformer_lora_layers"] = transformer_lora_layers
            called["text_encoder_lora_layers"] = text_encoder_lora_layers
            called["text_encoder_2_lora_layers"] = text_encoder_2_lora_layers
            called["weight_name"] = weight_name
            called["safe_serialization"] = safe_serialization
            (tmp_path / weight_name).write_text("ok", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(get_peft_model_state_dict=lambda model: {"base.weight": 1}),
    )
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(FluxPipeline=_FluxPipeline))
    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils",
        SimpleNamespace(convert_state_dict_to_diffusers=lambda state: {"converted": state}),
    )

    target = export_lora_weights(
        output_path=tmp_path / "flux.safetensors",
        family="flux",
        backbone=object(),
        backbone_save_arg_name="transformer_lora_layers",
        pipeline_class_name="FluxPipeline",
        text_encoder=object(),
        text_encoder_2=object(),
        include_text_encoder=True,
        include_text_encoder_2=True,
        metadata={"recipe": "flux_text2img_lora"},
    )

    assert target == tmp_path / "flux.safetensors"
    assert called["weight_name"] == "flux.safetensors"
    assert called["transformer_lora_layers"] == {"converted": {"base.weight": 1}}
    assert called["text_encoder_2_lora_layers"] == {"converted": {"base.weight": 1}}
    assert (tmp_path / "flux.metadata.json").exists()
