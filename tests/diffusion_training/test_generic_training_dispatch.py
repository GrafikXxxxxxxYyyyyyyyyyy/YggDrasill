from __future__ import annotations

from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.family_registry import TrainingFamilySpec, register_training_family
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_generic_training_dispatch_can_run_custom_family_without_trainer_changes(monkeypatch, tmp_path) -> None:
    import torch
    from PIL import Image

    image = Image.new("RGB", (16, 16), color="green")
    image.save(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("green square", encoding="utf-8")

    class _Objective:
        def __init__(self, *, components, targets, config, device):
            self._backbone = targets.backbone

        def compute_loss(self, batch):
            return self._backbone.weight.sum() * 0 + 1.0

    def _resolve_targets(components, config):
        return TrainingTargetSetup(
            backbone=components.backbone,
            backbone_key="transformer",
            trainable_parameters=list(components.backbone.parameters()),
            adapter_metadata={"transformer": {"adapter_name": "default"}},
        )

    register_training_family(
        TrainingFamilySpec(
            family="toy_registry_family",
            load_keys=[],
            backbone_key="transformer",
            supported_tasks={"text2img"},
            objective_factories={"text2img": _Objective},
            target_resolver=_resolve_targets,
            pipeline_class_name="ToyPipeline",
            backbone_save_arg_name="transformer_lora_layers",
        )
    )

    components = TrainingComponents(
        backbone=torch.nn.Linear(1, 1),
        backbone_key="transformer",
        vae=SimpleNamespace(),
        scheduler=SimpleNamespace(),
    )

    def _fake_export_lora_weights(**kwargs):
        output_path = kwargs["output_path"]
        output_path.write_text("adapter", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.training.trainer.export_lora_weights",
        _fake_export_lora_weights,
    )

    trainer_config = TrainingConfig(
        pretrained_model_name_or_path="toy/model",
        data_dir=str(tmp_path),
        output_path=str(tmp_path / "adapter.safetensors"),
        family="toy_registry_family",
        resolution=16,
        batch_size=1,
        max_train_steps=1,
        num_epochs=1,
    )

    from yggdrasill.integrations.diffusers.training.trainer import DiffusionLoRATrainer

    result = DiffusionLoRATrainer(trainer_config, components=components).train()

    assert result.global_step == 1
    assert result.output_path.exists()
