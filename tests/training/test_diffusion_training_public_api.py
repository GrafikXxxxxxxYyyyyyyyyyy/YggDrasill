"""Stable diffusion-training package surface (requires torch + diffusion extras)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from yggdrasill.engine.executor import run
from yggdrasill.engine.planner import build_training_plan
import yggdrasill.integrations.diffusers.training as diffusion_training


def test_training_package_exports() -> None:
    assert hasattr(diffusion_training, "list_training_templates")
    assert callable(diffusion_training.list_training_templates)
    names = diffusion_training.list_training_templates()
    assert "diffusion_lora" in names
    assert "sd15_lora_train" in names
    assert hasattr(diffusion_training, "TrainingConfig")
    assert hasattr(diffusion_training, "build_diffusion_lora_training_hypergraph")


def test_engine_training_planner_co_located() -> None:
    assert callable(build_training_plan)
    assert callable(run)


def test_legacy_entrypoints_still_importable() -> None:
    """Convenience trainers are no longer in ``__all__`` but remain on the module."""
    assert hasattr(diffusion_training, "train_diffusion_lora")
    assert hasattr(diffusion_training, "train_sd15_lora")
    assert callable(diffusion_training.train_diffusion_lora)
    assert callable(diffusion_training.train_sd15_lora)


def test_diffusion_graph_builder_from_template_train() -> None:
    from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder

    class _Obj:
        def compute_loss(self, batch: object) -> object:
            import torch

            b = batch  # type: ignore[assignment]
            return torch.as_tensor(b["pixel_values"], dtype=torch.float32).mean()

    builder = DiffusionGraphBuilder.from_template(
        "diffusion_lora",
        task="train",
        objective=_Obj(),
    )
    meta = builder.graph.metadata.get("training") or {}
    assert meta.get("loss_node_id") == "loss"
