"""Stable diffusion-training package surface (requires torch + diffusion extras)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from yggdrasill.engine.executor import run
from yggdrasill.engine.planner import build_training_plan
import yggdrasill.integrations.diffusers.training as diffusion_training
from yggdrasill.integrations.diffusers.train_recipe_registry import TRAIN_RECIPE_SPECS


def test_training_package_exports() -> None:
    assert hasattr(diffusion_training, "list_training_templates")
    assert callable(diffusion_training.list_training_templates)
    names = set(diffusion_training.list_training_templates())
    assert names == set(TRAIN_RECIPE_SPECS.keys())
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


def test_diffusion_graph_builder_from_template_train_rejected() -> None:
    from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder

    with pytest.raises(ValueError, match="no longer supports task"):
        DiffusionGraphBuilder.from_template("anything", task="train", objective=object())


def test_build_lora_training_hypergraph_still_available() -> None:
    from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder

    class _Obj:
        def compute_loss(self, batch: object) -> object:
            import torch

            b = batch  # type: ignore[assignment]
            return torch.as_tensor(b["pixel_values"], dtype=torch.float32).mean()

    graph = DiffusionGraphBuilder.build_lora_training_hypergraph(objective=_Obj())
    meta = graph.metadata.get("training") or {}
    assert meta.get("loss_node_id") == "loss"
