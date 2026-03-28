"""DiffusionGraphBuilder named train recipes (sd15_lora_train, …)."""
from __future__ import annotations

from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder
from yggdrasill.integrations.diffusers.train_recipe_registry import (
    TRAIN_RECIPE_SPECS,
    is_named_train_recipe,
)


def test_sd15_lora_train_registry() -> None:
    assert is_named_train_recipe("sd15_lora_train")
    assert TRAIN_RECIPE_SPECS["sd15_lora_train"]["default_pretrained"] == "runwayml/stable-diffusion-v1-5"


def test_from_template_sd15_lora_train_placeholder() -> None:
    b = DiffusionGraphBuilder.from_template("sd15_lora_train")
    assert b._graph.metadata.get("ygg_diffusion_train_recipe") == "sd15_lora_train"
    assert b._completed is True
