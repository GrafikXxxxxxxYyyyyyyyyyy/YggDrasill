"""Named LoRA train recipes for :class:`DiffusionGraphBuilder` (no training package import side effects)."""
from __future__ import annotations

from typing import Any, Dict, FrozenSet

TRAIN_RECIPE_SPECS: Dict[str, Dict[str, Any]] = {
    "sd15_lora_train": {
        "family": "sd15",
        "default_pretrained": "runwayml/stable-diffusion-v1-5",
    },
    "sdxl_lora_train": {
        "family": "sdxl",
        "default_pretrained": "stabilityai/stable-diffusion-xl-base-1.0",
    },
    "flux_lora_train": {
        "family": "flux",
        "default_pretrained": "black-forest-labs/FLUX.1-dev",
    },
}

NAMED_TRAIN_RECIPE_KEYS: FrozenSet[str] = frozenset(TRAIN_RECIPE_SPECS.keys())


def is_named_train_recipe(template_key: str) -> bool:
    """Whether *template_key* is a family LoRA recipe (e.g. ``sd15_lora_train``)."""
    return template_key in NAMED_TRAIN_RECIPE_KEYS
