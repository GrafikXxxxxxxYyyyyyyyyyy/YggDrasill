"""Training template names for :class:`DiffusionGraphBuilder`.

High-level training entrypoints are **named recipes** (``sd15_lora_train``, ``sdxl_lora_train``,
``flux_lora_train``, …) registered in :mod:`yggdrasill.integrations.diffusers.train_recipe_registry`.
Use :meth:`~yggdrasill.integrations.diffusers.builder.DiffusionGraphBuilder.from_template` with
one of those names, then :meth:`~yggdrasill.integrations.diffusers.builder.DiffusionGraphBuilder.run`.

For a custom training :class:`~yggdrasill.hypergraph.structure.Hypergraph` (e.g. graph-native
``run(..., run_mode=\"train\")``), build the graph with
:func:`~yggdrasill.integrations.diffusers.training.training_hypergraph.build_diffusion_lora_training_hypergraph`
or :meth:`~yggdrasill.integrations.diffusers.builder.DiffusionGraphBuilder.build_lora_training_hypergraph`.
"""
from __future__ import annotations

from typing import Final, Tuple

from yggdrasill.integrations.diffusers.train_recipe_registry import TRAIN_RECIPE_SPECS

TRAIN_RECIPE_TEMPLATE_NAMES: Final[Tuple[str, ...]] = tuple(sorted(TRAIN_RECIPE_SPECS.keys()))

# Backward-compatible alias: training "templates" are the named family recipes only.
TRAINING_GRAPH_TEMPLATES: Final[Tuple[str, ...]] = TRAIN_RECIPE_TEMPLATE_NAMES

ALL_TRAIN_TEMPLATE_NAMES: Final[Tuple[str, ...]] = TRAIN_RECIPE_TEMPLATE_NAMES


def list_training_templates() -> Tuple[str, ...]:
    """Names accepted by :meth:`DiffusionGraphBuilder.from_template` for full LoRA training."""
    return ALL_TRAIN_TEMPLATE_NAMES
