"""Named templates for training :class:`~yggdrasill.hypergraph.structure.Hypergraph` graphs.

Level-3 entry: built on the same primitives as :class:`DiffusionGraphBuilder` level 2
(:func:`build_training_template` delegates to :mod:`training_hypergraph`, not ad-hoc assembly).

**Named train recipes** (e.g. ``sd15_lora_train``) are registered in
:mod:`yggdrasill.integrations.diffusers.train_recipe_registry`; :meth:`DiffusionGraphBuilder.run`
builds :class:`~yggdrasill.integrations.diffusers.training.config.TrainingConfig` and runs
:class:`~yggdrasill.integrations.diffusers.training.trainer.DiffusionLoRATrainer`.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Final, Tuple

from yggdrasill.integrations.diffusers.train_recipe_registry import (
    NAMED_TRAIN_RECIPE_KEYS,
    TRAIN_RECIPE_SPECS,
)


def _build_diffusion_lora_training(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.training.training_hypergraph import (
        build_diffusion_lora_training_hypergraph,
    )

    return build_diffusion_lora_training_hypergraph(**kwargs)


_TRAINING_TEMPLATE_BUILDERS: Dict[str, Callable[..., Any]] = {
    "diffusion_lora": _build_diffusion_lora_training,
}

TRAINING_GRAPH_TEMPLATES: Final[Tuple[str, ...]] = tuple(
    sorted(_TRAINING_TEMPLATE_BUILDERS.keys())
)

TRAIN_RECIPE_TEMPLATE_NAMES: Final[Tuple[str, ...]] = tuple(
    sorted(TRAIN_RECIPE_SPECS.keys())
)

ALL_TRAIN_TEMPLATE_NAMES: Final[Tuple[str, ...]] = tuple(
    sorted(set(_TRAINING_TEMPLATE_BUILDERS.keys()) | set(NAMED_TRAIN_RECIPE_KEYS))
)


def list_training_templates() -> Tuple[str, ...]:
    """Registered training template names for :meth:`DiffusionGraphBuilder.from_template`.

    Includes small-graph templates (``diffusion_lora``) and full-recipe names (``sd15_lora_train``, …).
    """
    return ALL_TRAIN_TEMPLATE_NAMES


def build_training_template(template_name: str, **kwargs: Any) -> Any:
    """Instantiate a training graph by template name (``task=\"train\"`` on the builder)."""
    key = template_name.strip().lower().replace("-", "_")
    fn = _TRAINING_TEMPLATE_BUILDERS.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown training template {template_name!r}. "
            f"Try one of: {', '.join(ALL_TRAIN_TEMPLATE_NAMES)}"
        )
    return fn(**kwargs)
