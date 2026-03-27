"""Engine package: inference executor, planner, validator.

Training helpers (:func:`run_training_step`, etc.) are **lazy-loaded** so that
``import yggdrasill.engine`` does not require PyTorch; they load on first access.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from yggdrasill.engine.buffers import EdgeBuffers
from yggdrasill.engine.edge import Edge
from yggdrasill.hypergraph.structure import Hypergraph
from yggdrasill.engine.validator import ValidationResult, validate
from yggdrasill.engine.planner import build_plan, clear_plan_cache
from yggdrasill.engine.executor import RunResult, ValidationError, run, run_stream

_TRAINING_LAZY: Dict[str, Tuple[str, str]] = {
    "TrainingPlan": ("yggdrasill.training.plan", "TrainingPlan"),
    "TrainingStepContext": ("yggdrasill.training.context", "TrainingStepContext"),
    "TrainingStepOutcome": ("yggdrasill.training.context", "TrainingStepOutcome"),
    "build_training_plan": ("yggdrasill.training.plan", "build_training_plan"),
    "fit_training_graph": ("yggdrasill.training.executor", "fit_training_graph"),
    "resume_training": ("yggdrasill.training.executor", "resume_training"),
    "run_training_step": ("yggdrasill.training.executor", "run_training_step"),
    "training_plan_signature": ("yggdrasill.training.plan", "training_plan_signature"),
}


def __getattr__(name: str) -> Any:
    if name in _TRAINING_LAZY:
        import yggdrasill.training.blocks  # noqa: F401 — register block types
        mod_path, attr = _TRAINING_LAZY[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    "Edge",
    "EdgeBuffers",
    "Hypergraph",
    "RunResult",
    "TrainingPlan",
    "TrainingStepContext",
    "TrainingStepOutcome",
    "ValidationError",
    "ValidationResult",
    "build_plan",
    "build_training_plan",
    "clear_plan_cache",
    "fit_training_graph",
    "resume_training",
    "run",
    "run_stream",
    "run_training_step",
    "training_plan_signature",
    "validate",
]
