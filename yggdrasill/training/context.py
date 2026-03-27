"""Mutable runtime state for hypergraph-native training steps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class TrainingStepContext:
    """Shared training state updated across :func:`run_training_step` calls."""

    optimizer: Any
    lr_scheduler: Any
    trainable_parameters: List[Any]
    grad_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    scaler: Any = None
    micro_step: int = 0
    global_step: int = 0
    last_optimizer_ran: bool = False
    last_loss: Optional[float] = None
    autocast_cm: Optional[Callable[[], Any]] = None

    # Optional diffusion checkpoint hook (set by domain layer)
    save_checkpoint_fn: Optional[Callable[..., Any]] = None
    checkpoint_every_n_steps: int = 0
    logging_steps: int = 0


@dataclass
class TrainingStepOutcome:
    loss: float
    optimizer_ran: bool
    global_step: int
    micro_step: int
    should_stop: bool = False

