"""Types for the minimal diffusion training subsystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingComponents:
    """Loaded pretrained components needed by the SD1.5 training recipe."""

    tokenizer: Any
    text_encoder: Any
    unet: Any
    vae: Any
    scheduler: Any


@dataclass
class TrainingTargetSetup:
    """Resolved trainable targets and adapter metadata for export."""

    unet: Any
    text_encoder: Any
    trainable_parameters: List[Any] = field(default_factory=list)
    adapter_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Result returned by the public training API."""

    output_path: Path
    global_step: int
    checkpoints: List[Path] = field(default_factory=list)
    final_loss: Optional[float] = None
