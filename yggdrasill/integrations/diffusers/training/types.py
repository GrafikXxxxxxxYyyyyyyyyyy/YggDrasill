"""Types for the diffusion training subsystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class TrainingComponents:
    """Loaded pretrained components needed by a training recipe."""

    tokenizer: Any = None
    text_encoder: Any = None
    backbone: Any = None
    vae: Any = None
    scheduler: Any = None
    tokenizer_2: Any = None
    text_encoder_2: Any = None
    components: Dict[str, Any] = field(default_factory=dict)
    backbone_key: str = "unet"

    @property
    def unet(self) -> Any:
        if self.backbone_key == "unet":
            return self.backbone
        return self.components.get("unet")

    @property
    def transformer(self) -> Any:
        if self.backbone_key == "transformer":
            return self.backbone
        return self.components.get("transformer")


@dataclass
class TrainingTargetSetup:
    """Resolved trainable targets and adapter metadata for export."""

    backbone: Any
    text_encoder: Any = None
    text_encoder_2: Any = None
    trainable_parameters: List[Any] = field(default_factory=list)
    adapter_metadata: Dict[str, Any] = field(default_factory=dict)
    backbone_key: str = "unet"

    @property
    def unet(self) -> Any:
        return self.backbone if self.backbone_key == "unet" else None

    @property
    def transformer(self) -> Any:
        return self.backbone if self.backbone_key == "transformer" else None


@dataclass
class ConditioningBundle:
    """Family-specific conditioning tensors for a single training step."""

    encoder_hidden_states: Any = None
    pooled_prompt_embeds: Any = None
    added_cond_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatentRepresentation:
    """Represents clean/noisy latents and any family-specific latent forms."""

    clean_latents: Any
    noisy_latents: Any
    timesteps: Any
    noise: Any
    packed_latents: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


class TrainingObjective(Protocol):
    """Protocol implemented by family/task-specific training objectives."""

    def compute_loss(self, batch: Dict[str, Any]) -> Any:
        ...


@dataclass
class TrainResult:
    """Result returned by the public training API."""

    output_path: Path
    global_step: int
    checkpoints: List[Path] = field(default_factory=list)
    final_loss: Optional[float] = None
