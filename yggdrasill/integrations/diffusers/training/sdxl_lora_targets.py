"""Trainable LoRA target resolution for SDXL recipes."""
from __future__ import annotations

from typing import Any

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.lora_targets import (
    _DEFAULT_TEXT_ENCODER_TARGET_MODULES,
    _DEFAULT_UNET_TARGET_MODULES,
    attach_standard_lora_targets,
)
from yggdrasill.integrations.diffusers.training.types import TrainingTargetSetup


def attach_sdxl_lora_targets(
    *,
    unet: Any,
    text_encoder: Any,
    text_encoder_2: Any,
    config: TrainingConfig,
) -> TrainingTargetSetup:
    """Attach LoRA adapters for SDXL UNet and one or both text encoders."""
    return attach_standard_lora_targets(
        backbone=unet,
        backbone_key="unet",
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        config=config,
        default_backbone_target_modules=list(_DEFAULT_UNET_TARGET_MODULES),
        default_text_encoder_target_modules=list(_DEFAULT_TEXT_ENCODER_TARGET_MODULES),
    )
