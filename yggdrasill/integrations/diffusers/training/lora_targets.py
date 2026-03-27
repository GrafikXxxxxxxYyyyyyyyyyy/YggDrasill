"""Trainable LoRA target resolution for diffusion training recipes."""
from __future__ import annotations

from typing import Any, Dict, List

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.types import TrainingTargetSetup

_DEFAULT_UNET_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
_DEFAULT_TEXT_ENCODER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]


def _require_peft() -> Any:
    try:
        from peft import LoraConfig
        return LoraConfig
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for LoRA training. Install the diffusion extra dependencies."
        ) from exc


def _enable_training_mode(model: Any) -> None:
    if model is None:
        return
    if hasattr(model, "train"):
        model.train()


def _freeze_module(model: Any) -> None:
    if model is None:
        return
    if hasattr(model, "requires_grad_"):
        model.requires_grad_(False)
    if hasattr(model, "eval"):
        model.eval()


def _collect_trainable_parameters(*models: Any) -> List[Any]:
    params: List[Any] = []
    for model in models:
        if model is None or not hasattr(model, "parameters"):
            continue
        for parameter in model.parameters():
            if getattr(parameter, "requires_grad", False):
                params.append(parameter)
    return params


def _cast_trainable_parameters_to_float32(*models: Any) -> None:
    try:
        import torch
    except ImportError:
        return

    for model in models:
        if model is None or not hasattr(model, "parameters"):
            continue
        for parameter in model.parameters():
            if not getattr(parameter, "requires_grad", False):
                continue
            data = getattr(parameter, "data", None)
            if data is not None and getattr(data, "dtype", None) in (torch.float16, torch.bfloat16):
                parameter.data = data.float()


def _upcast_module_to_float32(model: Any) -> None:
    try:
        import torch
    except ImportError:
        return

    if model is None or not hasattr(model, "to"):
        return

    try:
        model.to(dtype=torch.float32)
    except Exception:
        # Some wrappers may not support dtype-only upcast cleanly.
        return


def _add_adapter(model: Any, config: Any, adapter_name: str) -> Any:
    try:
        from peft import get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for LoRA training. Install the diffusion extra dependencies."
        ) from exc

    if hasattr(model, "add_adapter"):
        model.add_adapter(config, adapter_name=adapter_name)
        return model
    return get_peft_model(model, config, adapter_name=adapter_name)


def attach_standard_lora_targets(
    *,
    backbone: Any,
    backbone_key: str,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    config: TrainingConfig,
    default_backbone_target_modules: list[str],
    default_text_encoder_target_modules: list[str],
) -> TrainingTargetSetup:
    """Attach LoRA adapters for a generic backbone and optional text encoders."""
    LoraConfig = _require_peft()

    if backbone is None:
        raise ValueError("Backbone must be provided for LoRA target attachment")

    _freeze_module(text_encoder)
    _freeze_module(text_encoder_2)
    _freeze_module(backbone)

    backbone_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=config.resolved_backbone_target_modules or list(default_backbone_target_modules),
    )
    backbone = _add_adapter(backbone, backbone_cfg, adapter_name="default")
    _enable_training_mode(backbone)

    adapter_metadata: Dict[str, Any] = {
        backbone_key: {
            "adapter_name": "default",
            "target_modules": config.resolved_backbone_target_modules or list(default_backbone_target_modules),
        }
    }

    if config.train_text_encoder:
        text_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=config.text_encoder_target_modules or list(default_text_encoder_target_modules),
        )
        text_encoder = _add_adapter(text_encoder, text_cfg, adapter_name="default")
        _enable_training_mode(text_encoder)
        adapter_metadata["text_encoder"] = {
            "adapter_name": "default",
            "target_modules": config.text_encoder_target_modules or list(default_text_encoder_target_modules),
        }
        if config.mixed_precision == "fp16":
            _upcast_module_to_float32(text_encoder)

    if config.train_text_encoder_2 and text_encoder_2 is not None:
        text_encoder_2_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=config.text_encoder_2_target_modules or list(default_text_encoder_target_modules),
        )
        text_encoder_2 = _add_adapter(text_encoder_2, text_encoder_2_cfg, adapter_name="default")
        _enable_training_mode(text_encoder_2)
        adapter_metadata["text_encoder_2"] = {
            "adapter_name": "default",
            "target_modules": config.text_encoder_2_target_modules or list(default_text_encoder_target_modules),
        }
        if config.mixed_precision == "fp16":
            _upcast_module_to_float32(text_encoder_2)

    _cast_trainable_parameters_to_float32(
        backbone,
        text_encoder if config.train_text_encoder else None,
        text_encoder_2 if config.train_text_encoder_2 else None,
    )

    trainable_parameters = _collect_trainable_parameters(
        backbone,
        text_encoder if config.train_text_encoder else None,
        text_encoder_2 if config.train_text_encoder_2 else None,
    )
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters were produced by LoRA target attachment")

    return TrainingTargetSetup(
        backbone=backbone,
        backbone_key=backbone_key,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        trainable_parameters=trainable_parameters,
        adapter_metadata=adapter_metadata,
    )


def attach_lora_targets(
    *,
    unet: Any,
    text_encoder: Any,
    config: TrainingConfig,
) -> TrainingTargetSetup:
    """Attach LoRA adapters and return the resolved trainable parameter surface."""
    return attach_standard_lora_targets(
        backbone=unet,
        backbone_key="unet",
        text_encoder=text_encoder,
        config=config,
        default_backbone_target_modules=list(_DEFAULT_UNET_TARGET_MODULES),
        default_text_encoder_target_modules=list(_DEFAULT_TEXT_ENCODER_TARGET_MODULES),
    )
