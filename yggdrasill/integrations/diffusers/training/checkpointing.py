"""Checkpoint and export helpers for diffusion training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from yggdrasill.integrations.diffusers.training.config import TrainingConfig


def save_training_state(
    checkpoint_dir: str | Path,
    *,
    global_step: int,
    optimizer: Any,
    lr_scheduler: Any,
    scaler: Any,
    config: TrainingConfig,
) -> Path:
    import torch

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": config.to_dict(),
        },
        path / "trainer_state.pt",
    )
    (path / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_training_state(
    checkpoint_dir: str | Path,
    *,
    optimizer: Any,
    lr_scheduler: Any,
    scaler: Any,
    map_location: Optional[str] = None,
) -> int:
    import torch

    state = torch.load(Path(checkpoint_dir) / "trainer_state.pt", map_location=map_location or "cpu")
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("global_step", 0))


def export_lora_weights(
    *,
    output_path: str | Path,
    family: str,
    unet: Any = None,
    backbone: Any = None,
    pipeline_class_name: Optional[str] = None,
    backbone_save_arg_name: Optional[str] = None,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    include_text_encoder: bool = False,
    include_text_encoder_2: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export LoRA weights in a format compatible with diffusers loading APIs."""
    from peft import get_peft_model_state_dict

    try:
        import diffusers
        from diffusers.utils import convert_state_dict_to_diffusers
    except ImportError as exc:
        raise ImportError(
            "Diffusers is required to export LoRA weights in an inference-compatible format."
        ) from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if pipeline_class_name is None:
        if family == "sd15":
            pipeline_class_name = "StableDiffusionPipeline"
        elif family == "sdxl":
            pipeline_class_name = "StableDiffusionXLPipeline"
        elif family == "flux":
            pipeline_class_name = "FluxPipeline"
        else:
            raise ValueError(f"Unsupported export family: {family!r}")
    pipeline_cls = getattr(diffusers, pipeline_class_name, None)
    if pipeline_cls is None:
        raise ImportError(
            f"Diffusers does not expose the expected pipeline class {pipeline_class_name!r} for family {family!r}"
        )

    if backbone_save_arg_name is None:
        backbone_save_arg_name = "transformer_lora_layers" if family == "flux" else "unet_lora_layers"
    backbone_model = backbone if backbone is not None else unet
    if backbone_model is None:
        raise ValueError("A backbone model must be provided for LoRA export")

    save_kwargs: Dict[str, Any] = {
        "save_directory": str(target.parent),
        backbone_save_arg_name: convert_state_dict_to_diffusers(get_peft_model_state_dict(backbone_model)),
        "text_encoder_lora_layers": None,
        "weight_name": target.name,
        "safe_serialization": target.suffix == ".safetensors",
    }
    if include_text_encoder and text_encoder is not None:
        save_kwargs["text_encoder_lora_layers"] = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(text_encoder)
        )
    if include_text_encoder_2 and text_encoder_2 is not None:
        save_kwargs["text_encoder_2_lora_layers"] = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(text_encoder_2)
        )

    pipeline_cls.save_lora_weights(**save_kwargs)
    if metadata is not None:
        metadata_path = target.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return target


def export_sd15_lora_weights(
    *,
    output_path: str | Path,
    unet: Any,
    text_encoder: Any = None,
    include_text_encoder: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    return export_lora_weights(
        output_path=output_path,
        family="sd15",
        unet=unet,
        text_encoder=text_encoder,
        include_text_encoder=include_text_encoder,
        metadata=metadata,
    )
