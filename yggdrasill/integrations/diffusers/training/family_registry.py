"""Training-side family registry for diffusion LoRA recipes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from yggdrasill.integrations.diffusers.components import load_components_from_pretrained
from yggdrasill.integrations.diffusers.family_registry import get_family_spec as get_inference_family_spec
from yggdrasill.integrations.diffusers.model_store import ModelStore
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.flux_lora_targets import attach_flux_lora_targets
from yggdrasill.integrations.diffusers.training.lora_targets import (
    _DEFAULT_TEXT_ENCODER_TARGET_MODULES,
    _DEFAULT_UNET_TARGET_MODULES,
    attach_lora_targets,
)
from yggdrasill.integrations.diffusers.training.sdxl_lora_targets import attach_sdxl_lora_targets
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


ObjectiveFactory = Callable[..., Any]
TargetResolver = Callable[[TrainingComponents, TrainingConfig], TrainingTargetSetup]
ConfigValidator = Callable[[TrainingConfig], None]


@dataclass(frozen=True)
class TrainingFamilySpec:
    """Training behavior owned by a diffusion family."""

    family: str
    load_keys: list[str]
    backbone_key: str
    supported_tasks: set[str]
    objective_factories: Dict[str, ObjectiveFactory]
    target_resolver: TargetResolver
    pipeline_class_name: str
    backbone_save_arg_name: str
    default_resolution: int = 512
    default_backbone_target_modules: list[str] = field(default_factory=list)
    default_text_encoder_target_modules: list[str] = field(default_factory=list)
    validate_config: Optional[ConfigValidator] = None


_TRAINING_FAMILY_SPECS: Dict[str, TrainingFamilySpec] = {}


def register_training_family(spec: TrainingFamilySpec) -> None:
    _TRAINING_FAMILY_SPECS[spec.family] = spec


def get_training_family_spec(family: str) -> TrainingFamilySpec:
    spec = _TRAINING_FAMILY_SPECS.get(family)
    if spec is None:
        raise KeyError(
            f"Unknown training family {family!r}. Registered: {sorted(_TRAINING_FAMILY_SPECS.keys())}"
        )
    return spec


def registered_training_families() -> list[str]:
    return sorted(_TRAINING_FAMILY_SPECS.keys())


def _validate_sd15_config(config: TrainingConfig) -> None:
    if config.task == "refiner":
        raise ValueError("SD1.5 does not support the 'refiner' task")
    if config.train_text_encoder_2:
        raise ValueError("train_text_encoder_2 is not supported for SD1.5")


def _validate_sdxl_config(config: TrainingConfig) -> None:
    if config.resolution == 512:
        config.resolution = 1024
    config.original_size = config.original_size or (config.resolution, config.resolution)
    config.target_size = config.target_size or (config.resolution, config.resolution)
    config.negative_original_size = config.negative_original_size or config.original_size
    config.negative_target_size = config.negative_target_size or config.target_size
    config.negative_crops_coords_top_left = (
        config.negative_crops_coords_top_left or config.crops_coords_top_left
    )


def _validate_flux_config(config: TrainingConfig) -> None:
    if config.task != "text2img":
        raise ValueError("FLUX proof-point training currently supports only 'text2img'")
    if config.resolution == 512:
        config.resolution = 1024


def _resolve_training_load_variant(*, config: TrainingConfig, torch_dtype: Any) -> str:
    try:
        import torch
    except ImportError:
        return ""

    if config.family not in {"sd15", "sdxl"}:
        return ""
    if torch_dtype == torch.bfloat16:
        # SD1.5/SDXL repos commonly publish efficient fp16 weight variants.
        # We can still run the runtime in bf16 while loading the faster/smaller
        # fp16 checkpoint files.
        return "fp16"
    return ""


def load_training_components(*, config: TrainingConfig, device: Any, torch_dtype: Any) -> TrainingComponents:
    spec = get_training_family_spec(config.family)
    components = load_components_from_pretrained(
        spec.load_keys,
        config.pretrained_model_name_or_path,
        family=config.family,
        store=ModelStore.default(),
        torch_dtype=torch_dtype,
        variant=_resolve_training_load_variant(config=config, torch_dtype=torch_dtype),
        force_reload=True,
    )
    loaded = TrainingComponents(
        tokenizer=components.get("tokenizer"),
        tokenizer_2=components.get("tokenizer_2"),
        text_encoder=components.get("text_encoder"),
        text_encoder_2=components.get("text_encoder_2"),
        backbone=components.get(spec.backbone_key),
        backbone_key=spec.backbone_key,
        vae=components.get("vae"),
        scheduler=components.get("scheduler"),
        components=dict(components),
    )
    for module in (loaded.text_encoder, loaded.text_encoder_2, loaded.backbone, loaded.vae):
        if module is not None and hasattr(module, "to"):
            module.to(device)
    if loaded.vae is not None and hasattr(loaded.vae, "requires_grad_"):
        loaded.vae.requires_grad_(False)
    if loaded.vae is not None and hasattr(loaded.vae, "eval"):
        loaded.vae.eval()
    return loaded


def resolve_training_config(config: TrainingConfig) -> TrainingConfig:
    spec = get_training_family_spec(config.family)
    if config.task not in spec.supported_tasks:
        raise ValueError(
            f"Family {config.family!r} does not support task {config.task!r}. "
            f"Supported: {sorted(spec.supported_tasks)}"
        )
    if spec.validate_config is not None:
        spec.validate_config(config)
    return config


def _attach_flux_targets(components: TrainingComponents, config: TrainingConfig) -> TrainingTargetSetup:
    return attach_flux_lora_targets(
        transformer=components.transformer,
        text_encoder=components.text_encoder,
        text_encoder_2=components.text_encoder_2,
        config=config,
    )


def _attach_sd15_targets(components: TrainingComponents, config: TrainingConfig) -> TrainingTargetSetup:
    return attach_lora_targets(unet=components.unet, text_encoder=components.text_encoder, config=config)


def _attach_sdxl_targets(components: TrainingComponents, config: TrainingConfig) -> TrainingTargetSetup:
    return attach_sdxl_lora_targets(
        unet=components.unet,
        text_encoder=components.text_encoder,
        text_encoder_2=components.text_encoder_2,
        config=config,
    )


def _bootstrap_training_families() -> None:
    if _TRAINING_FAMILY_SPECS:
        return

    from yggdrasill.integrations.diffusers.training.flux_text2img_objective import FluxText2ImgLoRAObjective
    from yggdrasill.integrations.diffusers.training.sd15_img2img_objective import SD15Img2ImgLoRAObjective
    from yggdrasill.integrations.diffusers.training.sd15_inpaint_objective import SD15InpaintLoRAObjective
    from yggdrasill.integrations.diffusers.training.sd15_lora_objective import SD15LoRAObjective
    from yggdrasill.integrations.diffusers.training.sdxl_img2img_objective import SDXLImg2ImgLoRAObjective
    from yggdrasill.integrations.diffusers.training.sdxl_inpaint_objective import SDXLInpaintLoRAObjective
    from yggdrasill.integrations.diffusers.training.sdxl_refiner_objective import SDXLRefinerLoRAObjective
    from yggdrasill.integrations.diffusers.training.sdxl_text2img_objective import SDXLText2ImgLoRAObjective

    register_training_family(
        TrainingFamilySpec(
            family="sd15",
            load_keys=["tokenizer", "text_encoder", "unet", "vae", "scheduler"],
            backbone_key="unet",
            supported_tasks={"text2img", "img2img", "inpaint"},
            objective_factories={
                "text2img": SD15LoRAObjective,
                "img2img": SD15Img2ImgLoRAObjective,
                "inpaint": SD15InpaintLoRAObjective,
            },
            target_resolver=_attach_sd15_targets,
            pipeline_class_name="StableDiffusionPipeline",
            backbone_save_arg_name="unet_lora_layers",
            default_resolution=512,
            default_backbone_target_modules=list(_DEFAULT_UNET_TARGET_MODULES),
            default_text_encoder_target_modules=list(_DEFAULT_TEXT_ENCODER_TARGET_MODULES),
            validate_config=_validate_sd15_config,
        )
    )
    register_training_family(
        TrainingFamilySpec(
            family="sdxl",
            load_keys=["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "unet", "vae", "scheduler"],
            backbone_key="unet",
            supported_tasks={"text2img", "img2img", "inpaint", "refiner"},
            objective_factories={
                "text2img": SDXLText2ImgLoRAObjective,
                "img2img": SDXLImg2ImgLoRAObjective,
                "inpaint": SDXLInpaintLoRAObjective,
                "refiner": SDXLRefinerLoRAObjective,
            },
            target_resolver=_attach_sdxl_targets,
            pipeline_class_name="StableDiffusionXLPipeline",
            backbone_save_arg_name="unet_lora_layers",
            default_resolution=1024,
            default_backbone_target_modules=list(_DEFAULT_UNET_TARGET_MODULES),
            default_text_encoder_target_modules=list(_DEFAULT_TEXT_ENCODER_TARGET_MODULES),
            validate_config=_validate_sdxl_config,
        )
    )
    register_training_family(
        TrainingFamilySpec(
            family="flux",
            load_keys=["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "transformer", "vae", "scheduler"],
            backbone_key="transformer",
            supported_tasks={"text2img"},
            objective_factories={"text2img": FluxText2ImgLoRAObjective},
            target_resolver=_attach_flux_targets,
            pipeline_class_name="FluxPipeline",
            backbone_save_arg_name="transformer_lora_layers",
            default_resolution=1024,
            default_backbone_target_modules=list(_DEFAULT_UNET_TARGET_MODULES),
            default_text_encoder_target_modules=list(_DEFAULT_TEXT_ENCODER_TARGET_MODULES),
            validate_config=_validate_flux_config,
        )
    )


_bootstrap_training_families()
