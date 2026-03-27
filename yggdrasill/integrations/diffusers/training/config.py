"""Configuration objects for diffusion LoRA training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class TrainingConfig:
    """Configuration for family-aware diffusion LoRA training recipes."""

    pretrained_model_name_or_path: str
    data_dir: str
    family: str = "sd15"
    task: str = "text2img"
    output_path: Optional[str] = None
    output_dir: Optional[str] = None
    resolution: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    backbone_learning_rate: Optional[float] = None
    text_encoder_learning_rate: Optional[float] = None
    text_encoder_2_learning_rate: Optional[float] = None
    num_epochs: int = 1
    max_train_steps: Optional[int] = None
    seed: int = 42
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    train_text_encoder: bool = False
    train_text_encoder_2: bool = False
    guidance: Optional[float] = None
    train_vae: bool = False
    mixed_precision: Optional[str] = None
    checkpoint_every_n_steps: int = 0
    logging_steps: int = 10
    caption_column: str = "caption"
    image_column: str = "image"
    prompt_2_column: Optional[str] = None
    init_image_column: str = "init_image"
    mask_column: str = "mask_image"
    masked_image_column: Optional[str] = "masked_image"
    aesthetic_score_column: Optional[str] = None
    dataset_split: str = "train"
    dataset_config_name: Optional[str] = None
    lr_warmup_steps: int = 0
    max_grad_norm: Optional[float] = 1.0
    num_workers: int = 0
    device: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    strength: float = 0.8
    prompt_2_fallback_to_caption: bool = True
    original_size: Optional[Tuple[int, int]] = None
    target_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    negative_original_size: Optional[Tuple[int, int]] = None
    negative_target_size: Optional[Tuple[int, int]] = None
    negative_crops_coords_top_left: Optional[Tuple[int, int]] = None
    requires_aesthetics_score: bool = False
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    pretrained_refiner_model_name_or_path: Optional[str] = None
    backbone_target_modules: Optional[list[str]] = None
    unet_target_modules: Optional[list[str]] = None
    text_encoder_target_modules: Optional[list[str]] = None
    text_encoder_2_target_modules: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("family must be provided")
        if not self.task:
            raise ValueError("task must be provided")
        if not self.pretrained_model_name_or_path:
            raise ValueError("pretrained_model_name_or_path must be provided")
        if not self.data_dir:
            raise ValueError("data_dir must be provided")
        if self.output_path is None and self.output_dir is None:
            raise ValueError("Either output_path or output_dir must be provided")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.backbone_learning_rate is not None and self.backbone_learning_rate <= 0:
            raise ValueError("backbone_learning_rate must be > 0 when provided")
        if self.text_encoder_learning_rate is not None and self.text_encoder_learning_rate <= 0:
            raise ValueError("text_encoder_learning_rate must be > 0 when provided")
        if self.text_encoder_2_learning_rate is not None and self.text_encoder_2_learning_rate <= 0:
            raise ValueError("text_encoder_2_learning_rate must be > 0 when provided")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when provided")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.lora_alpha < 1:
            raise ValueError("lora_alpha must be >= 1")
        if self.strength <= 0 or self.strength > 1:
            raise ValueError("strength must be in the interval (0, 1]")
        if self.train_vae:
            raise NotImplementedError("VAE training is out of scope for the diffusion training subsystem")
        if self.mixed_precision not in (None, "fp16", "bf16"):
            raise ValueError("mixed_precision must be one of: None, 'fp16', 'bf16'")
        if self.task in {"img2img", "inpaint"} and not self.init_image_column:
            raise ValueError("img2img and inpaint recipes require init_image_column")
        if self.task == "inpaint" and not self.mask_column:
            raise ValueError("inpaint recipes require mask_column")
        if self.task == "refiner":
            self.requires_aesthetics_score = True
        if self.family == "sd15" and self.train_text_encoder_2:
            raise ValueError("train_text_encoder_2 is not supported for SD1.5")
        try:
            from yggdrasill.integrations.diffusers.training.family_registry import resolve_training_config

            resolve_training_config(self)
        except Exception:
            # Keep base config construction available even if the training registry
            # is not fully importable in a reduced environment.
            pass

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def output_dir_path(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        assert self.output_path is not None
        return Path(self.output_path).parent

    @property
    def final_output_path(self) -> Path:
        if self.output_path is not None:
            return Path(self.output_path)
        return self.output_dir_path / "pytorch_lora_weights.safetensors"

    @property
    def recipe_name(self) -> str:
        return f"{self.family}_{self.task}_lora"

    @property
    def resolved_backbone_target_modules(self) -> Optional[list[str]]:
        return self.backbone_target_modules or self.unet_target_modules

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["final_output_path"] = str(self.final_output_path)
        data["recipe_name"] = self.recipe_name
        return data
