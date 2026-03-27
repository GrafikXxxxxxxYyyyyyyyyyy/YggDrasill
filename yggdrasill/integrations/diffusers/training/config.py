"""Configuration objects for minimal SD1.5 LoRA training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    """Configuration for the first supported diffusion training recipe."""

    pretrained_model_name_or_path: str
    data_dir: str
    output_path: Optional[str] = None
    output_dir: Optional[str] = None
    resolution: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_train_steps: Optional[int] = None
    seed: int = 42
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    train_text_encoder: bool = False
    train_vae: bool = False
    mixed_precision: Optional[str] = None
    checkpoint_every_n_steps: int = 0
    logging_steps: int = 10
    caption_column: str = "caption"
    image_column: str = "image"
    dataset_split: str = "train"
    dataset_config_name: Optional[str] = None
    lr_warmup_steps: int = 0
    num_workers: int = 0
    device: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    unet_target_modules: Optional[list[str]] = None
    text_encoder_target_modules: Optional[list[str]] = None

    def __post_init__(self) -> None:
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
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.lora_alpha < 1:
            raise ValueError("lora_alpha must be >= 1")
        if self.train_vae:
            raise NotImplementedError("VAE training is out of scope for the minimal diffusion training subsystem")
        if self.mixed_precision not in (None, "fp16", "bf16"):
            raise ValueError("mixed_precision must be one of: None, 'fp16', 'bf16'")

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

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["final_output_path"] = str(self.final_output_path)
        return data
