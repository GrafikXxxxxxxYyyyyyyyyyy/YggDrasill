"""Minimal diffusion training subsystem for YggDrasill."""
from __future__ import annotations

from typing import Optional

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.trainer import SD15LoRATrainer
from yggdrasill.integrations.diffusers.training.types import TrainResult, TrainingComponents, TrainingTargetSetup


def train_sd15_lora(
    *,
    data_dir: str,
    pretrained: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    resolution: int = 512,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    lr: float = 1e-4,
    num_epochs: int = 1,
    max_train_steps: Optional[int] = None,
    seed: int = 42,
    lora_rank: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    train_vae: bool = False,
    train_text_encoder: bool = False,
    mixed_precision: Optional[str] = None,
    checkpoint_every_n_steps: int = 0,
    logging_steps: int = 10,
    caption_column: str = "caption",
    image_column: str = "image",
    dataset_split: str = "train",
    dataset_config_name: Optional[str] = None,
    lr_warmup_steps: int = 0,
    num_workers: int = 0,
    device: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> TrainResult:
    """Train a minimal SD1.5 LoRA from a local or Hugging Face dataset."""
    config = TrainingConfig(
        pretrained_model_name_or_path=pretrained,
        data_dir=data_dir,
        output_path=output_path,
        output_dir=output_dir,
        resolution=resolution,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_epochs=num_epochs,
        max_train_steps=max_train_steps,
        seed=seed,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        train_text_encoder=train_text_encoder,
        train_vae=train_vae,
        mixed_precision=mixed_precision,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        logging_steps=logging_steps,
        caption_column=caption_column,
        image_column=image_column,
        dataset_split=dataset_split,
        dataset_config_name=dataset_config_name,
        lr_warmup_steps=lr_warmup_steps,
        num_workers=num_workers,
        device=device,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return SD15LoRATrainer(config).train()


def train_sd15_lora_from_folder(**kwargs) -> TrainResult:
    """Backward-compatible alias for :func:`train_sd15_lora`."""
    return train_sd15_lora(**kwargs)


__all__ = [
    "TrainingConfig",
    "TrainResult",
    "TrainingComponents",
    "TrainingTargetSetup",
    "SD15LoRATrainer",
    "train_sd15_lora",
    "train_sd15_lora_from_folder",
]
