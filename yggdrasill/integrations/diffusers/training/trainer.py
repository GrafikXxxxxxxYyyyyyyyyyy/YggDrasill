"""Minimal trainer for SD1.5 LoRA diffusion training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers.components import load_components_from_pretrained
from yggdrasill.integrations.diffusers.model_store import ModelStore
from yggdrasill.integrations.diffusers.training.checkpointing import (
    export_sd15_lora_weights,
    load_training_state,
    save_training_state,
)
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.dataset import FolderCaptionDataset
from yggdrasill.integrations.diffusers.training.lora_targets import attach_lora_targets
from yggdrasill.integrations.diffusers.training.sd15_lora_objective import SD15LoRAObjective
from yggdrasill.integrations.diffusers.training.types import TrainResult, TrainingComponents


class SD15LoRATrainer:
    """Trainer for the first supported diffusion training recipe."""

    def __init__(self, config: TrainingConfig, *, components: Optional[TrainingComponents] = None) -> None:
        self.config = config
        self._components_override = components

    def _resolve_device(self) -> Any:
        import torch

        if self.config.device:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_training_dtype(self) -> Any:
        import torch

        if self.config.mixed_precision == "fp16":
            return torch.float16
        if self.config.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _build_dataloader(self) -> Any:
        import torch

        dataset = FolderCaptionDataset(
            self.config.data_dir,
            resolution=self.config.resolution,
            image_column=self.config.image_column,
            caption_column=self.config.caption_column,
            dataset_split=self.config.dataset_split,
            dataset_config_name=self.config.dataset_config_name,
        )

        def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            captions = [item["caption"] for item in batch]
            image_paths = [item["image_path"] for item in batch]
            return {
                "pixel_values": pixel_values,
                "caption": captions,
                "image_path": image_paths,
            }

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=_collate,
        )

    def _load_components(self, device: Any) -> TrainingComponents:
        if self._components_override is not None:
            return self._components_override

        components = load_components_from_pretrained(
            ["tokenizer", "text_encoder", "unet", "vae", "scheduler"],
            self.config.pretrained_model_name_or_path,
            family="sd15",
            store=ModelStore.default(),
            torch_dtype=self._resolve_training_dtype(),
        )
        loaded = TrainingComponents(
            tokenizer=components["tokenizer"],
            text_encoder=components["text_encoder"],
            unet=components["unet"],
            vae=components["vae"],
            scheduler=components["scheduler"],
        )
        for module in (loaded.text_encoder, loaded.unet, loaded.vae):
            if module is not None and hasattr(module, "to"):
                module.to(device)
        if loaded.vae is not None and hasattr(loaded.vae, "requires_grad_"):
            loaded.vae.requires_grad_(False)
        if loaded.vae is not None and hasattr(loaded.vae, "eval"):
            loaded.vae.eval()
        return loaded

    def _build_optimizer(self, parameters: List[Any]) -> Any:
        import torch

        return torch.optim.AdamW(parameters, lr=self.config.learning_rate)

    def _build_lr_scheduler(self, optimizer: Any) -> Any:
        import torch

        warmup_steps = max(0, int(self.config.lr_warmup_steps))

        def _schedule(step: int) -> float:
            if warmup_steps <= 0:
                return 1.0
            return min(1.0, float(step + 1) / float(warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_schedule)

    def _build_scaler(self) -> Any:
        import torch

        if self.config.mixed_precision == "fp16" and torch.cuda.is_available():
            return torch.cuda.amp.GradScaler()
        return None

    def _autocast_context(self, device: Any) -> Any:
        import contextlib
        import torch

        if self.config.mixed_precision is None or device.type != "cuda":
            return contextlib.nullcontext()

        dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _resume_if_needed(self, *, optimizer: Any, lr_scheduler: Any, scaler: Any) -> int:
        if not self.config.resume_from_checkpoint:
            return 0
        return load_training_state(
            self.config.resume_from_checkpoint,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            map_location="cpu",
        )

    def _write_training_metadata(self, result: TrainResult) -> None:
        metadata = {
            "recipe": "sd15_lora",
            "pretrained_model_name_or_path": self.config.pretrained_model_name_or_path,
            "train_text_encoder": self.config.train_text_encoder,
            "final_output_path": str(result.output_path),
            "global_step": result.global_step,
            "final_loss": result.final_loss,
            "checkpoints": [str(path) for path in result.checkpoints],
            "config": self.config.to_dict(),
        }
        metadata_path = result.output_path.with_suffix(".training.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def train(self) -> TrainResult:
        import torch

        torch.manual_seed(self.config.seed)
        device = self._resolve_device()
        dataloader = self._build_dataloader()
        components = self._load_components(device)
        targets = attach_lora_targets(
            unet=components.unet,
            text_encoder=components.text_encoder,
            config=self.config,
        )

        objective = SD15LoRAObjective(
            components=components,
            targets=targets,
            config=self.config,
            device=device,
        )
        optimizer = self._build_optimizer(targets.trainable_parameters)
        lr_scheduler = self._build_lr_scheduler(optimizer)
        scaler = self._build_scaler()
        checkpoints: List[Path] = []
        global_step = self._resume_if_needed(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )
        final_loss: Optional[float] = None

        grad_accum = self.config.gradient_accumulation_steps
        should_stop = False
        for _epoch in range(self.config.num_epochs):
            for batch in dataloader:
                with self._autocast_context(device):
                    loss = objective.compute_loss(batch) / grad_accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                step_in_accum = (global_step + 1) % grad_accum == 0
                if step_in_accum:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()

                global_step += 1
                final_loss = float(loss.detach().item() * grad_accum)

                if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                    print(f"[yggdrasill][train] step={global_step} loss={final_loss:.6f}")

                if self.config.checkpoint_every_n_steps > 0 and global_step % self.config.checkpoint_every_n_steps == 0:
                    checkpoint_dir = self.config.output_dir_path / f"checkpoint-{global_step}"
                    checkpoints.append(
                        save_training_state(
                            checkpoint_dir,
                            global_step=global_step,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            scaler=scaler,
                            config=self.config,
                        )
                    )

                if self.config.max_train_steps is not None and global_step >= self.config.max_train_steps:
                    should_stop = True
                    break
            if should_stop:
                break

        output_path = export_sd15_lora_weights(
            output_path=self.config.final_output_path,
            unet=targets.unet,
            text_encoder=targets.text_encoder,
            include_text_encoder=self.config.train_text_encoder,
            metadata={
                "recipe": "sd15_lora",
                "adapter_metadata": targets.adapter_metadata,
                "config": self.config.to_dict(),
            },
        )
        result = TrainResult(
            output_path=output_path,
            global_step=global_step,
            checkpoints=checkpoints,
            final_loss=final_loss,
        )
        self._write_training_metadata(result)
        return result
