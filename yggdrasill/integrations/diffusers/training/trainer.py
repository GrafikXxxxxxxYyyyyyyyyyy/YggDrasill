"""Family-aware trainer shell for diffusion LoRA training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers.training.checkpointing import (
    export_lora_weights,
    save_training_state,
)
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.dataset import DiffusionTrainingDataset
from yggdrasill.integrations.diffusers.training.family_registry import (
    get_training_family_spec,
    load_training_components,
    resolve_training_config,
)
from yggdrasill.integrations.diffusers.training.types import TrainResult, TrainingComponents, TrainingTargetSetup


class BaseLoRATrainer:
    """Reusable trainer shell shared by SD1.5 and SDXL recipes."""

    def __init__(self, config: TrainingConfig, *, components: Optional[TrainingComponents] = None) -> None:
        self.config = resolve_training_config(config)
        self._components_override = components

    def _effective_mixed_precision(self) -> Optional[str]:
        import torch

        requested = self.config.mixed_precision
        if requested != "fp16":
            return requested
        if self.config.family != "sdxl" or not self.config.train_text_encoder:
            return requested
        if not torch.cuda.is_available():
            return requested
        try:
            if torch.cuda.is_bf16_supported():
                return "bf16"
        except Exception:
            return requested
        return requested

    def _resolve_device(self) -> Any:
        import torch

        if self.config.device:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_training_dtype(self) -> Any:
        import torch

        effective_mixed_precision = self._effective_mixed_precision()
        if effective_mixed_precision == "fp16":
            return torch.float16
        if effective_mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _build_dataloader(self) -> Any:
        import torch

        dataset = DiffusionTrainingDataset(self.config)

        def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            collated: Dict[str, Any] = {
                "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
                "caption": [item["caption"] for item in batch],
                "prompt_2": [item.get("prompt_2", item["caption"]) for item in batch],
                "image_path": [item["image_path"] for item in batch],
            }
            if "init_pixel_values" in batch[0]:
                collated["init_pixel_values"] = torch.stack([item["init_pixel_values"] for item in batch])
                collated["init_image_path"] = [item.get("init_image_path") for item in batch]
            if "mask_values" in batch[0]:
                collated["mask_values"] = torch.stack([item["mask_values"] for item in batch])
                collated["mask_path"] = [item.get("mask_path") for item in batch]
            if "masked_pixel_values" in batch[0]:
                collated["masked_pixel_values"] = torch.stack([item["masked_pixel_values"] for item in batch])
                collated["masked_image_path"] = [item.get("masked_image_path") for item in batch]
            if "aesthetic_score" in batch[0]:
                collated["aesthetic_score"] = torch.tensor([float(item["aesthetic_score"]) for item in batch])
            return collated

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
        raise NotImplementedError

    def _attach_targets(self, components: TrainingComponents) -> TrainingTargetSetup:
        raise NotImplementedError

    def _build_objective(
        self,
        *,
        components: TrainingComponents,
        targets: TrainingTargetSetup,
        device: Any,
    ) -> Any:
        raise NotImplementedError

    def _export_weights(self, *, targets: TrainingTargetSetup) -> Path:
        raise NotImplementedError

    def _training_metadata(
        self,
        result: TrainResult,
        targets: TrainingTargetSetup,
        *,
        training_plan_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "recipe": self.config.recipe_name,
            "family": self.config.family,
            "task": self.config.task,
            "pretrained_model_name_or_path": self.config.pretrained_model_name_or_path,
            "train_text_encoder": self.config.train_text_encoder,
            "train_text_encoder_2": self.config.train_text_encoder_2,
            "adapter_metadata": targets.adapter_metadata,
            "final_output_path": str(result.output_path),
            "global_step": result.global_step,
            "final_loss": result.final_loss,
            "checkpoints": [str(path) for path in result.checkpoints],
            "config": self.config.to_dict(),
            "training_execution": "hypergraph",
        }
        if training_plan_signature is not None:
            meta["training_plan_signature"] = training_plan_signature
        return meta

    def _trainable_parameters_for_model(self, model: Any) -> List[Any]:
        if model is None or not hasattr(model, "parameters"):
            return []
        return [parameter for parameter in model.parameters() if getattr(parameter, "requires_grad", False)]

    def _resolve_text_encoder_learning_rate(self) -> float:
        if self.config.text_encoder_learning_rate is not None:
            return self.config.text_encoder_learning_rate
        if (
            self.config.family == "sdxl"
            and self.config.mixed_precision == "fp16"
            and self.config.train_text_encoder
        ):
            return min(self.config.learning_rate, 1e-5)
        return self.config.learning_rate

    def _resolve_backbone_learning_rate(self) -> float:
        if self.config.backbone_learning_rate is not None:
            return self.config.backbone_learning_rate
        if (
            self.config.family == "sdxl"
            and self.config.mixed_precision == "fp16"
            and self.config.train_text_encoder
        ):
            return min(self.config.learning_rate, 5e-5)
        return self.config.learning_rate

    def _resolve_text_encoder_2_learning_rate(self) -> float:
        if self.config.text_encoder_2_learning_rate is not None:
            return self.config.text_encoder_2_learning_rate
        if self.config.text_encoder_learning_rate is not None:
            return self.config.text_encoder_learning_rate
        if (
            self.config.family == "sdxl"
            and self.config.mixed_precision == "fp16"
            and self.config.train_text_encoder_2
        ):
            return min(self.config.learning_rate, 1e-5)
        return self.config.learning_rate

    def _build_optimizer(self, targets: TrainingTargetSetup) -> Any:
        import torch

        param_groups: List[Dict[str, Any]] = []

        backbone_parameters = self._trainable_parameters_for_model(targets.backbone)
        if backbone_parameters:
            param_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": self._resolve_backbone_learning_rate(),
                }
            )

        if self.config.train_text_encoder:
            text_encoder_parameters = self._trainable_parameters_for_model(targets.text_encoder)
            if text_encoder_parameters:
                param_groups.append(
                    {
                        "params": text_encoder_parameters,
                        "lr": self._resolve_text_encoder_learning_rate(),
                    }
                )

        if self.config.train_text_encoder_2:
            text_encoder_2_parameters = self._trainable_parameters_for_model(targets.text_encoder_2)
            if text_encoder_2_parameters:
                param_groups.append(
                    {
                        "params": text_encoder_2_parameters,
                        "lr": self._resolve_text_encoder_2_learning_rate(),
                    }
                )

        if not param_groups:
            raise RuntimeError("No trainable parameter groups were produced for the optimizer")

        return torch.optim.AdamW(param_groups)

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

        if self._effective_mixed_precision() == "fp16" and torch.cuda.is_available():
            return torch.amp.GradScaler("cuda")
        return None

    def _autocast_context(self, device: Any) -> Any:
        import contextlib
        import torch

        effective_mixed_precision = self._effective_mixed_precision()
        if effective_mixed_precision is None or device.type != "cuda":
            return contextlib.nullcontext()

        dtype = torch.float16 if effective_mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _enable_gradient_checkpointing(self, module: Any) -> None:
        if module is None:
            return
        if hasattr(module, "enable_gradient_checkpointing"):
            try:
                module.enable_gradient_checkpointing()
                return
            except Exception:
                pass
        if hasattr(module, "gradient_checkpointing_enable"):
            try:
                module.gradient_checkpointing_enable()
            except Exception:
                pass

    def _enable_attention_slicing(self, module: Any) -> None:
        if module is None:
            return
        if hasattr(module, "set_attention_slice"):
            try:
                module.set_attention_slice("auto")
            except Exception:
                pass

    def _enable_memory_efficient_training(self, *, targets: TrainingTargetSetup) -> None:
        self._enable_gradient_checkpointing(targets.backbone)
        if self.config.train_text_encoder:
            self._enable_gradient_checkpointing(targets.text_encoder)
        if self.config.train_text_encoder_2:
            self._enable_gradient_checkpointing(targets.text_encoder_2)
        self._enable_attention_slicing(targets.backbone)

    def _write_training_metadata(self, result: TrainResult) -> None:
        metadata = getattr(self, "_latest_metadata", None)
        if metadata is None:
            metadata = self._training_metadata(result, getattr(self, "_latest_targets"))
        metadata_path = result.output_path.with_suffix(".training.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def train(self) -> TrainResult:
        import torch

        import yggdrasill.training.blocks  # noqa: F401 — register training/* and outer_module/*

        from yggdrasill.engine.executor import run as engine_run
        from yggdrasill.engine.planner import training_plan_signature
        from yggdrasill.integrations.diffusers.training.training_hypergraph import (
            build_diffusion_lora_training_hypergraph,
        )
        from yggdrasill.training.context import TrainingStepContext
        from yggdrasill.training.executor import resume_training as resume_training_graph

        torch.manual_seed(self.config.seed)
        device = self._resolve_device()
        effective_mixed_precision = self._effective_mixed_precision()
        if effective_mixed_precision != self.config.mixed_precision:
            print(
                "[yggdrasill][train] promoting runtime mixed precision "
                f"from {self.config.mixed_precision} to {effective_mixed_precision} "
                "for SDXL text-encoder LoRA stability"
            )
        dataloader = self._build_dataloader()
        components = self._load_components(device)
        targets = self._attach_targets(components)
        self._enable_memory_efficient_training(targets=targets)
        self._latest_targets = targets
        objective = self._build_objective(components=components, targets=targets, device=device)
        optimizer = self._build_optimizer(targets)
        lr_scheduler = self._build_lr_scheduler(optimizer)
        scaler = self._build_scaler()

        graph = build_diffusion_lora_training_hypergraph(objective=objective)
        plan_sig = training_plan_signature(graph)
        ctx = TrainingStepContext(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            trainable_parameters=targets.trainable_parameters,
            grad_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            scaler=scaler,
            micro_step=0,
            global_step=0,
            autocast_cm=lambda: self._autocast_context(device),
        )
        if self.config.resume_from_checkpoint:
            resume_training_graph(
                graph,
                self.config.resume_from_checkpoint,
                ctx,
                map_location="cpu",
            )

        checkpoints: List[Path] = []
        global_step = ctx.global_step
        final_loss: Optional[float] = None

        pbar: Any = None
        pbar_mode: Optional[str] = None
        if self.config.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                start_gs = int(ctx.global_step)
                if self.config.max_train_steps is not None:
                    rem = max(0, int(self.config.max_train_steps) - start_gs)
                    if rem > 0:
                        pbar = tqdm(
                            total=rem,
                            desc="LoRA train",
                            unit="step",
                            dynamic_ncols=True,
                        )
                        pbar_mode = "optimizer"
                else:
                    try:
                        n_per_epoch = len(dataloader)
                    except TypeError:
                        n_per_epoch = None
                    total_batches = (
                        int(n_per_epoch) * max(1, int(self.config.num_epochs))
                        if n_per_epoch is not None
                        else None
                    )
                    pbar = tqdm(
                        total=total_batches,
                        desc="LoRA train",
                        unit="batch",
                        dynamic_ncols=True,
                    )
                    pbar_mode = "batch"
            except ImportError:
                pbar = None
                pbar_mode = None

        def _after_batch(_ctx: Any, outcome: Any) -> None:
            nonlocal global_step, final_loss
            global_step = outcome.global_step
            final_loss = outcome.loss
            if pbar is not None:
                if pbar_mode == "optimizer" and outcome.optimizer_ran:
                    pbar.update(1)
                elif pbar_mode == "batch":
                    pbar.update(1)
                pbar.set_postfix(loss=f"{outcome.loss:.4f}", step=global_step, refresh=False)
            if (
                outcome.optimizer_ran
                and self.config.logging_steps > 0
                and global_step % self.config.logging_steps == 0
            ):
                print(f"[yggdrasill][train] step={global_step} loss={final_loss:.6f}")

            if (
                outcome.optimizer_ran
                and self.config.checkpoint_every_n_steps > 0
                and global_step % self.config.checkpoint_every_n_steps == 0
            ):
                checkpoint_dir = self.config.output_dir_path / f"checkpoint-{global_step}"
                checkpoints.append(
                    save_training_state(
                        checkpoint_dir,
                        global_step=global_step,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        config=self.config,
                        training_plan_signature=plan_sig,
                    )
                )

        train_payload: Dict[str, Any] = {
            "training_step_context": ctx,
            "training_dataloader": dataloader,
            "num_epochs": self.config.num_epochs,
            "training_batch_end": _after_batch,
        }
        if self.config.max_train_steps is not None:
            train_payload["max_train_steps"] = self.config.max_train_steps

        try:
            engine_run(
                graph,
                train_payload,
                run_mode="train",
                validate_before=True,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "Non-finite" in msg:
                guidance = ""
                if (
                    self.config.family == "sdxl"
                    and self.config.mixed_precision == "fp16"
                    and self.config.train_text_encoder
                ):
                    guidance = (
                        " For SDXL, a safer starting point is "
                        "train_text_encoder=False, or mixed_precision='bf16'. "
                        "YggDrasill now auto-upgrades this runtime to bf16 when supported, "
                        "auto-reduces LR, and enables memory-efficient training for this mode, "
                        "but some datasets can still destabilize it."
                    )
                raise RuntimeError(
                    "Non-finite training loss encountered. "
                    f"family={self.config.family!r} task={self.config.task!r} "
                    f"mixed_precision={self.config.mixed_precision!r}. "
                    "Check dataset values and mixed-precision stability."
                    f"{guidance}"
                ) from exc
            raise
        finally:
            if pbar is not None:
                pbar.close()

        output_path = self._export_weights(targets=targets)
        result = TrainResult(
            output_path=output_path,
            global_step=global_step,
            checkpoints=checkpoints,
            final_loss=final_loss,
        )
        self._latest_metadata = self._training_metadata(
            result, targets, training_plan_signature=plan_sig,
        )
        self._write_training_metadata(result)
        return result


class DiffusionLoRATrainer(BaseLoRATrainer):
    """Generic trainer dispatching through the training family registry."""

    def __init__(self, config: TrainingConfig, *, components: Optional[TrainingComponents] = None) -> None:
        super().__init__(config, components=components)
        self._family_spec = get_training_family_spec(self.config.family)

    def _load_components(self, device: Any) -> TrainingComponents:
        if self._components_override is not None:
            return self._components_override
        return load_training_components(
            config=self.config,
            device=device,
            torch_dtype=self._resolve_training_dtype(),
        )

    def _attach_targets(self, components: TrainingComponents) -> TrainingTargetSetup:
        return self._family_spec.target_resolver(components, self.config)

    def _build_objective(
        self,
        *,
        components: TrainingComponents,
        targets: TrainingTargetSetup,
        device: Any,
    ) -> Any:
        objective_cls = self._family_spec.objective_factories[self.config.task]
        return objective_cls(components=components, targets=targets, config=self.config, device=device)

    def _export_weights(self, *, targets: TrainingTargetSetup) -> Path:
        return export_lora_weights(
            output_path=self.config.final_output_path,
            family=self.config.family,
            backbone=targets.backbone,
            pipeline_class_name=self._family_spec.pipeline_class_name,
            backbone_save_arg_name=self._family_spec.backbone_save_arg_name,
            text_encoder=targets.text_encoder,
            text_encoder_2=targets.text_encoder_2,
            include_text_encoder=self.config.train_text_encoder,
            include_text_encoder_2=self.config.train_text_encoder_2,
            metadata={
                "recipe": self.config.recipe_name,
                "backbone_key": targets.backbone_key,
                "adapter_metadata": targets.adapter_metadata,
                "config": self.config.to_dict(),
            },
        )


class SD15LoRATrainer(DiffusionLoRATrainer):
    """Trainer wrapper for SD1.5 LoRA recipes."""


class SDXLLoRATrainer(DiffusionLoRATrainer):
    """Trainer wrapper for SDXL LoRA recipes."""


class FluxLoRATrainer(DiffusionLoRATrainer):
    """Trainer wrapper for FLUX LoRA recipes."""
