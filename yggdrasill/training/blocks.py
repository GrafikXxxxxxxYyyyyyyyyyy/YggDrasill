"""Training task nodes (loss, optim step, lr schedule, checkpoint hook, dataloader)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.foundation.registry import register_block
from yggdrasill.task_nodes.abstract import AbstractConverter, AbstractHelper, AbstractOuterModule


@register_block("outer_module/training_dataloader")
class OuterModuleTrainingDataLoader(AbstractOuterModule):
    """Supplies the next batch each forward (iterates a PyTorch DataLoader)."""

    @property
    def block_type(self) -> str:
        return "outer_module/training_dataloader"

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        dataloader: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(node_id, block_id=block_id, config=dict(config or {}))
        self._dataloader = dataloader
        self._iterator: Any = None

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._dataloader is None:
            raise RuntimeError("outer_module/training_dataloader requires a bound dataloader")
        if self._iterator is None:
            self._iterator = iter(self._dataloader)
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            batch = next(self._iterator)
        return {"output": batch}


@register_block("training/mse_loss")
class TrainingMSELoss(AbstractConverter):
    """(pred, target) -> scalar MSE loss."""

    @property
    def block_type(self) -> str:
        return "training/mse_loss"

    def declare_ports(self) -> List[Port]:
        return [
            Port("pred", PortDirection.IN, PortType.TENSOR),
            Port("target", PortDirection.IN, PortType.TENSOR),
            Port("loss", PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pred = inputs["pred"]
        target = inputs["target"]
        return {"loss": F.mse_loss(pred, target, reduction="mean")}

    @property
    def supports_backward(self) -> bool:
        return True


@register_block("training/toy_linear")
class TrainingToyLinear(AbstractHelper):
    """Tiny trainable linear map for engine tests (emits pred/target)."""

    @property
    def block_type(self) -> str:
        return "training/toy_linear"

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(node_id, block_id=block_id, config=dict(config or {}))
        dim = int(self._config.get("dim", 4))
        self._lin = nn.Linear(dim, dim)

    def declare_ports(self) -> List[Port]:
        return [
            Port("x", PortDirection.IN, PortType.TENSOR),
            Port("pred", PortDirection.OUT, PortType.TENSOR),
            Port("target", PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs["x"]
        pred = self._lin(x)
        target = torch.zeros_like(pred)
        return {"pred": pred, "target": target}

    def to(self, device: Any) -> "TrainingToyLinear":
        self._lin = self._lin.to(device)
        return self

    def trainable_parameters(self):
        return self._lin.parameters()

    @property
    def supports_backward(self) -> bool:
        return True


@register_block("training/optim_step")
class TrainingOptimizerStep(AbstractHelper):
    """Gradient clip, optimizer.step, scaler update, zero_grad; sets ctx flags."""

    @property
    def block_type(self) -> str:
        return "training/optim_step"

    def declare_ports(self) -> List[Port]:
        return [
            Port("query", PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ctx = inputs.get("training_step")
        if ctx is None:
            raise KeyError("training/optim_step requires inputs['training_step'] (TrainingStepContext)")

        grad_accum = max(1, int(ctx.grad_accumulation_steps))
        step_in_accum = (int(ctx.micro_step) + 1) % grad_accum == 0
        optimizer_ran = False

        if step_in_accum:
            if ctx.max_grad_norm is not None and ctx.trainable_parameters:
                if ctx.scaler is not None:
                    ctx.scaler.unscale_(ctx.optimizer)
                torch.nn.utils.clip_grad_norm_(ctx.trainable_parameters, float(ctx.max_grad_norm))

            if ctx.scaler is not None:
                scale_before = ctx.scaler.get_scale()
                ctx.scaler.step(ctx.optimizer)
                ctx.scaler.update()
                optimizer_ran = ctx.scaler.get_scale() >= scale_before
            else:
                ctx.optimizer.step()
                optimizer_ran = True

            ctx.optimizer.zero_grad(set_to_none=True)

        ctx.last_optimizer_ran = bool(optimizer_ran)
        return {"result": {"optimizer_ran": optimizer_ran, "step_in_accum": step_in_accum}}


@register_block("training/lr_scheduler_step")
class TrainingLRSchedulerStep(AbstractHelper):
    """Calls lr_scheduler.step() when the optimizer actually ran."""

    @property
    def block_type(self) -> str:
        return "training/lr_scheduler_step"

    def declare_ports(self) -> List[Port]:
        return [
            Port("query", PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ctx = inputs.get("training_step")
        if ctx is None:
            raise KeyError("training/lr_scheduler_step requires inputs['training_step']")

        if ctx.last_optimizer_ran and ctx.lr_scheduler is not None:
            ctx.lr_scheduler.step()
        return {"result": {"scheduler_stepped": bool(ctx.last_optimizer_ran)}}


@register_block("training/checkpoint_hook")
class TrainingCheckpointHook(AbstractHelper):
    """Optional periodic checkpoint via ctx.save_checkpoint_fn."""

    @property
    def block_type(self) -> str:
        return "training/checkpoint_hook"

    def declare_ports(self) -> List[Port]:
        return [
            Port("query", PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ctx = inputs.get("training_step")
        if ctx is None:
            return {"result": {"checkpoint": None}}
        fn = ctx.save_checkpoint_fn
        if fn is None:
            return {"result": {"checkpoint": None}}
        path = fn()
        return {"result": {"checkpoint": path}}
