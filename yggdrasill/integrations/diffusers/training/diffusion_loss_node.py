"""Converter node wrapping a diffusion :class:`TrainingObjective` for graph-native training."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.foundation.registry import register_block
from yggdrasill.integrations.diffusers.training.types import TrainingObjective
from yggdrasill.task_nodes.abstract import AbstractConverter


@register_block("converter/diffusion_lora_loss")
class DiffusionLoRALossConverter(AbstractConverter):
    """``converter/diffusion_lora_loss``: batch dict -> scalar loss tensor (autograd)."""

    @property
    def block_type(self) -> str:
        return "converter/diffusion_lora_loss"

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        objective: Optional[TrainingObjective] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(node_id, block_id=block_id, config=dict(config or {}))
        self._objective = objective

    def declare_ports(self) -> List[Port]:
        return [
            Port("batch", PortDirection.IN, PortType.DICT),
            Port("loss", PortDirection.OUT, PortType.TENSOR),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._objective is None:
            raise RuntimeError("DiffusionLoRALossConverter requires a bound objective")
        loss = self._objective.compute_loss(inputs["batch"])
        return {"loss": loss}

    @property
    def supports_backward(self) -> bool:
        return True
