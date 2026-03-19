"""SDXL added conditioning node: time_ids + text_embeds for UNet."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractHelper


def _zeros_like_prompt_tensor(pooled: Any) -> Any:
    """Match ``torch.zeros_like(pooled)`` but support tensor-like test stubs (e.g. FakeTensor)."""
    import torch

    if isinstance(pooled, torch.Tensor):
        return torch.zeros_like(pooled)
    shape = tuple(pooled.shape) if hasattr(pooled, "shape") else (1,)
    device = getattr(pooled, "device", "cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device) if isinstance(device, str) else torch.device("cpu")
    dtype = getattr(pooled, "dtype", torch.float32)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, torch.float32)
    return torch.zeros(shape, device=device, dtype=dtype)


def _device_dtype_like(ref: Any) -> Tuple[Any, Any]:
    """``(device, dtype)`` suitable for ``Tensor.to(...)``, including tensor-like test stubs."""
    import torch

    if isinstance(ref, torch.Tensor):
        return ref.device, ref.dtype
    dev = getattr(ref, "device", "cpu")
    if not isinstance(dev, torch.device):
        dev = torch.device(dev) if isinstance(dev, str) else torch.device("cpu")
    dtype = getattr(ref, "dtype", torch.float32)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, torch.float32)
    return dev, dtype


class SDXLAddedConditioningNode(AbstractHelper):
    """Builds add_text_embeds and add_time_ids for SDXL UNet.

    Supports both size/crop conditioning (base) and aesthetic score
    conditioning (refiner) modes.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)

    @property
    def block_type(self) -> str:
        return "sdxl/added_conditioning"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_POOLED_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_ADD_TEXT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_ADD_TIME_IDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_ADD_TIME_IDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def _build_time_ids(
        self,
        original_size: Tuple[int, int],
        crops_coords_top_left: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Any:
        import torch
        return torch.tensor(list(original_size) + list(crops_coords_top_left) + list(target_size))

    def _build_aesthetic_time_ids(
        self,
        original_size: Tuple[int, int],
        crops_coords_top_left: Tuple[int, int],
        aesthetic_score: float,
    ) -> Any:
        import torch
        return torch.tensor(list(original_size) + list(crops_coords_top_left) + [aesthetic_score])

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        pooled = inputs[C.PORT_POOLED_PROMPT_EMBEDS]
        neg_pooled = inputs.get(C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS)
        if neg_pooled is None:
            neg_pooled = _zeros_like_prompt_tensor(pooled)

        requires_aesthetics = self._config.get("requires_aesthetics_score", False)
        original_size = tuple(self._config.get("original_size", (1024, 1024)))
        target_size = tuple(self._config.get("target_size", (1024, 1024)))
        crops_coords = tuple(self._config.get("crops_coords_top_left", (0, 0)))

        if requires_aesthetics:
            aesthetic_score = self._config.get("aesthetic_score", 6.0)
            neg_aesthetic = self._config.get("negative_aesthetic_score", 2.5)
            time_ids = self._build_aesthetic_time_ids(original_size, crops_coords, aesthetic_score)
            neg_time_ids = self._build_aesthetic_time_ids(original_size, crops_coords, neg_aesthetic)
        else:
            time_ids = self._build_time_ids(original_size, crops_coords, target_size)
            neg_original = tuple(self._config.get("negative_original_size", original_size))
            neg_target = tuple(self._config.get("negative_target_size", target_size))
            neg_crops = tuple(self._config.get("negative_crops_coords_top_left", crops_coords))
            neg_time_ids = self._build_time_ids(neg_original, neg_crops, neg_target)

        batch_size = pooled.shape[0] if pooled.ndim > 1 else 1
        cast_dev, cast_dtype = _device_dtype_like(pooled)
        time_ids = time_ids.unsqueeze(0).expand(batch_size, -1).to(device=cast_dev, dtype=cast_dtype)
        neg_time_ids = neg_time_ids.unsqueeze(0).expand(batch_size, -1).to(device=cast_dev, dtype=cast_dtype)

        return {
            C.PORT_ADD_TEXT_EMBEDS: pooled,
            C.PORT_ADD_TIME_IDS: time_ids,
            C.PORT_NEGATIVE_ADD_TIME_IDS: neg_time_ids,
        }
