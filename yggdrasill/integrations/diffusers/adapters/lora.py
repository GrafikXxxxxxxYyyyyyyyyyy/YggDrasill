"""LoRA adapter support for SD1.5/SDXL UNet and text encoders."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractHelper


class LoRALoaderNode(AbstractHelper):
    """Loads and fuses LoRA weights into UNet and/or text encoder(s).

    This is a runtime overlay node: it mutates model weights in-place
    rather than producing output tensors through an edge. It runs once
    during graph setup, before the denoising loop.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        pipe: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._pipe = pipe

    @property
    def block_type(self) -> str:
        return "adapter/lora_loader"

    def declare_ports(self) -> List[Port]:
        return [
            Port("trigger", PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_LORA_SCALE, PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        lora_weights = self._config.get("lora_weights", [])
        adapter_names: List[str] = []
        adapter_weights: List[float] = []

        for i, lora in enumerate(lora_weights):
            name = lora.get("name", f"lora_{i}")
            weight_path = lora.get("path", "")
            weight_name = lora.get("weight_name")
            scale = lora.get("scale", 1.0)

            kwargs: Dict[str, Any] = {"adapter_name": name}
            if weight_name:
                kwargs["weight_name"] = weight_name

            self._pipe.load_lora_weights(weight_path, **kwargs)
            adapter_names.append(name)
            adapter_weights.append(scale)

        if adapter_names:
            self._pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

        lora_scale = inputs.get(C.PORT_LORA_SCALE)
        if lora_scale is not None and len(adapter_names) == 1:
            self._pipe.set_adapters(adapter_names, adapter_weights=[float(lora_scale)])

        return {"result": {"loaded_loras": adapter_names, "weights": adapter_weights}}

    def unfuse(self) -> None:
        """Remove fused LoRA weights from the pipeline."""
        if self._pipe is not None and hasattr(self._pipe, "unfuse_lora"):
            self._pipe.unfuse_lora()
            self._pipe.unload_lora_weights()
