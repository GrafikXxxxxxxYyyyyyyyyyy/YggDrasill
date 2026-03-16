"""SD1.5 safety checker and postprocess nodes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractHelper


class SD15SafetyNode(AbstractHelper):
    """Runs StableDiffusionSafetyChecker on decoded images."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        safety_checker: Any = None,
        feature_extractor: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._safety_checker = safety_checker
        self._feature_extractor = feature_extractor

    @property
    def block_type(self) -> str:
        return "sd15/safety"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_DECODED_IMAGE, PortDirection.IN, PortType.IMAGE),
            Port(C.PORT_OUTPUT_IMAGE, PortDirection.OUT, PortType.IMAGE),
            Port(C.PORT_NSFW_DETECTED, PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        images = inputs[C.PORT_DECODED_IMAGE]

        if self._safety_checker is None or not self._config.get("enabled", True):
            return {
                C.PORT_OUTPUT_IMAGE: images,
                C.PORT_NSFW_DETECTED: [False] * (len(images) if isinstance(images, list) else 1),
            }

        import numpy as np

        if isinstance(images, list):
            np_images = np.stack([np.array(img) for img in images])
        elif hasattr(images, "numpy"):
            np_images = images.cpu().numpy() if hasattr(images, "cpu") else images.numpy()
        else:
            np_images = np.array(images)

        if np_images.ndim == 3:
            np_images = np_images[np.newaxis, ...]

        safety_input = self._feature_extractor(
            [img for img in np_images],
            return_tensors="pt",
        ).to(self._safety_checker.device)

        filtered, nsfw = self._safety_checker(
            images=np_images,
            clip_input=safety_input.pixel_values,
        )

        from yggdrasill.integrations.diffusers.common.image_utils import numpy_to_pil
        result_images = numpy_to_pil(filtered) if isinstance(filtered, np.ndarray) else filtered

        return {
            C.PORT_OUTPUT_IMAGE: result_images,
            C.PORT_NSFW_DETECTED: nsfw,
        }

    def to(self, device: Any) -> "SD15SafetyNode":
        if self._safety_checker is not None:
            self._safety_checker.to(device)
        return self
