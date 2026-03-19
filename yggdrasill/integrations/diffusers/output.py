"""Structured output wrapper for diffusion graph results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiffusionOutput:
    """Wraps raw executor output into a user-friendly object.

    Attributes
    ----------
    images : list
        Generated images (PIL, numpy, or tensor depending on output_type).
    latents : optional
        Raw latent tensors when output_type is ``"latent"``.
    nsfw_content_detected : optional
        Per-image NSFW flag list (when safety checker is active).
    raw : dict
        The original executor output dict for advanced access.
    """

    images: List[Any] = field(default_factory=list)
    latents: Optional[Any] = None
    nsfw_content_detected: Optional[List[bool]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_executor_output(
        cls,
        output: Dict[str, Any],
        *,
        image_key: Optional[str] = None,
        latent_key: str = "latents",
        nsfw_key: str = "nsfw_content_detected",
    ) -> "DiffusionOutput":
        """Build a ``DiffusionOutput`` from the raw dict returned by the executor."""
        if isinstance(output, cls):
            return output
        from yggdrasill.integrations.diffusers.contracts import (
            PORT_DECODED_IMAGE,
            PORT_OUTPUT_IMAGE,
        )
        if image_key is None:
            image_key = (
                PORT_DECODED_IMAGE if PORT_DECODED_IMAGE in output else
                PORT_OUTPUT_IMAGE if PORT_OUTPUT_IMAGE in output else
                next(
                    (k for k in output if k.endswith(":" + PORT_DECODED_IMAGE) or k.endswith(":" + PORT_OUTPUT_IMAGE)),
                    "decoded_image",
                )
            )
        images_raw = output.get(image_key)
        images: List[Any] = []
        if images_raw is not None:
            if isinstance(images_raw, list):
                images = images_raw
            else:
                images = [images_raw]

        return cls(
            images=images,
            latents=output.get(latent_key),
            nsfw_content_detected=output.get(nsfw_key),
            raw=output,
        )
