"""Diffusion-specific enums and value types used across all providers."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


class DiffusionTask(Enum):
    TEXT2IMG = "text2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    UPSCALE = "upscale"


class OutputType(Enum):
    PIL = "pil"
    NUMPY = "np"
    TORCH = "pt"
    LATENT = "latent"


class ModelDType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    def to_torch(self) -> Any:
        import torch
        return {
            ModelDType.FP32: torch.float32,
            ModelDType.FP16: torch.float16,
            ModelDType.BF16: torch.bfloat16,
        }[self]


class ModelVariant(Enum):
    DEFAULT = ""
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class ModelRef:
    """Reference to a pretrained model or component, either HF Hub or local."""

    repo_id: str = ""
    subfolder: str = ""
    local_path: str = ""
    variant: ModelVariant = ModelVariant.DEFAULT
    revision: Optional[str] = None

    @property
    def is_local(self) -> bool:
        return bool(self.local_path)


@dataclass
class SchedulerConfig:
    """Provider-agnostic scheduler runtime parameters."""

    num_inference_steps: int = 50
    timesteps: Optional[List[int]] = None
    sigmas: Optional[List[float]] = None
    strength: float = 1.0
    denoising_start: Optional[float] = None
    denoising_end: Optional[float] = None


@dataclass
class GuidanceConfig:
    """Classifier-free guidance parameters."""

    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0
    do_classifier_free_guidance: bool = True

    def __post_init__(self) -> None:
        self.do_classifier_free_guidance = self.guidance_scale > 1.0


@dataclass
class ImageSize:
    """Standard image/latent size descriptor."""

    height: int = 512
    width: int = 512

    @property
    def latent_height(self) -> int:
        return self.height // 8

    @property
    def latent_width(self) -> int:
        return self.width // 8


@dataclass
class SDXLMicroConditioning:
    """SDXL micro-conditioning parameters for add_time_ids."""

    original_size: Tuple[int, int] = (1024, 1024)
    target_size: Tuple[int, int] = (1024, 1024)
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5


@dataclass
class GenerationParams:
    """Unified runtime generation parameters across tasks."""

    image_size: ImageSize = field(default_factory=lambda: ImageSize(512, 512))
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    output_type: OutputType = OutputType.PIL
    seed: Optional[int] = None
    num_images_per_prompt: int = 1
    clip_skip: Optional[int] = None
    sdxl_micro: Optional[SDXLMicroConditioning] = None
