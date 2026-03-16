"""Diffusers-specific configuration and environment utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from yggdrasill.integrations.diffusers.types import ModelDType


@dataclass
class DiffusersComponentConfig:
    """Configuration for a single Diffusers component (model, tokenizer, etc.)."""

    repo_id: str = ""
    subfolder: str = ""
    local_path: str = ""
    variant: str = ""
    revision: Optional[str] = None
    torch_dtype: str = "fp16"
    use_safetensors: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def dtype_enum(self) -> ModelDType:
        return {
            "fp32": ModelDType.FP32,
            "fp16": ModelDType.FP16,
            "bf16": ModelDType.BF16,
            "float32": ModelDType.FP32,
            "float16": ModelDType.FP16,
            "bfloat16": ModelDType.BF16,
        }.get(self.torch_dtype, ModelDType.FP16)

    @property
    def source(self) -> str:
        return self.local_path or self.repo_id


@dataclass
class SD15PipelineConfig:
    """Full configuration for an SD1.5 pipeline expressed as components."""

    model: DiffusersComponentConfig = field(default_factory=lambda: DiffusersComponentConfig(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    ))
    scheduler_class: str = "EulerDiscreteScheduler"
    enable_safety_checker: bool = True
    device: str = "cuda"
    torch_dtype: str = "fp16"
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False


@dataclass
class SDXLPipelineConfig:
    """Full configuration for an SDXL pipeline expressed as components."""

    model: DiffusersComponentConfig = field(default_factory=lambda: DiffusersComponentConfig(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    ))
    refiner: Optional[DiffusersComponentConfig] = None
    scheduler_class: str = "EulerDiscreteScheduler"
    device: str = "cuda"
    torch_dtype: str = "fp16"
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    force_zeros_for_empty_prompt: bool = True
