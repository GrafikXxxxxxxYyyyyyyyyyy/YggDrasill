"""Per-component loaders: each model loads separately via its own from_pretrained.

No full pipeline load. Caching happens in ModelStore.
Scheduler for sd15 is loaded from repo config (_class_name) for parity with diffusers.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

# (cls, subfolder) per (family, load_key). cls=None means use dynamic loader.
_COMPONENT_REGISTRY: Dict[Tuple[str, str], Tuple[Optional[Type[Any]], str]] = {}


def _register(family: str, load_key: str, cls: Optional[Type[Any]], subfolder: str) -> None:
    _COMPONENT_REGISTRY[(family, load_key)] = (cls, subfolder)


def load_scheduler_from_repo(repo_id: str, subfolder: str = "scheduler") -> Any:
    """Load scheduler from repo config (_class_name) — parity with diffusers pipeline."""
    from huggingface_hub import hf_hub_download

    import json
    config_path = hf_hub_download(repo_id, f"{subfolder}/scheduler_config.json")
    with open(config_path) as f:
        config = json.load(f)
    class_name = config.pop("_class_name", "DDIMScheduler")
    if class_name.startswith("diffusers."):
        class_name = class_name.split(".")[-1]
    mod = __import__("diffusers.schedulers", fromlist=[class_name])
    scheduler_cls = getattr(mod, class_name)
    return scheduler_cls.from_pretrained(repo_id, subfolder=subfolder)


def _init_sd15() -> None:
    from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    _register("sd15", "unet", UNet2DConditionModel, "unet")
    _register("sd15", "vae", AutoencoderKL, "vae")
    _register("sd15", "text_encoder", CLIPTextModel, "text_encoder")
    _register("sd15", "tokenizer", CLIPTokenizer, "tokenizer")
    _register("sd15", "scheduler", None, "scheduler")
    _register("sd15", "controlnet", ControlNetModel, "")


def _init_sdxl() -> None:
    from diffusers import AutoencoderKL, ControlNetModel, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    _register("sdxl", "unet", UNet2DConditionModel, "unet")
    _register("sdxl", "controlnet", ControlNetModel, "")
    _register("sdxl", "vae", AutoencoderKL, "vae")
    _register("sdxl", "text_encoder", CLIPTextModel, "text_encoder")
    _register("sdxl", "text_encoder_2", CLIPTextModelWithProjection, "text_encoder_2")
    _register("sdxl", "tokenizer", CLIPTokenizer, "tokenizer")
    _register("sdxl", "tokenizer_2", CLIPTokenizer, "tokenizer_2")
    _register("sdxl", "scheduler", EulerDiscreteScheduler, "scheduler")


def _init_adapter() -> None:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    _register("adapter", "image_encoder", CLIPVisionModelWithProjection, "")
    _register("adapter", "feature_extractor", CLIPImageProcessor, "")


def _init_flux() -> None:
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        FlowMatchEulerDiscreteScheduler,
        FluxTransformer2DModel,
    )
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer

    _register("flux", "transformer", FluxTransformer2DModel, "transformer")
    _register("flux", "vae", AutoencoderKL, "vae")
    _register("flux", "text_encoder", CLIPTextModelWithProjection, "text_encoder")
    _register("flux", "text_encoder_2", T5EncoderModel, "text_encoder_2")
    _register("flux", "tokenizer", CLIPTokenizer, "tokenizer")
    _register("flux", "tokenizer_2", T5Tokenizer, "tokenizer_2")
    _register("flux", "scheduler", FlowMatchEulerDiscreteScheduler, "scheduler")
    _register("flux", "controlnet", ControlNetModel, "")


def _ensure_registry() -> None:
    if not _COMPONENT_REGISTRY:
        _init_sd15()
        _init_sdxl()
        _init_adapter()
        _init_flux()


def get_loader(family: str, load_key: str) -> Optional[Tuple[Type[Any], str]]:
    """Return (cls, subfolder) for (family, load_key), or None."""
    _ensure_registry()
    return _COMPONENT_REGISTRY.get((family, load_key))


def get_block_type_load_keys(block_type: str) -> list:
    """Map block_type (e.g. sd15/unet) to load_keys (e.g. [unet])."""
    if "/" not in block_type:
        return []
    _family, suffix = block_type.split("/", 1)
    key_map: Dict[str, list] = {
        "unet": ["unet"],
        "transformer": ["transformer"],
        "tokenizer": ["tokenizer"],
        "tokenizer_2": ["tokenizer_2"],
        "prompt_encoder": ["text_encoder"],
        "text_encoder": ["text_encoder"],
        "text_encoder_2": ["text_encoder_2"],
        "scheduler_setup": ["scheduler"],
        "scheduler_step": ["scheduler"],
        "vae_encode": ["vae"],
        "vae_decode": ["vae"],
    }
    return key_map.get(suffix, [])
