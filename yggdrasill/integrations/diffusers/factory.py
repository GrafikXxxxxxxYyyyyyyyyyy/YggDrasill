"""High-level factory API for building SD1.5/SDXL/FLUX pipelines.

These functions provide the simplest way to create a ready-to-run
diffusion graph from a model repo_id or local checkpoint path.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.integrations.diffusers.presets.sd15 import (
    build_sd15_text2img_graph,
    build_sd15_img2img_graph,
    build_sd15_inpaint_graph,
)
from yggdrasill.integrations.diffusers.presets.sdxl import (
    build_sdxl_text2img_graph,
    build_sdxl_img2img_graph,
    build_sdxl_inpaint_graph,
    build_sdxl_base_refiner_workflow,
)
from yggdrasill.integrations.diffusers.presets.flux import (
    build_flux_text2img_graph,
    build_flux_img2img_graph,
    build_flux_inpaint_graph,
    build_flux_controlnet_text2img_graph,
)
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers.model_store import ModelStore
from yggdrasill.workflow.workflow import Workflow


def build_sd15_pipeline(
    repo_id: str = "runwayml/stable-diffusion-v1-5",
    *,
    task: str = "text2img",
    variant: str = "",
    torch_dtype: str = "float32",
    device: str = "cuda",
    enable_safety: bool = True,
    store: Optional[ModelStore] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Hypergraph:
    """Build a complete SD1.5 pipeline graph from a HF repo.

    Args:
        repo_id: HF Hub repo id or local path.
        task: One of "text2img", "img2img", "inpaint".
        variant: Model variant (e.g. "fp16").
        torch_dtype: PyTorch dtype string.
        device: Target device.
        enable_safety: Whether to include safety checker.
        store: Optional ModelStore for component caching.
        config: Additional config overrides.

    Scheduler and other nodes can be replaced via graph.replace_node().
    """
    import torch
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(torch_dtype, torch.float32)

    ms = store or ModelStore.default()
    sd15_keys = ["tokenizer", "text_encoder", "unet", "vae", "scheduler"]
    components = ms.load_components_by_keys(
        "sd15", sd15_keys, repo_id,
        variant=variant if variant else ("fp16" if dtype == torch.float16 else ""),
        torch_dtype=dtype,
    )

    cfg = dict(config or {})
    cfg.setdefault("device", device)
    cfg.setdefault("dtype", torch_dtype)
    cfg.setdefault("enable_safety", enable_safety)
    cfg.update(kwargs)

    comp_kwargs = {
        "tokenizer": components.get("tokenizer"),
        "text_encoder": components.get("text_encoder"),
        "unet": components.get("unet"),
        "vae": components.get("vae"),
        "scheduler": components.get("scheduler"),
        "config": cfg,
    }

    if enable_safety:
        comp_kwargs["safety_checker"] = None
        comp_kwargs["feature_extractor"] = None

    builders = {
        "text2img": build_sd15_text2img_graph,
        "img2img": build_sd15_img2img_graph,
        "inpaint": build_sd15_inpaint_graph,
    }

    if task not in builders:
        raise ValueError(f"Unknown task '{task}', expected one of {list(builders.keys())}")

    graph = builders[task](**comp_kwargs)

    if device != "cpu":
        graph.to(device)

    return graph


def build_sdxl_pipeline(
    repo_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    *,
    task: str = "text2img",
    variant: str = "fp16",
    torch_dtype: str = "float16",
    device: str = "cuda",
    store: Optional[ModelStore] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Hypergraph:
    """Build a complete SDXL pipeline graph from a HF repo."""
    import torch
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(torch_dtype, torch.float16)

    ms = store or ModelStore.default()
    sdxl_keys = ["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "unet", "vae", "scheduler"]
    components = ms.load_components_by_keys(
        "sdxl", sdxl_keys, repo_id,
        variant=variant, torch_dtype=dtype,
    )

    cfg = dict(config or {})
    cfg.setdefault("device", device)
    cfg.setdefault("dtype", torch_dtype)
    cfg.update(kwargs)

    comp_kwargs = {
        "tokenizer": components.get("tokenizer"),
        "tokenizer_2": components.get("tokenizer_2"),
        "text_encoder": components.get("text_encoder"),
        "text_encoder_2": components.get("text_encoder_2"),
        "unet": components.get("unet"),
        "vae": components.get("vae"),
        "scheduler": components.get("scheduler"),
        "config": cfg,
    }

    builders = {
        "text2img": build_sdxl_text2img_graph,
        "img2img": build_sdxl_img2img_graph,
        "inpaint": build_sdxl_inpaint_graph,
    }

    if task not in builders:
        raise ValueError(f"Unknown task '{task}', expected one of {list(builders.keys())}")

    graph = builders[task](**comp_kwargs)

    if device != "cpu":
        graph.to(device)

    return graph


def build_sdxl_base_refiner(
    base_repo_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    refiner_repo_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
    *,
    variant: str = "fp16",
    torch_dtype: str = "float16",
    device: str = "cuda",
    denoising_end: float = 0.8,
    store: Optional[ModelStore] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Workflow:
    """Build SDXL base+refiner as a two-stage Workflow."""
    import torch
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(torch_dtype, torch.float16)

    ms = store or ModelStore.default()
    sdxl_keys = ["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "unet", "vae", "scheduler"]

    base_components = ms.load_components_by_keys(
        "sdxl", sdxl_keys, base_repo_id,
        variant=variant, torch_dtype=dtype,
    )
    refiner_components = ms.load_components_by_keys(
        "sdxl", sdxl_keys, refiner_repo_id,
        variant=variant, torch_dtype=dtype,
    )

    base_kwargs = {
        "tokenizer": base_components.get("tokenizer"),
        "tokenizer_2": base_components.get("tokenizer_2"),
        "text_encoder": base_components.get("text_encoder"),
        "text_encoder_2": base_components.get("text_encoder_2"),
        "unet": base_components.get("unet"),
        "vae": base_components.get("vae"),
        "scheduler": base_components.get("scheduler"),
    }

    refiner_kwargs = {
        "tokenizer": refiner_components.get("tokenizer"),
        "tokenizer_2": refiner_components.get("tokenizer_2"),
        "text_encoder": refiner_components.get("text_encoder"),
        "text_encoder_2": refiner_components.get("text_encoder_2"),
        "unet": refiner_components.get("unet"),
        "vae": base_components.get("vae"),
        "scheduler": refiner_components.get("scheduler"),
    }

    cfg = dict(config or {})
    cfg.setdefault("device", device)
    cfg.setdefault("dtype", torch_dtype)
    cfg["denoising_end"] = denoising_end

    workflow = build_sdxl_base_refiner_workflow(
        base_components=base_kwargs,
        refiner_components=refiner_kwargs,
        config=cfg,
    )

    if device != "cpu":
        workflow.to(device)

    return workflow


def build_flux_pipeline(
    repo_id: str = "black-forest-labs/FLUX.1-dev",
    *,
    task: str = "text2img",
    variant: str = "",
    torch_dtype: str = "bfloat16",
    device: str = "cuda",
    store: Optional[ModelStore] = None,
    controlnet_repo_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Hypergraph:
    """Build a complete FLUX pipeline graph from a HF repo.

    Args:
        repo_id: HF Hub repo id or local path.
        task: One of "text2img", "img2img", "inpaint", "controlnet_text2img".
        variant: Model variant (empty for FLUX which uses bf16 natively).
        torch_dtype: PyTorch dtype string (default "bfloat16").
        device: Target device.
        store: Optional ModelStore for component caching.
        controlnet_repo_id: HF repo for ControlNet (required for controlnet tasks).
        config: Additional config overrides.
    """
    import torch
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    ms = store or ModelStore.default()
    flux_keys = ["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "transformer", "vae", "scheduler"]
    components = ms.load_components_by_keys(
        "flux", flux_keys, repo_id,
        variant=variant or "", torch_dtype=dtype,
    )

    cfg = dict(config or {})
    cfg.setdefault("device", device)
    cfg.setdefault("dtype", torch_dtype)
    cfg.update(kwargs)

    comp_kwargs: Dict[str, Any] = {
        "tokenizer": components.get("tokenizer"),
        "tokenizer_2": components.get("tokenizer_2"),
        "text_encoder": components.get("text_encoder"),
        "text_encoder_2": components.get("text_encoder_2"),
        "transformer": components.get("transformer"),
        "vae": components.get("vae"),
        "scheduler": components.get("scheduler"),
        "config": cfg,
    }

    builders: Dict[str, Any] = {
        "text2img": build_flux_text2img_graph,
        "img2img": build_flux_img2img_graph,
        "inpaint": build_flux_inpaint_graph,
        "controlnet_text2img": build_flux_controlnet_text2img_graph,
    }

    if task not in builders:
        raise ValueError(f"Unknown task '{task}', expected one of {list(builders.keys())}")

    if task == "controlnet_text2img":
        if controlnet_repo_id is None:
            raise ValueError("controlnet_repo_id is required for controlnet tasks")
        cn_components = ms.load_components_by_keys(
            "flux", ["controlnet"], controlnet_repo_id,
            variant=variant or "", torch_dtype=dtype,
        )
        comp_kwargs["controlnet"] = cn_components.get("controlnet") or cn_components.get("model")

    graph = builders[task](**comp_kwargs)

    if device != "cpu":
        graph.to(device)

    return graph
