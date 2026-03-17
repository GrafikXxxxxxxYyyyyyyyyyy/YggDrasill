"""Unified family registry for diffusion model families.

Register a family with register_family() to enable component resolution,
pretrained loading, implicit nodes, and canonical exposed ports.
Adding a new family (e.g. SD3) = one register_family() call + node implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class FamilySpec:
    """Per-family configuration for the diffusion add-on."""

    component_specs: Dict[str, Any]
    load_keys: List[str]
    implicit_specs: List[Tuple[str, str]]
    block_defaults: Dict[str, Dict[str, Any]]
    canonical_required: Set[str]
    torch_dtype_default: str = "float16"


FAMILY_SPECS: Dict[str, FamilySpec] = {}


def register_family(
    family: str,
    *,
    component_specs: Dict[str, Any],
    load_keys: List[str],
    implicit_specs: List[Tuple[str, str]],
    block_defaults: Dict[str, Dict[str, Any]],
    canonical_required: Set[str],
    torch_dtype_default: str = "float16",
) -> None:
    """Register a diffusion model family.

    Adding SD3 example:
        register_family("sd3", component_specs={...}, load_keys=[...], ...)
    """
    FAMILY_SPECS[family] = FamilySpec(
        component_specs=component_specs,
        load_keys=load_keys,
        implicit_specs=implicit_specs,
        block_defaults=block_defaults,
        canonical_required=canonical_required,
        torch_dtype_default=torch_dtype_default,
    )


def unregister_family(family: str) -> None:
    """Remove a family from the registry."""
    FAMILY_SPECS.pop(family, None)


def get_family_spec(family: str) -> FamilySpec:
    """Get spec for a family. Raises KeyError if unknown."""
    if family not in FAMILY_SPECS:
        raise KeyError(
            f"Unknown family '{family}'. Registered: {sorted(FAMILY_SPECS.keys())}"
        )
    return FAMILY_SPECS[family]


def _bootstrap_families() -> None:
    """Register sd15, sdxl, flux, adapter with component_specs and metadata."""
    from yggdrasill.integrations.diffusers.components import (
        _SD15_COMPONENTS,
        _SDXL_COMPONENTS,
        _FLUX_COMPONENTS,
        _ADAPTER_COMPONENTS,
    )

    _ADAPTER_ONLY = {k: v for k, v in _ADAPTER_COMPONENTS.items() if k.startswith("adapter.")}

    # SD15
    register_family(
        "sd15",
        component_specs=dict(_SD15_COMPONENTS),
        load_keys=["tokenizer", "text_encoder", "unet", "vae", "scheduler"],
        implicit_specs=[
            ("latent_init", "sd15/latent_init"),
            ("tokenizer", "sd15/tokenizer"),
            ("scheduler_setup", "sd15/scheduler_setup"),
            ("scheduler_step", "sd15/scheduler_step"),
        ],
        block_defaults={},
        canonical_required={"prompt", "decoded_image"},
    )

    # SDXL
    register_family(
        "sdxl",
        component_specs=dict(_SDXL_COMPONENTS),
        load_keys=[
            "tokenizer", "tokenizer_2",
            "text_encoder", "text_encoder_2",
            "unet", "vae", "scheduler",
        ],
        implicit_specs=[
            ("latent_init", "sdxl/latent_init"),
            ("tokenizer", "sdxl/tokenizer"),
            ("scheduler_setup", "sdxl/scheduler_setup"),
            ("scheduler_step", "sdxl/scheduler_step"),
        ],
        block_defaults={},
        canonical_required={"prompt", "prompt_2", "decoded_image"},
    )

    _FLUX_ADAPTERS = {k: v for k, v in _ADAPTER_COMPONENTS.items() if k.startswith("flux.")}

    # FLUX
    register_family(
        "flux",
        component_specs=dict(**_FLUX_COMPONENTS, **_FLUX_ADAPTERS),
        load_keys=[
            "tokenizer", "tokenizer_2",
            "text_encoder", "text_encoder_2",
            "transformer", "vae", "scheduler",
        ],
        implicit_specs=[
            ("latent_init", "flux/latent_init"),
            ("tokenizer", "flux/tokenizer"),
            ("scheduler_setup", "flux/scheduler_setup"),
            ("scheduler_step", "flux/scheduler_step"),
        ],
        block_defaults={},
        canonical_required={"prompt", "prompt_2", "decoded_image"},
        torch_dtype_default="bfloat16",
    )

    # Adapter (controlnet for sd15/sdxl; controlnet_flux for FLUX)
    register_family(
        "adapter",
        component_specs=dict(_ADAPTER_ONLY),
        load_keys=["controlnet", "image_encoder", "feature_extractor"],
        implicit_specs=[],
        block_defaults={},
        canonical_required=set(),
    )
