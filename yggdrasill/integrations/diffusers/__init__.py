"""Hugging Face Diffusers integration for YggDrasill.

Provides concrete task-node implementations backed by real Diffusers models
for SD1.5, SDXL, and adapter support (LoRA, ControlNet, IP-Adapter).

On import, registers diffusion blocks in BlockRegistry and bootstraps
the family registry. Use ``from_template``, ``build_sd15_pipeline``, etc.
"""
from __future__ import annotations

from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.integrations.diffusers.family_registry import _bootstrap_families

# Register blocks and families on import
register_diffusion_nodes(BlockRegistry.global_registry())
_bootstrap_families()

# Public API
from yggdrasill.integrations.diffusers.contracts import *  # noqa: F401, F403
from yggdrasill.integrations.diffusers.output import DiffusionOutput
from yggdrasill.integrations.diffusers.run import run as run_diffusion
from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder
from yggdrasill.integrations.diffusers.templates import (
    build_template,
    from_template,
    list_templates,
)
from yggdrasill.integrations.diffusers.factory import (
    build_sd15_pipeline,
    build_sdxl_pipeline,
    build_sdxl_base_refiner,
    build_flux_pipeline,
)

__all__ = [
    "DiffusionOutput",
    "DiffusionGraphBuilder",
    "run_diffusion",
    "build_template",
    "from_template",
    "list_templates",
    "build_sd15_pipeline",
    "build_sdxl_pipeline",
    "build_sdxl_base_refiner",
    "build_flux_pipeline",
    "register_diffusion_nodes",
]
