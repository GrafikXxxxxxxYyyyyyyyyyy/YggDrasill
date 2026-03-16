"""Hugging Face Diffusers integration for YggDrasill.

Provides concrete task-node implementations backed by real Diffusers models
for SD1.5, SDXL, and adapter support (LoRA, ControlNet, IP-Adapter).

On import, registers diffusion blocks in BlockRegistry and bootstraps
the family registry. Use ``from_template``, ``build_sd15_pipeline``, etc.
"""
from __future__ import annotations


def _suppress_hf_loading_messages() -> None:
    """Disable Hugging Face transformers/hub progress bars and load reports."""
    try:
        from transformers import logging as tf_logging
        tf_logging.set_verbosity_error()
        if hasattr(tf_logging, "disable_progress_bar"):
            tf_logging.disable_progress_bar()
    except ImportError:
        pass
    try:
        from huggingface_hub import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass


_suppress_hf_loading_messages()

from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.integrations.diffusers.family_registry import _bootstrap_families

# Register blocks and families on import
register_diffusion_nodes(BlockRegistry.global_registry())
_bootstrap_families()


def _wrap_hypergraph_run_for_diffusion():
    """Patch Hypergraph.run to return DiffusionOutput when output has diffusion keys."""
    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers.contracts import (
        PORT_DECODED_IMAGE,
        PORT_OUTPUT_IMAGE,
    )

    _original_run = Hypergraph.run

    def _run_with_diffusion_output(self, *args, **kwargs):
        result = _original_run(self, *args, **kwargs)
        if isinstance(result, dict) and (
            PORT_OUTPUT_IMAGE in result or PORT_DECODED_IMAGE in result
        ):
            return DiffusionOutput.from_executor_output(result)
        return result

    Hypergraph.run = _run_with_diffusion_output


_wrap_hypergraph_run_for_diffusion()

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
