"""Hugging Face Diffusers integration for YggDrasill.

Provides concrete task-node implementations backed by real Diffusers models
for SD1.5, SDXL, FLUX, and adapter support (LoRA, ControlNet, IP-Adapter).

Importing this addon registers diffusion blocks in the registry and exposes
diffusion-specific builders and runners. It does **not** modify core
``Hypergraph`` semantics globally.
"""
from __future__ import annotations


def _suppress_hf_loading_messages() -> None:
    """Suppress transformers verbosity and HF Hub progress bars (avoids ipywidgets render errors)."""
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

from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes  # noqa: E402
from yggdrasill.foundation.registry import BlockRegistry  # noqa: E402
from yggdrasill.integrations.diffusers.family_registry import _bootstrap_families  # noqa: E402

# Register blocks and families on opt-in addon import.
register_diffusion_nodes(BlockRegistry.global_registry())
_bootstrap_families()

# Public API
from yggdrasill.integrations.diffusers.contracts import *  # noqa: E402, F401, F403
from yggdrasill.integrations.diffusers.output import DiffusionOutput  # noqa: E402
from yggdrasill.integrations.diffusers.run import run as run_diffusion, verify_devices  # noqa: E402
from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder  # noqa: E402
from yggdrasill.integrations.diffusers.templates import (  # noqa: E402
    build_template,
    from_template,
    list_templates,
)
from yggdrasill.integrations.diffusers.factory import (  # noqa: E402
    build_sd15_pipeline,
    build_sdxl_pipeline,
    build_sdxl_base_refiner,
    build_flux_pipeline,
)
from yggdrasill.integrations.diffusers.common.ip_adapter_embeds import (  # noqa: E402
    prepare_ip_adapter_image_embeds,
)
from yggdrasill.integrations.diffusers.common.ip_adapter_mask_prep import (  # noqa: E402
    IPAdapterMaskPrepNode,
    prepare_ip_adapter_masks_tensor,
)

__all__ = [
    "DiffusionOutput",
    "verify_devices",
    "DiffusionGraphBuilder",
    "run_diffusion",
    "prepare_ip_adapter_image_embeds",
    "IPAdapterMaskPrepNode",
    "prepare_ip_adapter_masks_tensor",
    "build_template",
    "from_template",
    "list_templates",
    "build_sd15_pipeline",
    "build_sdxl_pipeline",
    "build_sdxl_base_refiner",
    "build_flux_pipeline",
    "register_diffusion_nodes",
]
