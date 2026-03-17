"""Hugging Face Diffusers integration for YggDrasill.

Provides concrete task-node implementations backed by real Diffusers models
for SD1.5, SDXL, and adapter support (LoRA, ControlNet, IP-Adapter).

On import, registers diffusion blocks in BlockRegistry and bootstraps
the family registry. Use ``from_template``, ``build_sd15_pipeline``, etc.
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

from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.integrations.diffusers.family_registry import _bootstrap_families

# Register blocks and families on import
register_diffusion_nodes(BlockRegistry.global_registry())
_bootstrap_families()


def _wrap_hypergraph_run_for_diffusion():
    """Patch Hypergraph.run to expand diffusion kwargs and return DiffusionOutput."""
    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers.contracts import (
        PORT_DECODED_IMAGE,
        PORT_OUTPUT_IMAGE,
        PORT_CONTROL_IMAGE,
        PORT_IP_ADAPTER_IMAGE,
    )

    _original_run = Hypergraph.run

    def _run_with_diffusion_output(self, inputs=None, **kwargs):
        from yggdrasill.integrations.diffusers.run import (
            _assign_to_single_exposed,
            _inject_ip_adapter_scale,
            _inject_node_config,
        )
        merged = dict(inputs or {})
        controlnet_image = kwargs.pop("controlnet_image", None)
        if isinstance(controlnet_image, dict):
            for nid, img in controlnet_image.items():
                merged[f"{nid}:{PORT_CONTROL_IMAGE}"] = img
        elif controlnet_image is not None:
            _assign_to_single_exposed(merged, self, PORT_CONTROL_IMAGE, controlnet_image)
        ip_adapter_image = kwargs.pop("ip_adapter_image", None)
        if isinstance(ip_adapter_image, dict):
            for nid, img in ip_adapter_image.items():
                merged[f"{nid}:{PORT_IP_ADAPTER_IMAGE}"] = img
        elif ip_adapter_image is not None:
            _assign_to_single_exposed(merged, self, PORT_IP_ADAPTER_IMAGE, ip_adapter_image)
        controlnet_conditioning_scale = kwargs.pop("controlnet_conditioning_scale", None)
        if isinstance(controlnet_conditioning_scale, dict):
            _inject_node_config(self, controlnet_conditioning_scale, "conditioning_scale")
        ip_adapter_conditioning_scale = kwargs.pop("ip_adapter_conditioning_scale", None)
        if isinstance(ip_adapter_conditioning_scale, dict):
            _inject_ip_adapter_scale(self, ip_adapter_conditioning_scale)
        elif ip_adapter_conditioning_scale is not None:
            _inject_ip_adapter_scale(self, {"default": ip_adapter_conditioning_scale})
        result = _original_run(self, merged, **kwargs)
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
from yggdrasill.integrations.diffusers.run import run as run_diffusion, verify_devices
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
    "verify_devices",
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
