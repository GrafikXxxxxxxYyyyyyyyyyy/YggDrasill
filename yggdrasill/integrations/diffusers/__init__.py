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

from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes  # noqa: E402
from yggdrasill.foundation.registry import BlockRegistry  # noqa: E402
from yggdrasill.integrations.diffusers.family_registry import _bootstrap_families  # noqa: E402

# Register blocks and families on import
register_diffusion_nodes(BlockRegistry.global_registry())
_bootstrap_families()


def _wrap_hypergraph_run_for_diffusion():
    """Patch Hypergraph.run to expand diffusion kwargs and return DiffusionOutput."""
    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers.contracts import (
        PORT_DECODED_IMAGE,
        PORT_INIT_IMAGE,
        PORT_OUTPUT_IMAGE,
        PORT_CONTROL_IMAGE,
        PORT_IP_ADAPTER_IMAGE_EMBEDS,
        PORT_IP_ADAPTER_MASK_IMAGES,
    )

    _original_run = Hypergraph.run

    def _run_with_diffusion_output(self, inputs=None, **kwargs):
        from yggdrasill.integrations.diffusers.run import (
            _assign_to_single_exposed,
            _route_ip_adapter_masks,
            _inject_ip_adapter_scale,
            _inject_node_config,
            _prepare_diffusion_run,
            merge_ip_adapter_image_kwarg,
            normalize_merged_adapter_inputs,
        )
        merged = dict(inputs or {})
        normalize_merged_adapter_inputs(merged, self)
        if "image" in merged and PORT_INIT_IMAGE not in merged:
            merged[PORT_INIT_IMAGE] = merged.pop("image")
        if "image" in kwargs:
            _img = kwargs.pop("image")
            kwargs.setdefault(PORT_INIT_IMAGE, _img)
        controlnet_image = kwargs.pop("controlnet_image", None)
        if isinstance(controlnet_image, dict):
            for nid, img in controlnet_image.items():
                if img is not None:
                    merged[f"{nid}:{PORT_CONTROL_IMAGE}"] = img
        elif controlnet_image is not None:
            _assign_to_single_exposed(merged, self, PORT_CONTROL_IMAGE, controlnet_image)
        ip_adapter_image = kwargs.pop("ip_adapter_image", None)
        merge_ip_adapter_image_kwarg(merged, self, ip_adapter_image)
        ip_adapter_image_embeds = kwargs.pop("ip_adapter_image_embeds", None)
        if isinstance(ip_adapter_image_embeds, dict):
            for nid, emb in ip_adapter_image_embeds.items():
                if emb is not None:
                    merged[f"{nid}:{PORT_IP_ADAPTER_IMAGE_EMBEDS}"] = emb
        elif ip_adapter_image_embeds is not None:
            _assign_to_single_exposed(
                merged, self, PORT_IP_ADAPTER_IMAGE_EMBEDS, ip_adapter_image_embeds,
            )
        ip_adapter_mask_images = kwargs.pop("ip_adapter_mask_images", None)
        if isinstance(ip_adapter_mask_images, dict):
            for nid, imgs in ip_adapter_mask_images.items():
                if imgs is not None:
                    merged[f"{nid}:{PORT_IP_ADAPTER_MASK_IMAGES}"] = imgs
        elif ip_adapter_mask_images is not None:
            _assign_to_single_exposed(
                merged, self, PORT_IP_ADAPTER_MASK_IMAGES, ip_adapter_mask_images,
            )
        ip_adapter_masks = kwargs.pop("ip_adapter_masks", None)
        if ip_adapter_masks is not None:
            _route_ip_adapter_masks(
                merged, self, ip_adapter_masks,
                pin_data=kwargs.setdefault("pin_data", {}),
            )
        # Route guess_mode into ControlNet nodes (handled inside _prepare_diffusion_run).
        controlnet_conditioning_scale = kwargs.pop("controlnet_conditioning_scale", None)
        if isinstance(controlnet_conditioning_scale, dict):
            _inject_node_config(self, controlnet_conditioning_scale, "conditioning_scale")
        ip_adapter_conditioning_scale = kwargs.pop("ip_adapter_conditioning_scale", None)
        if isinstance(ip_adapter_conditioning_scale, list):
            _inject_ip_adapter_scale(self, ip_adapter_conditioning_scale, merged)
        elif isinstance(ip_adapter_conditioning_scale, dict):
            _inject_ip_adapter_scale(self, ip_adapter_conditioning_scale, merged)
        elif ip_adapter_conditioning_scale is not None:
            _inject_ip_adapter_scale(self, {"default": float(ip_adapter_conditioning_scale)}, merged)
        else:
            _inject_ip_adapter_scale(self, {"default": 1.0}, merged)
        # Same as run_diffusion(): sync device, width/height onto latent_init + ControlNet
        # (structure._resolve_run_kwargs only patches keys already present in node._config).
        _prepare_diffusion_run(self, kwargs, merged_inputs=merged)
        result = _original_run(self, merged, **kwargs)
        if isinstance(result, dict) and (
            PORT_OUTPUT_IMAGE in result or PORT_DECODED_IMAGE in result
        ):
            return DiffusionOutput.from_executor_output(result)
        return result

    Hypergraph.run = _run_with_diffusion_output


_wrap_hypergraph_run_for_diffusion()

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
