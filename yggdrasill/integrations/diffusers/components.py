"""Component-level type registry for diffusion graphs.

Maps high-level component types (e.g. ``"sdxl.unet"``) to the concrete
block types, ModelStore loading keys, and constructor kwargs needed to
materialise one or more YggDrasill graph nodes.

Two levels are supported:

* **Block-level types** (contain ``/``, e.g. ``"sdxl/unet"``): map 1:1
  to a registered ``block_type`` and are built directly via the
  :class:`BlockRegistry`.
* **Component-level types** (contain ``.``, e.g. ``"sdxl.unet"``): map
  to one or more block-level nodes and know which model component to
  load from a pretrained HF repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ComponentSpec:
    """Describes how a component type maps to block-level nodes.

    Attributes
    ----------
    block_types : list[str]
        Block types to create (can be more than one, e.g. scheduler
        creates both setup + step nodes).
    load_keys : list[str]
        ModelStore / pipeline component names to extract from the
        pretrained repo for each block node.
    constructor_map : dict[str, dict[str, str]]
        ``{block_type: {constructor_kwarg: component_name}}``.
        Tells the factory which loaded component goes into which
        constructor kwarg for each block type.
    group : str | None
        If set, multiple component types with the same group contribute
        to a **single shared node**.  The first ``add_node`` in the group
        creates the node; subsequent calls update the existing node with
        additional model components.
    node_id_suffix : str | None
        When a component creates multiple block nodes, this suffix is
        appended to differentiate them (e.g. ``"_setup"`` / ``"_step"``
        for schedulers).
    load_family : str | None
        When set, overrides the component-type prefix for ModelStore loading.
        E.g. ``adapter.controlnet_flux`` uses ``family="flux"`` to load
        from (flux, controlnet).
    load_pretrained_map : dict[str, str] | None
        When set, maps each load_key to a specific pretrained repo.
        E.g. IP-Adapter image_encoder loads from h94/IP-Adapter.
    load_subfolder_map : dict[str, str] | None
        When set, overrides subfolder per load_key (used with load_pretrained_map).
        E.g. IP-Adapter image_encoder needs subfolder "models/image_encoder".
    load_variant_map : dict[str, str] | None
        When set, overrides variant per load_key. Use "" for components that
        don't support fp16 variant (e.g. openai/clip-vit-large-patch14).
    """

    block_types: List[str]
    load_keys: List[str] = field(default_factory=list)
    constructor_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    group: Optional[str] = None
    node_id_suffix: Optional[str] = None
    load_family: Optional[str] = None
    load_pretrained_map: Optional[Dict[str, str]] = None
    load_subfolder_map: Optional[Dict[str, str]] = None
    load_variant_map: Optional[Dict[str, str]] = None


# ── SD 1.5 components ──────────────────────────────────────────────────

_SD15_COMPONENTS: Dict[str, ComponentSpec] = {
    "sd15.backbone": ComponentSpec(
        block_types=["sd15/unet"],
        load_keys=["unet"],
        constructor_map={"sd15/unet": {"unet": "unet"}},
    ),
    "sd15.unet": ComponentSpec(
        block_types=["sd15/unet"],
        load_keys=["unet"],
        constructor_map={"sd15/unet": {"unet": "unet"}},
    ),
    "sd15.autoencoder": ComponentSpec(
        block_types=["sd15/vae_decode"],
        load_keys=["vae"],
        constructor_map={"sd15/vae_decode": {"vae": "vae"}},
    ),
    "sd15.vae": ComponentSpec(
        block_types=["sd15/vae_decode"],
        load_keys=["vae"],
        constructor_map={"sd15/vae_decode": {"vae": "vae"}},
    ),
    "sd15.scheduler": ComponentSpec(
        block_types=["sd15/scheduler_setup", "sd15/scheduler_step"],
        load_keys=["scheduler"],
        constructor_map={
            "sd15/scheduler_setup": {"scheduler": "scheduler"},
            "sd15/scheduler_step": {"scheduler": "scheduler"},
        },
    ),
    "sd15.tokenizer": ComponentSpec(
        block_types=["sd15/tokenizer"],
        load_keys=["tokenizer"],
        constructor_map={"sd15/tokenizer": {"tokenizer": "tokenizer"}},
    ),
    "sd15.prompt_encoder": ComponentSpec(
        block_types=["sd15/prompt_encoder"],
        load_keys=["text_encoder"],
        constructor_map={"sd15/prompt_encoder": {"text_encoder": "text_encoder"}},
        group="sd15.prompt_encoder",
    ),
    "sd15.text_encoder": ComponentSpec(
        block_types=["sd15/prompt_encoder"],
        load_keys=["text_encoder"],
        constructor_map={"sd15/prompt_encoder": {"text_encoder": "text_encoder"}},
        group="sd15.prompt_encoder",
    ),
    "sd15.latent_init": ComponentSpec(
        block_types=["sd15/latent_init"],
        load_keys=[],
        constructor_map={},
    ),
}

# ── SDXL components ────────────────────────────────────────────────────

_SDXL_COMPONENTS: Dict[str, ComponentSpec] = {
    "sdxl.backbone": ComponentSpec(
        block_types=["sdxl/unet"],
        load_keys=["unet"],
        constructor_map={"sdxl/unet": {"unet": "unet"}},
    ),
    "sdxl.unet": ComponentSpec(
        block_types=["sdxl/unet"],
        load_keys=["unet"],
        constructor_map={"sdxl/unet": {"unet": "unet"}},
    ),
    "sdxl.autoencoder": ComponentSpec(
        block_types=["sdxl/vae_decode"],
        load_keys=["vae"],
        constructor_map={"sdxl/vae_decode": {"vae": "vae"}},
    ),
    "sdxl.vae": ComponentSpec(
        block_types=["sdxl/vae_decode"],
        load_keys=["vae"],
        constructor_map={"sdxl/vae_decode": {"vae": "vae"}},
    ),
    "sdxl.scheduler": ComponentSpec(
        block_types=["sdxl/scheduler_setup", "sdxl/scheduler_step"],
        load_keys=["scheduler"],
        constructor_map={
            "sdxl/scheduler_setup": {"scheduler": "scheduler"},
            "sdxl/scheduler_step": {"scheduler": "scheduler"},
        },
    ),
    "sdxl.tokenizer": ComponentSpec(
        block_types=["sdxl/tokenizer"],
        load_keys=["tokenizer", "tokenizer_2"],
        constructor_map={
            "sdxl/tokenizer": {"tokenizer": "tokenizer", "tokenizer_2": "tokenizer_2"},
        },
    ),
    "sdxl.tokenizer_1": ComponentSpec(
        block_types=["sdxl/tokenizer"],
        load_keys=["tokenizer"],
        constructor_map={"sdxl/tokenizer": {"tokenizer": "tokenizer"}},
        group="sdxl.tokenizer",
    ),
    "sdxl.tokenizer_2": ComponentSpec(
        block_types=["sdxl/tokenizer"],
        load_keys=["tokenizer_2"],
        constructor_map={"sdxl/tokenizer": {"tokenizer_2": "tokenizer_2"}},
        group="sdxl.tokenizer",
    ),
    "sdxl.prompt_encoder": ComponentSpec(
        block_types=["sdxl/prompt_encoder"],
        load_keys=["text_encoder", "text_encoder_2"],
        constructor_map={
            "sdxl/prompt_encoder": {
                "text_encoder": "text_encoder",
                "text_encoder_2": "text_encoder_2",
            },
        },
        group="sdxl.prompt_encoder",
    ),
    "sdxl.text_encoder": ComponentSpec(
        block_types=["sdxl/prompt_encoder"],
        load_keys=["text_encoder"],
        constructor_map={"sdxl/prompt_encoder": {"text_encoder": "text_encoder"}},
        group="sdxl.prompt_encoder",
    ),
    "sdxl.text_encoder_1": ComponentSpec(
        block_types=["sdxl/prompt_encoder"],
        load_keys=["text_encoder"],
        constructor_map={"sdxl/prompt_encoder": {"text_encoder": "text_encoder"}},
        group="sdxl.prompt_encoder",
    ),
    "sdxl.text_encoder_2": ComponentSpec(
        block_types=["sdxl/prompt_encoder"],
        load_keys=["text_encoder_2"],
        constructor_map={"sdxl/prompt_encoder": {"text_encoder_2": "text_encoder_2"}},
        group="sdxl.prompt_encoder",
    ),
    "sdxl.added_conditioning": ComponentSpec(
        block_types=["sdxl/added_conditioning"],
        load_keys=[],
        constructor_map={},
    ),
    "sdxl.latent_init": ComponentSpec(
        block_types=["sdxl/latent_init"],
        load_keys=[],
        constructor_map={},
    ),
}

# ── FLUX components ────────────────────────────────────────────────────

_FLUX_COMPONENTS: Dict[str, ComponentSpec] = {
    "flux.backbone": ComponentSpec(
        block_types=["flux/transformer"],
        load_keys=["transformer"],
        constructor_map={"flux/transformer": {"transformer": "transformer"}},
    ),
    "flux.transformer": ComponentSpec(
        block_types=["flux/transformer"],
        load_keys=["transformer"],
        constructor_map={"flux/transformer": {"transformer": "transformer"}},
    ),
    "flux.autoencoder": ComponentSpec(
        block_types=["flux/vae_decode"],
        load_keys=["vae"],
        constructor_map={"flux/vae_decode": {"vae": "vae"}},
    ),
    "flux.vae": ComponentSpec(
        block_types=["flux/vae_decode"],
        load_keys=["vae"],
        constructor_map={"flux/vae_decode": {"vae": "vae"}},
    ),
    "flux.scheduler": ComponentSpec(
        block_types=["flux/scheduler_setup", "flux/scheduler_step"],
        load_keys=["scheduler"],
        constructor_map={
            "flux/scheduler_setup": {"scheduler": "scheduler"},
            "flux/scheduler_step": {"scheduler": "scheduler"},
        },
    ),
    "flux.tokenizer": ComponentSpec(
        block_types=["flux/tokenizer"],
        load_keys=["tokenizer", "tokenizer_2"],
        constructor_map={
            "flux/tokenizer": {"tokenizer": "tokenizer", "tokenizer_2": "tokenizer_2"},
        },
    ),
    "flux.prompt_encoder": ComponentSpec(
        block_types=["flux/prompt_encoder"],
        load_keys=["text_encoder", "text_encoder_2"],
        constructor_map={
            "flux/prompt_encoder": {
                "text_encoder": "text_encoder",
                "text_encoder_2": "text_encoder_2",
            },
        },
        group="flux.prompt_encoder",
    ),
    "flux.text_encoder": ComponentSpec(
        block_types=["flux/prompt_encoder"],
        load_keys=["text_encoder"],
        constructor_map={"flux/prompt_encoder": {"text_encoder": "text_encoder"}},
        group="flux.prompt_encoder",
    ),
    "flux.text_encoder_2": ComponentSpec(
        block_types=["flux/prompt_encoder"],
        load_keys=["text_encoder_2"],
        constructor_map={"flux/prompt_encoder": {"text_encoder_2": "text_encoder_2"}},
        group="flux.prompt_encoder",
    ),
    "flux.latent_init": ComponentSpec(
        block_types=["flux/latent_init"],
        load_keys=[],
        constructor_map={},
    ),
}

# ── Adapter components ─────────────────────────────────────────────────

_ADAPTER_COMPONENTS: Dict[str, ComponentSpec] = {
    "adapter.controlnet": ComponentSpec(
        block_types=["adapter/controlnet"],
        load_keys=["controlnet"],
        constructor_map={"adapter/controlnet": {"controlnet": "controlnet"}},
    ),
    "sd15.controlnet": ComponentSpec(
        block_types=["adapter/controlnet"],
        load_keys=["controlnet"],
        constructor_map={"adapter/controlnet": {"controlnet": "controlnet"}},
        load_family="sd15",
    ),
    "sdxl.controlnet": ComponentSpec(
        block_types=["adapter/controlnet"],
        load_keys=["controlnet"],
        constructor_map={"adapter/controlnet": {"controlnet": "controlnet"}},
        load_family="sdxl",
    ),
    "sdxl.t2iadapter": ComponentSpec(
        block_types=["adapter/t2i_adapter"],
        load_keys=["t2iadapter"],
        constructor_map={"adapter/t2i_adapter": {"adapter": "t2iadapter"}},
        load_family="sdxl",
    ),
    "sd15.t2iadapter": ComponentSpec(
        block_types=["adapter/t2i_adapter"],
        load_keys=["t2iadapter"],
        constructor_map={"adapter/t2i_adapter": {"adapter": "t2iadapter"}},
        load_family="sd15",
    ),
    "sd15.ipadapter": ComponentSpec(
        block_types=["adapter/ip_adapter"],
        load_keys=["image_encoder", "feature_extractor"],
        constructor_map={
            "adapter/ip_adapter": {
                "image_encoder": "image_encoder",
                "feature_extractor": "feature_extractor",
            },
        },
        load_family="adapter",
        # ip-adapter_sd15.bin expects CLIP-ViT-H-14 (1024 dim)
        # image_encoder from h94; feature_extractor from openai (h94 lacks preprocessor_config.json)
        load_pretrained_map={
            "image_encoder": "h94/IP-Adapter",
            "feature_extractor": "openai/clip-vit-large-patch14",
        },
        load_subfolder_map={
            "image_encoder": "models/image_encoder",
        },
        load_variant_map={"image_encoder": "", "feature_extractor": ""},
    ),
    # Aliases for SD1.5 Plus / Plus-Face checkpoints (same loader; weight_name differs).
    # Builder sets defaults and enables `ip_adapter_use_hidden_states` for `plus*` weights.
    "sd15.ipadapter_plus": ComponentSpec(
        block_types=["adapter/ip_adapter"],
        load_keys=["image_encoder", "feature_extractor"],
        constructor_map={
            "adapter/ip_adapter": {
                "image_encoder": "image_encoder",
                "feature_extractor": "feature_extractor",
            },
        },
        load_family="adapter",
        load_pretrained_map={
            "image_encoder": "h94/IP-Adapter",
            "feature_extractor": "openai/clip-vit-large-patch14",
        },
        load_subfolder_map={"image_encoder": "models/image_encoder"},
        load_variant_map={"image_encoder": "", "feature_extractor": ""},
    ),
    "sd15.ipadapter_plus_face": ComponentSpec(
        block_types=["adapter/ip_adapter"],
        load_keys=["image_encoder", "feature_extractor"],
        constructor_map={
            "adapter/ip_adapter": {
                "image_encoder": "image_encoder",
                "feature_extractor": "feature_extractor",
            },
        },
        load_family="adapter",
        load_pretrained_map={
            "image_encoder": "h94/IP-Adapter",
            "feature_extractor": "openai/clip-vit-large-patch14",
        },
        load_subfolder_map={"image_encoder": "models/image_encoder"},
        load_variant_map={"image_encoder": "", "feature_extractor": ""},
    ),
    "sd15.ipadapter_faceid": ComponentSpec(
        # FaceID checkpoints operate on pre-computed InsightFace embeddings.
        # No CLIP vision encoder / feature extractor is required inside the graph.
        block_types=["adapter/ip_adapter"],
        load_keys=[],
        constructor_map={},
    ),
    "sdxl.ipadapter": ComponentSpec(
        block_types=["adapter/ip_adapter"],
        load_keys=["image_encoder", "feature_extractor"],
        constructor_map={
            "adapter/ip_adapter": {
                "image_encoder": "image_encoder",
                "feature_extractor": "feature_extractor",
            },
        },
        load_family="adapter",
        load_pretrained_map={
            # IMPORTANT: `ip-adapter_sdxl.bin` expects 1280-dim `image_embeds`
            # (see image_proj `proj.weight` with in_features=1280).
            #
            # `h94/IP-Adapter/models/image_encoder` (ViT-H) projects to 1024 and will crash:
            #   RuntimeError: mat1 and mat2 shapes cannot be multiplied (...x1024 and 1280x8192)
            #
            # The correct SDXL encoder lives under `sdxl_models/image_encoder` and has projection_dim=1280.
            "image_encoder": "h94/IP-Adapter",
            # Image processor is compatible; keep it stable and lightweight.
            "feature_extractor": "openai/clip-vit-large-patch14",
        },
        load_subfolder_map={
            "image_encoder": "sdxl_models/image_encoder",
        },
        load_variant_map={"image_encoder": "", "feature_extractor": ""},
    ),
    "sdxl.ipadapter_faceid": ComponentSpec(
        # FaceID checkpoints operate on pre-computed InsightFace embeddings.
        # No CLIP vision encoder / feature extractor is required inside the graph.
        block_types=["adapter/ip_adapter"],
        load_keys=[],
        constructor_map={},
    ),
    "adapter.controlnet_flux": ComponentSpec(
        block_types=["adapter/controlnet_flux"],
        load_keys=["controlnet"],
        constructor_map={"adapter/controlnet_flux": {"controlnet": "controlnet"}},
        load_family="flux",
    ),
    "adapter.ip_adapter": ComponentSpec(
        block_types=["adapter/ip_adapter"],
        load_keys=["image_encoder", "feature_extractor"],
        constructor_map={
            "adapter/ip_adapter": {
                "image_encoder": "image_encoder",
                "feature_extractor": "feature_extractor",
            },
        },
        # image_encoder from h94 (ViT-H 1024 dim); feature_extractor from openai (h94 lacks preprocessor)
        load_pretrained_map={
            "image_encoder": "h94/IP-Adapter",
            "feature_extractor": "openai/clip-vit-large-patch14",
        },
        load_subfolder_map={
            "image_encoder": "models/image_encoder",
        },
        load_variant_map={"image_encoder": "", "feature_extractor": ""},
    ),
    # Backward compat: flux.controlnet → same as adapter.controlnet_flux
    "flux.controlnet": ComponentSpec(
        block_types=["adapter/controlnet_flux"],
        load_keys=["controlnet"],
        constructor_map={"adapter/controlnet_flux": {"controlnet": "controlnet"}},
        load_family="flux",
    ),
}

# ── Unified registry ───────────────────────────────────────────────────

COMPONENT_REGISTRY: Dict[str, ComponentSpec] = {
    **_SD15_COMPONENTS,
    **_SDXL_COMPONENTS,
    **_FLUX_COMPONENTS,
    **_ADAPTER_COMPONENTS,
}


def resolve_component_type(component_type: str) -> ComponentSpec:
    """Look up a component type and return its spec.

    Raises ``KeyError`` if the type is unknown.
    """
    spec = COMPONENT_REGISTRY.get(component_type)
    if spec is None:
        raise KeyError(
            f"Unknown component type '{component_type}'. "
            f"Available: {sorted(COMPONENT_REGISTRY.keys())}"
        )
    return spec


def is_component_type(type_str: str) -> bool:
    """Return ``True`` when *type_str* uses the component-level namespace."""
    return "." in type_str and "/" not in type_str


def is_block_type(type_str: str) -> bool:
    """Return ``True`` when *type_str* uses the block-level namespace."""
    return "/" in type_str


def load_components_from_pretrained(
    load_keys: Sequence[str],
    pretrained: str,
    *,
    family: str,
    store: Optional[Any] = None,
    torch_dtype: Optional[Any] = None,
    variant: str = "",
    use_safetensors: Optional[bool] = None,
    pretrained_map: Optional[Dict[str, str]] = None,
    subfolder_map: Optional[Dict[str, str]] = None,
    variant_map: Optional[Dict[str, str]] = None,
    extra_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Load specific components. Each loads separately; cached in ModelStore.

    use_safetensors: If False, load .bin (PyTorch) weights instead of .safetensors.
        Some repos (e.g. Lykon/DreamShaper) have only .bin in unet/.
    pretrained_map: optional {load_key: repo_id} to load each key from a different repo.
    subfolder_map: optional {load_key: subfolder} to override subfolder per key.
    variant_map: optional {load_key: variant} to override variant per key (e.g. "" for CLIP).
    """
    if not load_keys:
        return {}

    from yggdrasill.integrations.diffusers.model_store import ModelStore

    ms = store or ModelStore.default()
    return ms.load_components_by_keys(
        family, list(load_keys), pretrained,
        variant=variant, torch_dtype=torch_dtype,
        use_safetensors=use_safetensors,
        pretrained_map=pretrained_map,
        subfolder_map=subfolder_map,
        variant_map=variant_map,
        extra_kwargs_map=extra_kwargs_map,
    )
