"""IP-Adapter weight loader: inject IP-Adapter weights into UNet (Injector).

Loads state dict from HF (either repo root or legacy `models/` subfolder)
and calls unet._load_ip_adapter_weights() per diffusers IPAdapterMixin.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _load_ip_adapter_state_dict(
    pretrained: str,
    *,
    subfolder: Optional[str] = None,
    weight_name: str = "ip-adapter_sd15.bin",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Load IP-Adapter state dict from HF repo.

    If *subfolder* is ``None`` (or falsy), we first try to resolve the file in repo root.
    On 404, we retry with ``subfolder="models"`` for backward compatibility.
    """
    from huggingface_hub import hf_hub_download

    def _try_download(attempt_subfolder: Optional[str]) -> str:
        return hf_hub_download(
            repo_id=pretrained,
            filename=weight_name,
            subfolder=attempt_subfolder if attempt_subfolder else None,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision or "main",
        )

    # Prefer repo root when subfolder isn't provided.
    try:
        model_file = _try_download(subfolder)
    except Exception as exc:
        # Only retry on a likely "file not found" situation.
        msg = str(exc).lower()
        if ("404" in msg) or ("not found" in msg):
            model_file = _try_download("models")
        else:
            raise

    if weight_name.endswith(".safetensors"):
        from safetensors import safe_open

        state_dict: Dict[str, Any] = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    else:
        import torch

        raw = torch.load(model_file, map_location="cpu", weights_only=True)
        keys = list(raw.keys())
        if "image_proj" in keys and "ip_adapter" in keys:
            state_dict = raw
        else:
            # Flat format: convert to nested
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            for k, v in raw.items():
                if k.startswith("image_proj."):
                    state_dict["image_proj"][k.replace("image_proj.", "")] = v
                elif k.startswith("ip_adapter."):
                    state_dict["ip_adapter"][k.replace("ip_adapter.", "")] = v

    keys = list(state_dict.keys())
    if "image_proj" not in keys and "ip_adapter" not in keys:
        raise ValueError(
            "Required keys (`image_proj` and `ip_adapter`) missing from IP-Adapter state dict."
        )
    return state_dict


def _set_ip_adapter_scale_on_unet(unet: Any, scale: Any) -> None:
    """Set IP-Adapter scale(s) on UNet (aligned with diffusers ``IPAdapterMixin.set_ip_adapter_scale``).

    *scale* may be a float; a list of per-adapter values (float, nested list, or InstantStyle dict); nested
    lists such as ``[[0.7, 0.7]]`` for **one** loaded adapter with **two** reference images under spatial
    masks (see HF IP-Adapter masking docs); or an InstantStyle dict
    ``{"down": {"block_2": [...]}, "up": {"block_0": [...]}}`` which is expanded via
    ``diffusers.loaders.unet_loader_utils._maybe_expand_lora_scales`` like the pipeline API.
    """
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    unet = resolve_if_lazy(unet)
    if unet is None:
        return

    try:
        from diffusers.loaders.unet_loader_utils import _maybe_expand_lora_scales
        from diffusers.models.attention_processor import (
            IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_0,
            IPAdapterXFormersAttnProcessor,
        )
    except ImportError:
        return

    if not isinstance(scale, list):
        scale = [scale]
    scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)

    for attn_name, attn_processor in getattr(unet, "attn_processors", {}).items():
        if not isinstance(
            attn_processor,
            (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor),
        ):
            continue
        # Keep behavior aligned with diffusers: either broadcast a single config across all slots
        # or fail fast on length mismatch. Silent "continue" would effectively disable IP-Adapter.
        sc = list(scale_configs)
        n_proc = len(attn_processor.scale)
        if len(sc) != n_proc:
            if len(sc) == 1 and n_proc >= 1:
                sc = sc * n_proc
            else:
                raise ValueError(
                    f"IP-Adapter scale mismatch: got {len(sc)} scale configs, but UNet attention "
                    f"processor {attn_name!r} has {n_proc} scale slots."
                )

        for i, scale_config in enumerate(sc):
            if isinstance(scale_config, dict):
                for k, s in scale_config.items():
                    if attn_name.startswith(k):
                        attn_processor.scale[i] = s
            else:
                attn_processor.scale[i] = scale_config


def load_ip_adapter_into_unet(
    unet: Any,
    pretrained: str,
    *,
    subfolder: str = "models",
    weight_name: str = "ip-adapter_sd15.bin",
    low_cpu_mem_usage: bool = True,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    ip_adapter_scale: Optional[float] = None,
) -> None:
    """Load IP-Adapter weights into UNet and optionally set scale.

    Calls unet._load_ip_adapter_weights([state_dict]) per diffusers.
    If ip_adapter_scale is provided and unet has set_ip_adapter_scale, calls it.

    Args:
        unet: UNet2DConditionModel (or FluxTransformer2DModel, etc.) from diffusers.
        pretrained: HF repo id, e.g. "h94/IP-Adapter".
        subfolder: Subfolder in repo, default "models".
        weight_name: Weight file, e.g. "ip-adapter_sd15.bin" or "ip-adapter_sd15.safetensors".
        low_cpu_mem_usage: Passed to _load_ip_adapter_weights.
        ip_adapter_scale: If set, call set_ip_adapter_scale on unet (diffusers pipeline API).
            The UNet may expose this via its attn processors; we call it if present.
    """
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    unet = resolve_if_lazy(unet)
    if unet is None:
        raise ValueError("UNet cannot be None for IP-Adapter weight loading.")

    state_dict = _load_ip_adapter_state_dict(
        pretrained,
        subfolder=subfolder,
        weight_name=weight_name,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
    )

    if not hasattr(unet, "_load_ip_adapter_weights"):
        raise ValueError(
            f"UNet type {type(unet).__name__} does not support IP-Adapter "
            "(missing _load_ip_adapter_weights). Use UNet2DConditionModel or compatible model."
        )

    unet._load_ip_adapter_weights([state_dict], low_cpu_mem_usage=low_cpu_mem_usage)

    if ip_adapter_scale is not None:
        _set_ip_adapter_scale_on_unet(unet, ip_adapter_scale)


def reload_ip_adapter_weights_on_unet(
    unet: Any,
    state_dicts: Any,
    *,
    low_cpu_mem_usage: bool = True,
) -> None:
    """Apply several IP-Adapter checkpoints in one call (same as diffusers ``load_ip_adapter`` with a list).

    Each ``state_dict`` must match :func:`_load_ip_adapter_state_dict` output. Calling
    ``_load_ip_adapter_weights`` replaces all IP-Adapter processors; pass **all** loaded adapters
    every time the list grows (see graph metadata accumulation in the diffusion builder).
    """
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    unet = resolve_if_lazy(unet)
    if unet is None:
        return
    sds = list(state_dicts) if isinstance(state_dicts, (list, tuple)) else [state_dicts]
    if not sds or not hasattr(unet, "_load_ip_adapter_weights"):
        return
    unet._load_ip_adapter_weights(sds, low_cpu_mem_usage=low_cpu_mem_usage)


def infer_ip_adapter_plus_token_embed_dim(unet: Any) -> Optional[int]:
    """Return per-token width expected by the first IP-Adapter Plus ``proj_in`` (``in_features``).

    Standard (non-Plus) IP-Adapter uses :class:`~diffusers.models.embeddings.ImageProjection`
    without ``proj_in`` — returns ``None``. Used to align CLIP vision ``hidden_states[-2]`` with
    loaded weights (e.g. h94 SDXL ViT width 1664 → contrastive 1280 via ``visual_projection``).
    """
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    unet = resolve_if_lazy(unet)
    if unet is None:
        return None
    wrap = getattr(unet, "encoder_hid_proj", None)
    if wrap is None:
        return None
    layers = getattr(wrap, "image_projection_layers", None)
    if not layers or len(layers) == 0:
        return None
    pin = getattr(layers[0], "proj_in", None)
    if pin is None or not hasattr(pin, "in_features"):
        return None
    n = int(getattr(pin, "in_features", 0) or 0)
    return n if n > 0 else None
