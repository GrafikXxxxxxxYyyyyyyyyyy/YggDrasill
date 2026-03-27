"""Align IP-Adapter image embeddings with diffusers UNet expectations.

Includes :func:`prepare_ip_adapter_image_embeds` to encode from graph IP-Adapter nodes
(no Diffusers pipeline). For Diffusers semantics, see ``prepare_ip_adapter_image_embeds`` in
``diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion`` and
``MultiIPAdapterImageProjection.forward`` in ``diffusers.models.embeddings``:
each tensor must be shaped ``(batch_size, num_images, embed_dim)`` (or 4D with a
sequence dim); with classifier-free guidance, batch is ``2 *`` the conditioning
batch, ordered ``[unconditional, conditional]``.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

try:
    import torch
except ImportError:  # pragma: no cover - exercised in no-diffusion environments
    torch = None  # type: ignore[assignment]

__all__ = [
    "format_ip_adapter_image_embeds",
    "pipeline_ip_adapter_embeds_cond_only",
    "prepare_ip_adapter_image_embeds",
    "raw_zero_ip_adapter_image_embeds_for_unet",
    "unet_requires_image_embeds_in_added_cond",
]


def _iter_ip_adapter_nodes(graph: Any) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for nid in sorted(getattr(graph, "node_ids", ())):
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        if getattr(node, "block_type", "") == "adapter/ip_adapter":
            out.append((nid, node))
    return out


def prepare_ip_adapter_image_embeds(
    graph: Any,
    ip_adapter_image: Union[Any, List[Any], Tuple[Any, ...]],
    *,
    node_id: Optional[str] = None,
    device: Optional[Any] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
) -> List[Any]:
    """Encode reference image(s) using **IP-Adapter node(s) already on the graph**.

    This mirrors the intent of Diffusers ``prepare_ip_adapter_image_embeds`` but runs the same
    CLIP preprocessor + ``image_encoder`` stack as :class:`~yggdrasill.integrations.diffusers.adapters.ip_adapter.IPAdapterNode`
    — no ``AutoPipeline*`` instance is required.

    Returns a **list** of tensors (one per IP-Adapter slot). **Pooled** IP-Adapter outputs are
    **conditional-only** (batch matches ``num_images_per_prompt``); the UNet doubles them with
    literal zero embeddings via :func:`format_ip_adapter_image_embeds` (same as Diffusers).

    **IP-Adapter Plus / Plus-Face** (hidden-state tensors) are returned **already CFG-packed** as
    ``[2, num_images, seq, dim]`` with dim 0 ordered ``[encoder(zeros_like(pixel_values)), encoder(image)]``
    — same as :meth:`~yggdrasill.integrations.diffusers.adapters.ip_adapter.IPAdapterNode.encode_ip_adapter_image`
    and Diffusers ``encode_image(..., output_hidden_states=True)``. Do **not** strip the first row:
    :func:`format_ip_adapter_image_embeds` must see ``shape[0] == 2`` so it does not replace the true
    uncond features with ``zeros_like`` (which destroys Plus / spatial masks).

    The *do_classifier_free_guidance* flag is accepted for API parity with Diffusers and is **not**
    used to change tensor shapes here.

    Args:
        graph: :class:`~yggdrasill.hypergraph.structure.Hypergraph` (or compatible) that contains
            at least one ``adapter/ip_adapter`` node with ``image_encoder`` + ``feature_extractor``.
        ip_adapter_image: One reference (URL, path, PIL, …) or a sequence with **one entry per**
            IP-Adapter node (after resolution). If there is a single image and multiple nodes, the
            same image is broadcast to every node.
        node_id: If set, only that node is used (must be ``adapter/ip_adapter``). Required when
            you want embeddings for one of several adapters without encoding for all.
        device: Optional device string or ``torch.device`` for the image encoders.
        num_images_per_prompt: ``repeat_interleave`` on dim 0, same semantics as Diffusers.
        do_classifier_free_guidance: Accepted for compatibility; does not alter outputs.

    Returns:
        ``List[torch.Tensor]`` — pass to ``run(..., ip_adapter_image_embeds={node_id: t})`` using
        the same ``node_id`` keys as in the graph, or a single tensor / list as supported by
        :meth:`~yggdrasill.integrations.diffusers.adapters.ip_adapter.IPAdapterNode.forward`.

    Raises:
        ValueError: If the graph has no IP-Adapter nodes, *node_id* is invalid, or image count
            does not match adapter count (after broadcast rules).
    """
    if torch is None:
        raise ImportError(
            "torch is required for IP-Adapter image embeddings. "
            "Install with: pip install 'yggdrasill[diffusion]'"
        )
    _ = do_classifier_free_guidance  # API parity with Diffusers; CFG handled in UNet.

    slots = _iter_ip_adapter_nodes(graph)
    if not slots:
        raise ValueError(
            "prepare_ip_adapter_image_embeds: graph has no adapter/ip_adapter node. "
            "Add one via DiffusionGraphBuilder.add_component(..., 'sdxl.ipadapter', ...) or equivalent."
        )

    if node_id is not None:
        node = graph.get_node(node_id) if hasattr(graph, "get_node") else None
        if node is None or getattr(node, "block_type", "") != "adapter/ip_adapter":
            raise ValueError(
                f"prepare_ip_adapter_image_embeds: node_id={node_id!r} is not an adapter/ip_adapter node."
            )
        target = [(node_id, node)]
    else:
        target = slots

    if isinstance(ip_adapter_image, (list, tuple)):
        imgs = list(ip_adapter_image)
    else:
        imgs = [ip_adapter_image]

    if len(imgs) == 1 and len(target) > 1:
        imgs = imgs * len(target)
    if len(imgs) != len(target):
        raise ValueError(
            f"prepare_ip_adapter_image_embeds: need one image per selected IP-Adapter node "
            f"({len(target)} nodes), got {len(imgs)} image(s). Pass a list or broadcast a single image."
        )

    out: List[Any] = []
    for (_, node), img in zip(target, imgs):
        emb = node.encode_ip_adapter_image(img, device=device)
        if torch.is_tensor(emb) and emb.ndim == 4 and emb.shape[0] == 2:
            if num_images_per_prompt != 1:
                raise ValueError(
                    "prepare_ip_adapter_image_embeds: num_images_per_prompt > 1 is not supported "
                    "for IP-Adapter Plus (hidden-state) outputs; use 1."
                )
        elif num_images_per_prompt > 1:
            r = int(num_images_per_prompt)
            if hasattr(emb, "repeat_interleave"):
                emb = emb.repeat_interleave(r, dim=0)
            elif torch.is_tensor(emb):
                emb = emb.repeat_interleave(r, dim=0)
            else:
                raise TypeError(
                    "encode_ip_adapter_image must return a tensor-like value with repeat_interleave "
                    "when num_images_per_prompt > 1"
                )
        out.append(emb)
    return out


def pipeline_ip_adapter_embeds_cond_only(
    ip_adapter_image_embeds: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Strip unconditional half from Diffusers **pooled** IP-Adapter ``prepare_*`` output.

    When ``do_classifier_free_guidance=True``, each tensor is ``cat([neg, pos], dim=0)``.
    For **pooled** embeddings, Yggdrasill can rebuild the neg half with ``zeros_like`` in
    :func:`format_ip_adapter_image_embeds`.

    Do **not** use this for **IP-Adapter Plus / Plus-Face** (4D hidden-state tensors): the negative
    half must stay the vision encoder on ``zeros_like(pixel_values)``, not literal zeros in embedding
    space. Pass those tensors through unchanged so ``shape[0] == 2`` is preserved.
    """
    out: List[torch.Tensor] = []
    for t in ip_adapter_image_embeds:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor entries, got {type(t)}")
        n = int(t.shape[0])
        if n % 2 != 0:
            raise ValueError(
                f"Expected even batch dim 0 for CFG-packed IP-Adapter embeds, got shape {tuple(t.shape)}"
            )
        out.append(t[n // 2 :].contiguous())
    return out


def unet_requires_image_embeds_in_added_cond(unet: Any) -> bool:
    """True when diffusers UNet expects ``added_cond_kwargs[\"image_embeds\"]`` (IP / Kandinsky-style)."""
    cfg = getattr(unet, "config", None)
    if cfg is None or getattr(unet, "encoder_hid_proj", None) is None:
        return False
    t = getattr(cfg, "encoder_hid_dim_type", None)
    return t in ("text_image_proj", "image_proj", "ip_image_proj")


def _first_linear_in_features(module: Any) -> int | None:
    import torch.nn as nn

    for m in module.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    return None


def _layer_expects_clip_token_embeds(layer: Any) -> bool:
    """True for diffusers :class:`IPAdapterPlusImageProjection` (ViT hidden states), not pooled / Face-ID."""
    import torch.nn as nn

    lat = getattr(layer, "latents", None)
    pi = getattr(layer, "proj_in", None)
    if not isinstance(lat, nn.Parameter) or not isinstance(pi, nn.Linear):
        return False
    return int(pi.out_features) == int(lat.shape[-1])


def raw_zero_ip_adapter_image_embeds_for_unet(
    unet: Any,
    cond_batch_size: int,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> List[torch.Tensor]:
    """Zeros per IP-Adapter slot for :func:`format_ip_adapter_image_embeds`.

    **Pooled** IP-Adapter: each tensor is ``[cond_batch, embed_dim]``.

    **IP-Adapter Plus** (CLIP/ViT token embeddings): each tensor is
    ``[cond_batch, 1, 1, embed_dim]`` so after CFG packing and
    ``MultiIPAdapterImageProjection`` reshape, ``proj_in`` sees 3D activations
    (avoids 2D vs 3D ``torch.cat`` in ``IPAdapterPlusImageProjectionBlock``).

    When *dtype* is ``None``, uses the first parameter dtype of *unet* so placeholders match
    fp16/bf16 weights (avoids Float vs Half matmul).
    """
    proj = getattr(unet, "encoder_hid_proj", None)
    if proj is None:
        return []
    zdt = dtype
    if zdt is None:
        try:
            zdt = next(unet.parameters()).dtype
        except (StopIteration, TypeError):
            zdt = torch.float32

    layers = getattr(proj, "image_projection_layers", None)
    if layers is not None and len(layers) > 0:
        out: List[torch.Tensor] = []
        for layer in layers:
            if _layer_expects_clip_token_embeds(layer):
                emb_in = int(layer.proj_in.in_features)
                out.append(
                    torch.zeros(
                        cond_batch_size, 1, 1, emb_in, device=device, dtype=zdt
                    )
                )
            else:
                dim = _first_linear_in_features(layer) or 1024
                out.append(
                    torch.zeros(cond_batch_size, dim, device=device, dtype=zdt)
                )
        return out

    dim = _first_linear_in_features(proj) or 1024
    return [torch.zeros(cond_batch_size, dim, device=device, dtype=zdt)]


def format_ip_adapter_image_embeds(
    image_embeds: Union[torch.Tensor, List[torch.Tensor]],
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    do_classifier_free_guidance: bool,
) -> List[torch.Tensor]:
    """Return ``image_embeds`` as a list (one entry per IP-Adapter projection layer).

    If *dtype* is ``None`` (default), tensors are moved with ``.to(device)`` only.
    SDXL/SD1.5 :class:`~yggdrasill.integrations.diffusers.sdxl.unet.SDXLUNetNode` passes
    the UNet parameter *dtype* so IP-Adapter Plus ``proj_in`` (fp16 weights) does not see
    float32 activations.
    """

    def _one(single: torch.Tensor) -> torch.Tensor:
        e = single.to(device=device)
        if dtype is not None:
            e = e.to(dtype=dtype)
        # [2, N, seq, dim]: already packed [uncond, cond] from :meth:`IPAdapterNode.encode_ip_adapter_image`
        # when ``ip_adapter_use_hidden_states`` (matches Diffusers SDXL ``prepare_ip_adapter_image_embeds``).
        if e.ndim == 4 and e.shape[0] == 2:
            if not do_classifier_free_guidance:
                e = e[1:2]
            return e
        if e.ndim == 2:
            # Encoder returns [num_images, embed_dim] for N references in one IP-Adapter slot.
            # Diffusers does single_image_embeds[None, :] → [1, N, D] before CFG (see SDXL
            # prepare_ip_adapter_image_embeds). Using unsqueeze(1) wrongly yields [N, 1, D]
            # and breaks IPAdapterPlusImageProjection (2D vs 3D in the resampler block).
            e = e.unsqueeze(0)
        elif e.ndim not in (3, 4):
            raise ValueError(
                f"IP-Adapter image_embeds must be 2D–4D, got shape {tuple(e.shape)}"
            )
        if do_classifier_free_guidance:
            e = torch.cat([torch.zeros_like(e), e], dim=0)
        return e

    if isinstance(image_embeds, list):
        return [_one(x) for x in image_embeds]
    return [_one(image_embeds)]
