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

import torch

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

    Returns a **list** of tensors (one per IP-Adapter slot), each in **conditional** form only
    (batch matches ``num_images_per_prompt``). Classifier-free guidance doubling is applied later
    inside the UNet via :func:`format_ip_adapter_image_embeds`; the *do_classifier_free_guidance*
    flag is accepted for API parity with Diffusers and is **not** used to change tensor shapes here.

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
        if num_images_per_prompt > 1:
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
    """Strip unconditional half from Diffusers ``prepare_ip_adapter_image_embeds`` output.

    When ``do_classifier_free_guidance=True``, each tensor is ``cat([neg, pos], dim=0)``.
    Yggdrasill applies CFG doubling in :func:`format_ip_adapter_image_embeds`, so feed **cond-only**
    tensors with batch size matching the **conditional** latent batch (typically the encoder batch).
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


def raw_zero_ip_adapter_image_embeds_for_unet(
    unet: Any,
    cond_batch_size: int,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> List[torch.Tensor]:
    """Raw (B, D) zeros per IP-Adapter slot; pass through :func:`format_ip_adapter_image_embeds`.

    Zeros use **float32** by default so they match typical CLIP vision ``image_embeds``;
    diffusers ``prepare_ip_adapter_image_embeds`` ends with ``.to(device)`` and does not
    cast IP tensors to the UNet weight dtype.
    """
    proj = getattr(unet, "encoder_hid_proj", None)
    if proj is None:
        return []
    layers = getattr(proj, "image_projection_layers", None)
    if layers is not None and len(layers) > 0:
        n_adapters = len(layers)
        dim = _first_linear_in_features(layers[0]) or 1024
    else:
        n_adapters = 1
        dim = _first_linear_in_features(proj) or 1024
    zdt = dtype if dtype is not None else torch.float32
    return [
        torch.zeros(cond_batch_size, dim, device=device, dtype=zdt)
        for _ in range(n_adapters)
    ]


def format_ip_adapter_image_embeds(
    image_embeds: Union[torch.Tensor, List[torch.Tensor]],
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    do_classifier_free_guidance: bool,
) -> List[torch.Tensor]:
    """Return ``image_embeds`` as a list (one entry per IP-Adapter projection layer).

    If *dtype* is ``None`` (default), tensors are moved with ``.to(device)`` only — same
    idea as diffusers ``prepare_ip_adapter_image_embeds`` (vision outputs often stay
    float32 while the UNet runs in fp16). Forcing fp16 here can destabilize IP-Adapter.
    """

    def _one(single: torch.Tensor) -> torch.Tensor:
        e = single.to(device=device)
        if dtype is not None:
            e = e.to(dtype=dtype)
        if e.ndim == 2:
            e = e.unsqueeze(1)
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
