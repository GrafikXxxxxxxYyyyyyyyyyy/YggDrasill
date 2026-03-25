"""Diffusion run wrapper: prepares diffusion-specific kwargs and wraps output in DiffusionOutput."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.builder import _META_IP_ADAPTER_ORDER
from yggdrasill.integrations.diffusers.output import DiffusionOutput


def _iter_controlnet_node_ids(graph: Any) -> List[str]:
    out: List[str] = []
    for nid in sorted(getattr(graph, "node_ids", ()) or ()):
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is not None and getattr(node, "block_type", "") == "adapter/controlnet":
            out.append(nid)
    return out


def _inject_guess_mode(graph: Any, guess_mode: Any) -> None:
    """Inject guess_mode into ControlNet nodes.

    Accepted forms:
    - bool/int/float: applied to every ControlNet node
    - {node_id: bool}: per-node
    """
    if guess_mode is None:
        return
    if isinstance(guess_mode, dict):
        _inject_node_config(graph, guess_mode, "guess_mode")
        return
    flag = bool(guess_mode)
    nodes = getattr(graph, "_nodes", None) or {}
    for nid in _iter_controlnet_node_ids(graph):
        node = nodes.get(nid) if isinstance(nodes, dict) else None
        if node is None:
            node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        if not hasattr(node, "_config"):
            node._config = {}
        node._config["guess_mode"] = flag


def _inject_control_guidance_window(graph: Any, *, start: Any, end: Any) -> None:
    """Inject control guidance window (diffusers control_guidance_start/end) into ControlNet nodes.

    Accepted forms:
    - scalar float/int: applied to every ControlNet node
    - {node_id: float}: per node
    """
    if start is not None:
        if isinstance(start, dict):
            _inject_node_config(graph, start, "control_guidance_start")
        else:
            _inject_node_config(
                graph,
                {nid: float(start) for nid in _iter_controlnet_node_ids(graph)},
                "control_guidance_start",
            )
    if end is not None:
        if isinstance(end, dict):
            _inject_node_config(graph, end, "control_guidance_end")
        else:
            _inject_node_config(
                graph,
                {nid: float(end) for nid in _iter_controlnet_node_ids(graph)},
                "control_guidance_end",
            )


def run(
    graph: Any,
    inputs: Optional[Dict[str, Any]] = None,
    *,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[Any] = None,
    wrap_output: bool = True,
    **kwargs: Any,
) -> DiffusionOutput | Dict[str, Any]:
    """Run a diffusion graph with optional DiffusionOutput wrapping.

    Args:
        graph: Hypergraph or Workflow to run.
        inputs: Input dict (e.g. prompt, negative_prompt). ``ip_adapter_image`` /
            ``controlnet_image`` / mask keys may be dicts ``{node_id: ...}`` here as well as
            in ``**kwargs``; they are normalized to ``{node_id}:{port}`` before the run.
        num_inference_steps: Override for denoising steps.
        seed: Random seed for latent init.
        device: Target device.
        wrap_output: If True, return DiffusionOutput; otherwise raw dict.
        **kwargs: Passed through to graph.run(). ``controlnet_image`` /
            ``ip_adapter_image`` / ``ip_adapter_image_embeds`` /
            ``ip_adapter_mask_images`` (masks for :class:`IPAdapterMaskPrepNode`) /
            ``ip_adapter_masks`` (prepacked tensor(s) for UNet ``cross_attention_kwargs``)
            may be dicts ``{node_id: ...}``;
            omit a node id (or pass
            ``None``) to disable that ControlNet / IP-Adapter for this run while leaving
            it on the graph. For IP-Adapter, supplying only ``ip_adapter_image_embeds`` counts
            as active (encoder can stay unloaded). ControlNet nodes without an image are skipped
            so stale residuals cannot corrupt the UNet; IP-Adapter nodes still run and emit
            inactive embeddings when unused. IP-Adapter strengths default to 0 for slots with
            no reference image on this run (so loaded IP weights do not keep diffusers' default
            scale 1.0).

    Returns:
        DiffusionOutput when wrap_output=True, else raw executor dict.
    """
    run_kw = dict(kwargs)
    if num_inference_steps is not None:
        run_kw["num_inference_steps"] = num_inference_steps
    if seed is not None:
        run_kw["seed"] = seed
    if device is not None:
        run_kw["device"] = device

    merged = dict(inputs or {})
    if "image" in merged and C.PORT_INIT_IMAGE not in merged:
        merged[C.PORT_INIT_IMAGE] = merged.pop("image")
    normalize_merged_adapter_inputs(merged, graph)

    controlnet_image = run_kw.pop("controlnet_image", None)
    if isinstance(controlnet_image, dict):
        for nid, img in controlnet_image.items():
            if img is not None:
                merged[f"{nid}:{C.PORT_CONTROL_IMAGE}"] = img
    elif controlnet_image is not None:
        _assign_to_single_exposed(merged, graph, C.PORT_CONTROL_IMAGE, controlnet_image)

    ip_adapter_image = run_kw.pop("ip_adapter_image", None)
    merge_ip_adapter_image_kwarg(merged, graph, ip_adapter_image)

    ip_adapter_image_embeds = run_kw.pop("ip_adapter_image_embeds", None)
    if isinstance(ip_adapter_image_embeds, dict):
        for nid, emb in ip_adapter_image_embeds.items():
            if emb is not None:
                merged[f"{nid}:{C.PORT_IP_ADAPTER_IMAGE_EMBEDS}"] = emb
    elif ip_adapter_image_embeds is not None:
        _assign_to_single_exposed(
            merged, graph, C.PORT_IP_ADAPTER_IMAGE_EMBEDS, ip_adapter_image_embeds,
        )

    ip_adapter_mask_images = run_kw.pop("ip_adapter_mask_images", None)
    if isinstance(ip_adapter_mask_images, dict):
        for nid, imgs in ip_adapter_mask_images.items():
            if imgs is not None:
                merged[f"{nid}:{C.PORT_IP_ADAPTER_MASK_IMAGES}"] = imgs
    elif ip_adapter_mask_images is not None:
        _assign_to_single_exposed(
            merged, graph, C.PORT_IP_ADAPTER_MASK_IMAGES, ip_adapter_mask_images,
        )

    # Do not pop or apply controlnet_conditioning_scale / ip_adapter_conditioning_scale here.
    # Hypergraph.run is patched to merge those kwargs and call _inject_*; if we popped them
    # above, the patch would see None and re-apply IP defaults (scale 1.0), wiping user scale.

    if "image" in run_kw:
        _img2img = run_kw.pop("image")
        run_kw.setdefault(C.PORT_INIT_IMAGE, _img2img)

    ip_masks_kw = run_kw.pop("ip_adapter_masks", None)
    if ip_masks_kw is not None:
        _route_ip_adapter_masks(
            merged, graph, ip_masks_kw,
            pin_data=run_kw.setdefault("pin_data", {}),
        )

    _prepare_diffusion_run(graph, run_kw, merged_inputs=merged)
    raw = graph.run(merged, **run_kw)

    if wrap_output:
        # Hypergraph.run is patched on diffusers import to return DiffusionOutput when
        # image ports are present; avoid double-wrapping.
        if isinstance(raw, DiffusionOutput):
            return raw
        return DiffusionOutput.from_executor_output(raw)
    if isinstance(raw, DiffusionOutput):
        return raw.raw
    return raw


def _inject_node_config(graph: Any, node_values: Dict[str, Any], config_key: str) -> None:
    """Inject per-node config values: {node_id: value} -> node._config[config_key] = value."""
    nodes = getattr(graph, "_nodes", None) or {}
    for nid, val in node_values.items():
        if nid in nodes:
            node = nodes[nid]
            if not hasattr(node, "_config"):
                node._config = {}
            node._config[config_key] = val


def _graph_has_ip_adapter_node(graph: Any) -> bool:
    for nid in getattr(graph, "node_ids", ()) or ():
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is not None and getattr(node, "block_type", "") == "adapter/ip_adapter":
            return True
    return False


def _enforce_ip_adapter_multi_ref_with_masks(graph: Any, merged: Dict[str, Any]) -> None:
    """Fail fast when several reference images are passed without spatial masks.

    Without ``ip_adapter_mask_images`` / ``ip_adapter_masks``, Diffusers' attention processors
    concatenate all reference tokens into one IP sequence and apply a **scalar** scale. That path
    is unstable for two-plus Plus-Face references (garbage latents, wrong subject count). The
    supported setups are: **one** reference, or **multiple references with one spatial mask each**
    (same order as images).
    """
    if not _graph_has_ip_adapter_node(graph):
        return
    if _merged_provides_ip_adapter_spatial_masks(merged):
        return
    img_suffix = f":{C.PORT_IP_ADAPTER_IMAGE}"
    for k, v in merged.items():
        if v is None or not isinstance(k, str):
            continue
        if not (k.endswith(img_suffix) or k == C.PORT_IP_ADAPTER_IMAGE):
            continue
        # Two references on one adapter without spatial masks is a common foot-gun (e.g. two faces).
        # ``[style_folder_list, face]`` is length 2 but the first slot is a batch of style images.
        if isinstance(v, (list, tuple)) and len(v) == 2:
            a0 = v[0]
            if isinstance(a0, (list, tuple)) and len(a0) >= 3:
                continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            # Match diffusers behavior: do not hard-fail; warn that results can be unstable.
            # (In diffusers, without masks both references are merged into one global IP sequence.)
            import warnings

            warnings.warn(
                "IP-Adapter: two reference images on one adapter are usually unstable without spatial masks. "
                "Pass `ip_adapter_mask_images` (one mask per image in the same order) or prepacked "
                "`ip_adapter_masks`. Without masks, both references are merged into one global IP sequence. "
                f"(Got {len(v)} images for {k!r}.)",
                UserWarning,
            )


def _merged_provides_ip_adapter_spatial_masks(merged: Dict[str, Any]) -> bool:
    """True when the run supplies IP-Adapter mask images or prepacked ``ip_adapter_masks`` tensors."""
    if not merged:
        return False
    if merged.get(C.PORT_IP_ADAPTER_MASK_IMAGES) is not None:
        return True
    if merged.get(C.PORT_IP_ADAPTER_MASKS) is not None:
        return True
    suf_im = f":{C.PORT_IP_ADAPTER_MASK_IMAGES}"
    suf_t = f":{C.PORT_IP_ADAPTER_MASKS}"
    for k, v in merged.items():
        if v is None:
            continue
        if isinstance(k, str) and (k.endswith(suf_im) or k.endswith(suf_t)):
            return True
    return False


_LAYOUT_IP_ADAPTER_SCALE_KEYS = frozenset(("down", "up", "mid"))


def _is_unet_ip_adapter_layout_scale_dict(d: Any) -> bool:
    """True when *d* is a diffusers / InstantStyle layer map (only ``down`` / ``up`` / ``mid`` keys)."""
    if not isinstance(d, dict) or not d:
        return False
    return frozenset(d) <= _LAYOUT_IP_ADAPTER_SCALE_KEYS


def _coerce_ip_adapter_scale_value(v: Any, *, mask_spatial: bool) -> Any:
    """Normalize a per-adapter strength: float, masked multi-ref list, or InstantStyle layout dict."""
    if isinstance(v, dict) and _is_unet_ip_adapter_layout_scale_dict(v):
        return v
    if isinstance(v, (list, tuple)):
        vf = [float(x) for x in v]
        if len(vf) > 1:
            return vf if mask_spatial else float(sum(vf) / len(vf))
        return vf[0]
    return float(v)


def _inject_ip_adapter_scale(
    graph: Any, scale_map: Union[Dict[str, Any], List[Any]], merged_inputs: Dict[str, Any],
) -> None:
    """Apply per–IP-Adapter scales on the UNet; adapters with no image get strength 0.

    *scale_map* forms:

    * **Per node:** ``{adapter_node_id: float | list | layout_dict, "default": float}``.
    * **InstantStyle (global):** ``{"down": {...}, "up": {...}}`` with *only* those top-level keys
      (same as diffusers ``pipeline.set_ip_adapter_scale``); applied to every adapter slot that
      has conditioning on this run.
    * **Per adapter (ordered):** a list of configs, one per IP-Adapter node in sorted node-id order
      (extras ignored; if shorter, the last entry is reused), matching diffusers' list form for
      multiple loaded IP-Adapters.
    """
    if not scale_map:
        return
    try:
        from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import (
            _set_ip_adapter_scale_on_unet,
        )
    except ImportError:
        return

    ip_nodes = iter_ip_adapter_node_ids_for_routing(graph)
    if not ip_nodes:
        return

    get_spec = getattr(graph, "get_input_spec", None)
    input_spec: List[Dict[str, Any]] = list(get_spec() or []) if callable(get_spec) else []
    mask_spatial = _merged_provides_ip_adapter_spatial_masks(merged_inputs)
    scales: List[Any] = []

    if isinstance(scale_map, list):
        cfgs = scale_map
        for i, nid in enumerate(ip_nodes):
            active = _ip_adapter_node_has_conditioning(
                merged_inputs, nid, input_spec,
            )
            if not active:
                scales.append(0.0)
                continue
            v = cfgs[i] if i < len(cfgs) else (cfgs[-1] if cfgs else 1.0)
            scales.append(_coerce_ip_adapter_scale_value(v, mask_spatial=mask_spatial))
    elif isinstance(scale_map, dict) and _is_unet_ip_adapter_layout_scale_dict(scale_map):
        for nid in ip_nodes:
            active = _ip_adapter_node_has_conditioning(
                merged_inputs, nid, input_spec,
            )
            if active:
                scales.append(scale_map)
            else:
                scales.append(0.0)
    elif isinstance(scale_map, dict):
        default_scale = float(scale_map.get("default", 1.0))
        for nid in ip_nodes:
            active = _ip_adapter_node_has_conditioning(
                merged_inputs, nid, input_spec,
            )
            if active:
                v = scale_map.get(nid, default_scale)
                scales.append(_coerce_ip_adapter_scale_value(v, mask_spatial=mask_spatial))
            else:
                scales.append(0.0)
    else:
        return
    # Diffusers ``set_ip_adapter_scale([[0.7, 0.7]])``: outer list = per loaded IP-Adapter *group* on the UNet
    # (matches ``attn_processor.scale`` length); inner list = per reference image when using spatial masks
    # (see SDXL IP-Adapter masking docs). Passing a bare ``[0.7, 0.7]`` is treated as *two* adapter groups
    # inside ``_maybe_expand_lora_scales`` → length mismatch → scales never applied (defaults stay 1.0).
    if len(scales) == 1:
        s0 = scales[0]
        scale_payload: Any = [s0] if isinstance(s0, list) else s0
    else:
        scale_payload = scales

    for uid in getattr(graph, "node_ids", ()) or ():
        node = graph.get_node(uid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        bt = getattr(node, "block_type", "") or ""
        if bt.endswith("/unet") or bt.endswith("/transformer"):
            unet = getattr(node, "_unet", None)
            if unet is not None:
                _set_ip_adapter_scale_on_unet(unet, scale_payload)
            break


def _route_ip_adapter_masks(
    merged: Dict[str, Any],
    graph: Any,
    masks: Any,
    *,
    pin_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Feed UNet ``ip_adapter_masks``: exposed graph inputs, ``pin_data``, or per-node dict."""
    if masks is None:
        return
    get_spec = getattr(graph, "get_input_spec", None)
    if not callable(get_spec):
        merged[C.PORT_IP_ADAPTER_MASKS] = masks
        return
    spec = list(get_spec() or [])
    exposed = [e for e in spec if e.get("port_name") == C.PORT_IP_ADAPTER_MASKS]
    if isinstance(masks, dict):
        pd = pin_data if pin_data is not None else {}
        for nid, val in masks.items():
            if val is not None:
                pd.setdefault(nid, {})[C.PORT_IP_ADAPTER_MASKS] = val
        return
    if len(exposed) == 1:
        nid = exposed[0].get("node_id")
        if nid:
            merged[f"{nid}:{C.PORT_IP_ADAPTER_MASKS}"] = masks
            return
    if not exposed and pin_data is not None:
        for nid in sorted(getattr(graph, "node_ids", ()) or ()):
            node = graph.get_node(nid) if hasattr(graph, "get_node") else None
            bt = getattr(node, "block_type", "") or ""
            if bt.endswith("/unet"):
                pin_data.setdefault(nid, {})[C.PORT_IP_ADAPTER_MASKS] = masks
                return
    merged[C.PORT_IP_ADAPTER_MASKS] = masks


def _assign_to_single_exposed(
    merged: Dict[str, Any], graph: Any, port_name: str, value: Any
) -> None:
    """When a single value is passed, find the one exposed node with that port and assign."""
    get_spec = getattr(graph, "get_input_spec", None)
    if get_spec is None:
        merged[port_name] = value
        return
    spec = get_spec()
    matches = [e for e in spec if e.get("port_name") == port_name]
    if len(matches) == 1:
        nid = matches[0].get("node_id")
        if nid:
            merged[f"{nid}:{port_name}"] = value
            return
    # Several nodes expose the same logical port name (e.g. two IP-Adapters → ``ip_adapter_image``).
    # A list is then diffusers-style ``[style_inputs, face_input]``, not a bare multi-port payload.
    if (
        port_name == C.PORT_IP_ADAPTER_IMAGE
        and len(matches) > 1
        and isinstance(value, (list, tuple))
    ):
        merge_ip_adapter_image_kwarg(merged, graph, value)
        return
    merged[port_name] = value


def iter_sorted_ip_adapter_node_ids(graph: Any) -> List[str]:
    """Lexicographic order of ``adapter/ip_adapter`` nodes (fallback only)."""
    out: List[str] = []
    for nid in sorted(getattr(graph, "node_ids", ()) or ()):
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is not None and getattr(node, "block_type", "") == "adapter/ip_adapter":
            out.append(nid)
    return out


def iter_ip_adapter_node_ids_for_routing(graph: Any) -> List[str]:
    """Order aligned with ``encoder_hid_proj`` layers (weight load order), then sorted ids."""
    meta = getattr(graph, "metadata", None) or {}
    order = list(meta.get(_META_IP_ADAPTER_ORDER) or [])
    alive = set(getattr(graph, "node_ids", ()) or ())
    if order:
        routed = [n for n in order if n in alive]
        sorted_all = iter_sorted_ip_adapter_node_ids(graph)
        if len(routed) == len(sorted_all):
            return routed
    return iter_sorted_ip_adapter_node_ids(graph)


def merge_ip_adapter_image_kwarg(merged: Dict[str, Any], graph: Any, value: Any) -> None:
    """Map ``ip_adapter_image`` like diffusers: ``dict`` or list aligned with IP-Adapter nodes."""
    if value is None:
        return
    if isinstance(value, dict):
        ip_ids = set(iter_ip_adapter_node_ids_for_routing(graph))
        if not ip_ids and value:
            raise ValueError(
                "ip_adapter_image was passed as a dict but this graph has no adapter/ip_adapter nodes."
            )
        if ip_ids:
            bad = [k for k in value if k not in ip_ids]
            if bad:
                raise ValueError(
                    f"ip_adapter_image dict keys {bad!r} are not IP-Adapter node ids on this graph. "
                    f"Use ids from {sorted(ip_ids)} (same order as add_component loads weights; "
                    f"see graph.metadata['ip_adapter_weight_node_ids'])."
                )
        for nid, item in value.items():
            if item is not None:
                merged[f"{nid}:{C.PORT_IP_ADAPTER_IMAGE}"] = item
        return
    ip_nodes = iter_ip_adapter_node_ids_for_routing(graph)
    if isinstance(value, (list, tuple)) and len(ip_nodes) > 1:
        if len(value) == len(ip_nodes):
            for nid, item in zip(ip_nodes, value):
                if item is not None:
                    merged[f"{nid}:{C.PORT_IP_ADAPTER_IMAGE}"] = item
            return
        if len(value) == 1:
            _assign_to_single_exposed(
                merged, graph, C.PORT_IP_ADAPTER_IMAGE, value[0],
            )
            return
        raise ValueError(
            f"ip_adapter_image: graph has {len(ip_nodes)} IP-Adapter node(s); "
            f"pass a list of length {len(ip_nodes)} (or 1 to broadcast), got {len(value)}."
        )
    _assign_to_single_exposed(merged, graph, C.PORT_IP_ADAPTER_IMAGE, value)


def _pop_expand_node_scoped_port(
    merged: Dict[str, Any],
    graph: Any,
    port_name: str,
) -> None:
    """If *merged* has a bare *port_name* entry, move it to ``{node_id}:{port_name}`` keys.

    Dict values are treated as ``{node_id: payload}`` (same as kwargs). Scalar values use
    :func:`_assign_to_single_exposed`. The bare key is always removed so EdgeBuffers never
    receives a raw ``dict`` on the IP / ControlNet image ports.
    """
    if port_name not in merged:
        return
    val = merged.pop(port_name)
    if val is None:
        return
    if isinstance(val, dict):
        for nid, item in val.items():
            if item is not None:
                merged[f"{nid}:{port_name}"] = item
    elif port_name == C.PORT_IP_ADAPTER_IMAGE and isinstance(val, (list, tuple)):
        merge_ip_adapter_image_kwarg(merged, graph, val)
    else:
        _assign_to_single_exposed(merged, graph, port_name, val)


def normalize_merged_adapter_inputs(merged: Dict[str, Any], graph: Any) -> None:
    """Normalize adapter payloads passed on the positional ``inputs`` dict (not only kwargs).

    Without this, ``graph.run({"ip_adapter_image": {"IPAdapter": img}, ...})`` leaves a dict on
    the generic port name; :meth:`~yggdrasill.engine.buffers.EdgeBuffers.init_from_inputs` then
    seeds the IP node with that dict and encoding fails or yields no conditioning.
    """
    _pop_expand_node_scoped_port(merged, graph, C.PORT_CONTROL_IMAGE)
    _pop_expand_node_scoped_port(merged, graph, C.PORT_IP_ADAPTER_IMAGE)
    _pop_expand_node_scoped_port(merged, graph, C.PORT_IP_ADAPTER_IMAGE_EMBEDS)
    _pop_expand_node_scoped_port(merged, graph, C.PORT_IP_ADAPTER_MASK_IMAGES)


def verify_devices(graph: Any, expected: str = "cuda") -> Dict[str, str]:
    """Check device placement of GPU-backed nodes. Returns {node_id: device} for inspection."""
    result: Dict[str, str] = {}
    nodes = getattr(graph, "_nodes", None) or {}
    for nid, node in nodes.items():
        dev = None
        for attr in ("_unet", "_controlnet", "_vae", "_text_encoder", "_image_encoder"):
            mod = getattr(node, attr, None)
            if mod is not None and hasattr(mod, "parameters"):
                try:
                    p = next(mod.parameters(), None)
                    if p is not None:
                        dev = str(p.device)
                        break
                except StopIteration:
                    pass
        if dev is not None:
            result[nid] = dev
    return result


def _latent_init_accepts_missing_encoded_latents(graph: Any) -> bool:
    """True if some ``latent_init`` node can run without ``init_latents`` (noise-only path)."""
    for nid in getattr(graph, "node_ids", ()) or ():
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        bt = getattr(node, "block_type", "") or ""
        if "latent_init" not in bt:
            continue
        port = node.get_port(C.PORT_INIT_LATENTS) if hasattr(node, "get_port") else None
        return port is None or bool(getattr(port, "optional", False))
    return False


def _iter_ip_adapter_mask_prep_node_ids(graph: Any) -> List[str]:
    out: List[str] = []
    for nid in getattr(graph, "node_ids", ()) or ():
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is not None and getattr(node, "block_type", "") == "common/ip_adapter_mask_prep":
            out.append(nid)
    return out


def _sync_ip_adapter_mask_prep_pin(
    graph: Any,
    run_kwargs: Dict[str, Any],
    merged_inputs: Dict[str, Any],
) -> None:
    """When mask prep is wired to the UNet, pin ``ip_adapter_masks=None`` if no mask images.

    Avoids stale mask tensors on the UNet input after a run that used masks. When mask images
    are supplied for a prep node, clear that node's ``pin_data`` so ``forward`` runs.

    If ``ip_adapter_masks`` is supplied directly on the backbone (prepacked tensor), skip mask
    prep so the edge does not override that input.
    """
    prep_ids = _iter_ip_adapter_mask_prep_node_ids(graph)
    if not prep_ids:
        return
    pin = run_kwargs["pin_data"]
    get_spec = getattr(graph, "get_input_spec", None)
    input_spec: List[Dict[str, Any]] = list(get_spec() or []) if callable(get_spec) else []
    merged = dict(merged_inputs or {})
    direct_unet_masks = _graph_has_merged_backbone_ip_adapter_masks(
        graph, merged, input_spec,
    )
    skip_extra: Set[str] = set()
    for nid in prep_ids:
        has_masks = _merged_provides_input_for_node_port(
            merged, nid, C.PORT_IP_ADAPTER_MASK_IMAGES, input_spec,
        )
        if has_masks:
            pin.pop(nid, None)
        elif direct_unet_masks:
            pin.pop(nid, None)
            skip_extra.add(nid)
        else:
            pin.setdefault(nid, {})[C.PORT_IP_ADAPTER_MASKS] = None
    if skip_extra:
        prev = set(run_kwargs.get("skip_node_ids") or ())
        run_kwargs["skip_node_ids"] = prev | skip_extra


def _merged_provides_input_for_node_port(
    merged: Dict[str, Any],
    nid: str,
    pname: str,
    input_spec: List[Dict[str, Any]],
) -> bool:
    """True if *merged* seeds this port the same way :meth:`EdgeBuffers.init_from_inputs` would."""
    for entry in input_spec:
        enid = entry.get("node_id") or entry.get("graph_id")
        if enid != nid or entry.get("port_name") != pname:
            continue
        name = entry.get("name")
        candidates: List[str] = []
        if name is not None:
            candidates.append(name)
        candidates.append(pname)
        candidates.append(f"{nid}:{pname}")
        for k in candidates:
            if k in merged and merged[k] is not None:
                return True
        if merged.get((nid, pname)) is not None:  # type: ignore[arg-type]
            return True
    return merged.get(f"{nid}:{pname}") is not None


def _graph_has_merged_backbone_ip_adapter_masks(
    graph: Any,
    merged: Dict[str, Any],
    input_spec: List[Dict[str, Any]],
) -> bool:
    """True if *merged* seeds ``ip_adapter_masks`` on a UNet / transformer exposed port."""
    for uid in getattr(graph, "node_ids", ()) or ():
        node = graph.get_node(uid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        bt = getattr(node, "block_type", "") or ""
        if not (bt.endswith("/unet") or bt.endswith("/transformer")):
            continue
        if _merged_provides_input_for_node_port(
            merged, uid, C.PORT_IP_ADAPTER_MASKS, input_spec,
        ):
            return True
    return False


def _diffusion_skip_inactive_adapters(graph: Any, merged: Dict[str, Any]) -> Set[str]:
    """Skip ControlNet nodes when this run supplies no conditioning image for them.

    ControlNet returns ``{}`` when ``control_image is None`` → no writes; skipping avoids stale
    residuals on the UNet.

    **IP-Adapter is never skipped here:** :class:`~yggdrasill.integrations.diffusers.adapters.ip_adapter.IPAdapterNode`
    always emits ``image_embeds`` (reference encoding or inactive zeros). With **multiple** IP nodes,
    edges CONCAT into the UNet; skipping one node removes its tensor so the list length no longer
    matches ``encoder_hid_proj`` slots → wrong adapter / dead conditioning. Per-run strength for
    unused slots is handled by :func:`_inject_ip_adapter_scale` (scale ``0.0``).
    """
    get_spec = getattr(graph, "get_input_spec", None)
    input_spec: List[Dict[str, Any]] = list(get_spec() or []) if callable(get_spec) else []

    skip: Set[str] = set()
    nids = list(getattr(graph, "node_ids", ()) or ())
    for nid in nids:
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        bt = getattr(node, "block_type", "") or ""
        if bt == "adapter/controlnet":
            if not _merged_provides_input_for_node_port(
                merged, nid, C.PORT_CONTROL_IMAGE, input_spec,
            ):
                skip.add(nid)
    return skip


def _ip_adapter_node_has_conditioning(
    merged: Dict[str, Any],
    nid: str,
    input_spec: List[Dict[str, Any]],
) -> bool:
    """True if this run supplies ``ip_adapter_image`` and/or ``ip_adapter_image_embeds``."""
    return (
        _merged_provides_input_for_node_port(
            merged, nid, C.PORT_IP_ADAPTER_IMAGE, input_spec,
        )
        or _merged_provides_input_for_node_port(
            merged, nid, C.PORT_IP_ADAPTER_IMAGE_EMBEDS, input_spec,
        )
    )


def _diffusion_universal_skip_nodes(
    graph: Any, merged: Dict[str, Any], run_kw: Dict[str, Any],
) -> Set[str]:
    """Skip encode / mask prep when no init image is provided.

    * Graphs completed by ``try_complete_*_universal_diffusion`` set
      ``sd15_universal`` / ``sdxl_universal``.
    * Img2img presets (``sd15_img2img``, etc.) also wire ``img_encode`` →
      ``latent_init`` but omit ``sd15_universal``; the same skip applies for
      text2img-style runs (e.g. template + ControlNet without ``image``).
    """
    def _has(keys: tuple) -> bool:
        for k in keys:
            if merged.get(k) is not None:
                return True
            if run_kw.get(k) is not None:
                return True
        return False

    if _has(("image", "init_image")):
        return set()

    meta = getattr(graph, "metadata", None) or {}
    if meta.get("sd15_universal") or meta.get("sdxl_universal"):
        return {"img_encode", "mask_prep"}

    nids = set(getattr(graph, "node_ids", ()) or ())
    if "img_encode" not in nids:
        return set()
    if not _latent_init_accepts_missing_encoded_latents(graph):
        return set()

    skip: Set[str] = {"img_encode"}
    if "mask_prep" in nids:
        skip.add("mask_prep")
    return skip


def _resolve_canvas_width_height(graph: Any, run_kwargs: Dict[str, Any]) -> tuple[Any, Any]:
    """Effective width/height for conditioning that must match the latent canvas.

    ``run(width=..., height=...)`` wins. If either is missing, fill from the first
    ``*latent_init*`` node's ``_config`` (template defaults). This keeps
    :class:`~yggdrasill.integrations.diffusers.common.ip_adapter_mask_prep.IPAdapterMaskPrepNode`
    resizing IP-Adapter masks to the **generation** size; otherwise masks stay at PNG resolution
    and Diffusers' ``IPAdapterMaskProcessor.downsample`` pads/crops per-token maps → visible
    8×8-aligned grid corruption.
    """
    w, h = run_kwargs.get("width"), run_kwargs.get("height")
    nodes = getattr(graph, "_nodes", None) or {}
    for node in nodes.values():
        bt = getattr(node, "block_type", "") or ""
        if "latent_init" not in bt:
            continue
        cfg = getattr(node, "_config", None) or {}
        if w is None and cfg.get("width") is not None:
            try:
                w = int(cfg["width"])
            except (TypeError, ValueError):
                pass
        if h is None and cfg.get("height") is not None:
            try:
                h = int(cfg["height"])
            except (TypeError, ValueError):
                pass
        if w is not None and h is not None:
            break
    return w, h


def _prepare_diffusion_run(
    graph: Any,
    run_kwargs: Dict[str, Any],
    merged_inputs: Optional[Dict[str, Any]] = None,
) -> None:
    """In-place preparation for diffusion run (device, node config overrides).

    Call before graph.run() when you need to inject num_inference_steps,
    seed, or device into specific node configs. The engine's run() already
    routes num_inference_steps→num_loop_steps and seed to the executor;
    this hook is for any extra diffusion-specific setup.
    """
    merged = dict(merged_inputs or {})
    if run_kwargs.get("pin_data") is None:
        run_kwargs["pin_data"] = {}
    # Apply guess_mode before node execution (ControlNetNode reads it from _config).
    _inject_guess_mode(graph, run_kwargs.pop("guess_mode", None))
    _inject_control_guidance_window(
        graph,
        start=run_kwargs.pop("control_guidance_start", None),
        end=run_kwargs.pop("control_guidance_end", None),
    )
    _enforce_ip_adapter_multi_ref_with_masks(graph, merged)
    _sync_ip_adapter_mask_prep_pin(graph, run_kwargs, merged)
    extra_skip = _diffusion_universal_skip_nodes(graph, merged, run_kwargs)
    extra_skip |= _diffusion_skip_inactive_adapters(graph, merged)
    if extra_skip:
        prev = set(run_kwargs.get("skip_node_ids") or ())
        run_kwargs["skip_node_ids"] = prev | extra_skip

    device = run_kwargs.get("device")
    if device is not None and hasattr(graph, "to") and callable(getattr(graph, "to")):
        graph.to(device)

    # Match pipeline_controlnet: width/height must drive latent_init *and* ControlNet conditioning.
    # ControlNet nodes from add_component often had no width/height keys, so graph.run(width=…)
    # did not update them (structure._resolve_run_kwargs only patches keys already in node._config).
    # Also infer missing dimensions from latent_init so IP-Adapter mask prep matches canvas
    # when the caller omits width/height on run().
    w, h = _resolve_canvas_width_height(graph, run_kwargs)
    if w is not None or h is not None:
        nodes = getattr(graph, "_nodes", None) or {}
        for node in nodes.values():
            bt = getattr(node, "block_type", "") or ""
            if (
                "latent_init" not in bt
                and "adapter/controlnet" not in bt
                and "sdxl/added_conditioning" not in bt
                and "ip_adapter_mask_prep" not in bt
            ):
                continue
            if not hasattr(node, "_config"):
                node._config = {}
            if w is not None:
                node._config["width"] = int(w)
            if h is not None:
                node._config["height"] = int(h)
            if "sdxl/added_conditioning" in bt and w is not None and h is not None:
                node._config["original_size"] = (int(h), int(w))
                node._config["target_size"] = (int(h), int(w))

    # CFG batch doubling must agree between UNet and ControlNet. If guidance_scale is only on the UNet
    # (or vice versa), one path runs batch 1 and the other batch 2 → shape errors or garbage latents.
    gs = run_kwargs.get("guidance_scale")
    if gs is not None:
        gsf = float(gs)
        nodes = getattr(graph, "_nodes", None) or {}
        for node in nodes.values():
            bt = getattr(node, "block_type", "") or ""
            if not (bt.endswith("/unet") or "adapter/controlnet" in bt):
                continue
            if not hasattr(node, "_config"):
                node._config = {}
            node._config["guidance_scale"] = gsf
