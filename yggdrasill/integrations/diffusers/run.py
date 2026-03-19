"""Diffusion run wrapper: prepares diffusion-specific kwargs and wraps output in DiffusionOutput."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.output import DiffusionOutput


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
        inputs: Input dict (e.g. prompt, negative_prompt).
        num_inference_steps: Override for denoising steps.
        seed: Random seed for latent init.
        device: Target device.
        wrap_output: If True, return DiffusionOutput; otherwise raw dict.
        **kwargs: Passed through to graph.run(). ``controlnet_image`` /
            ``ip_adapter_image`` may be dicts ``{node_id: image}``; omit a node id (or pass
            ``None``) to disable that ControlNet / IP-Adapter for this run while leaving
            it on the graph.

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

    controlnet_image = run_kw.pop("controlnet_image", None)
    if isinstance(controlnet_image, dict):
        for nid, img in controlnet_image.items():
            if img is not None:
                merged[f"{nid}:{C.PORT_CONTROL_IMAGE}"] = img
    elif controlnet_image is not None:
        _assign_to_single_exposed(merged, graph, C.PORT_CONTROL_IMAGE, controlnet_image)

    ip_adapter_image = run_kw.pop("ip_adapter_image", None)
    if isinstance(ip_adapter_image, dict):
        for nid, img in ip_adapter_image.items():
            if img is not None:
                merged[f"{nid}:{C.PORT_IP_ADAPTER_IMAGE}"] = img
    elif ip_adapter_image is not None:
        _assign_to_single_exposed(merged, graph, C.PORT_IP_ADAPTER_IMAGE, ip_adapter_image)

    controlnet_conditioning_scale = run_kw.pop("controlnet_conditioning_scale", None)
    if isinstance(controlnet_conditioning_scale, dict):
        _inject_node_config(graph, controlnet_conditioning_scale, "conditioning_scale")

    ip_adapter_conditioning_scale = run_kw.pop("ip_adapter_conditioning_scale", None)
    if isinstance(ip_adapter_conditioning_scale, dict):
        _inject_ip_adapter_scale(graph, ip_adapter_conditioning_scale, merged)
    elif ip_adapter_conditioning_scale is not None:
        _inject_ip_adapter_scale(graph, {"default": float(ip_adapter_conditioning_scale)}, merged)

    _prepare_diffusion_run(graph, run_kw)
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


def _inject_ip_adapter_scale(
    graph: Any, scale_map: Dict[str, Any], merged_inputs: Dict[str, Any],
) -> None:
    """Apply per–IP-Adapter scales on the UNet; adapters with no image get strength 0."""
    if not scale_map:
        return
    try:
        from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import (
            _set_ip_adapter_scale_on_unet,
        )
    except ImportError:
        return

    node_ids = list(getattr(graph, "node_ids", []) or getattr(graph, "_nodes", {}).keys())
    ip_nodes: List[str] = []
    for nid in sorted(node_ids):
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is not None and getattr(node, "block_type", "") == "adapter/ip_adapter":
            ip_nodes.append(nid)
    default_scale = float(scale_map.get("default", 1.0))
    if ip_nodes:
        scales: List[float] = []
        for nid in ip_nodes:
            key = f"{nid}:{C.PORT_IP_ADAPTER_IMAGE}"
            active = key in merged_inputs and merged_inputs.get(key) is not None
            if active:
                v = scale_map.get(nid, default_scale)
                scales.append(float(v))
            else:
                scales.append(0.0)
        scale_payload: Any = scales[0] if len(scales) == 1 else scales
    else:
        scale_payload = float(next(iter(scale_map.values())))

    for nid in node_ids:
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        bt = getattr(node, "block_type", "") or ""
        if bt.endswith("/unet") or bt.endswith("/transformer"):
            unet = getattr(node, "_unet", None)
            if unet is not None:
                _set_ip_adapter_scale_on_unet(unet, scale_payload)
            break


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
    merged[port_name] = value

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


def _prepare_diffusion_run(graph: Any, run_kwargs: Dict[str, Any]) -> None:
    """In-place preparation for diffusion run (device, node config overrides).

    Call before graph.run() when you need to inject num_inference_steps,
    seed, or device into specific node configs. The engine's run() already
    routes num_inference_steps→num_loop_steps and seed to the executor;
    this hook is for any extra diffusion-specific setup.
    """
    device = run_kwargs.get("device")
    if device is not None and hasattr(graph, "to") and callable(getattr(graph, "to")):
        graph.to(device)

    # Match pipeline_controlnet: width/height must drive latent_init *and* ControlNet conditioning.
    # ControlNet nodes from add_component often had no width/height keys, so graph.run(width=…)
    # did not update them (structure._resolve_run_kwargs only patches keys already in node._config).
    w, h = run_kwargs.get("width"), run_kwargs.get("height")
    if w is not None or h is not None:
        nodes = getattr(graph, "_nodes", None) or {}
        for node in nodes.values():
            bt = getattr(node, "block_type", "") or ""
            if "latent_init" not in bt and "adapter/controlnet" not in bt:
                continue
            if not hasattr(node, "_config"):
                node._config = {}
            if w is not None:
                node._config["width"] = int(w)
            if h is not None:
                node._config["height"] = int(h)

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
