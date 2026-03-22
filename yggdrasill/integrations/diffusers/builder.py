"""DiffusionGraphBuilder: high-level API for adding components to diffusion graphs.

Resolves component types via FamilyRegistry, loads from ModelStore,
adds implicit nodes, and delegates to graph.add_node with ready nodes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from yggdrasill.engine.edge import Edge
from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import (
    _load_ip_adapter_state_dict,
    reload_ip_adapter_weights_on_unet,
)
from yggdrasill.engine.structure import Hypergraph

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.components import (
    is_component_type,
    load_components_from_pretrained,
    resolve_component_type,
)
from yggdrasill.integrations.diffusers.family_registry import get_family_spec


_FALLBACK_SCHEDULER_REPO: Dict[str, str] = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-dev",
}

# After replace_component swaps UNet/transformer, encourage immediate CUDA reclaim
# (old weights are dropped in Hypergraph.remove_node via _release_node_gpu_backing).
_BACKBONE_REPLACE_COMPONENT_TYPES = frozenset({
    "sd15.unet",
    "sd15.backbone",
    "sdxl.unet",
    "sdxl.backbone",
    "flux.transformer",
    "flux.backbone",
})


def _reclaim_cuda_after_backbone_replace() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _instantiate_scheduler_by_type(
    scheduler_type: str,
    template: Any,
    family: str,
) -> Any:
    """Build a Diffusers scheduler from *scheduler_type*, optionally matching *template* config."""
    key = scheduler_type.strip().lower().replace("-", "_")
    import diffusers.schedulers as sched_mod

    ldm_classes: Dict[str, Any] = {
        "euler": sched_mod.EulerDiscreteScheduler,
        "euler_ancestral": sched_mod.EulerAncestralDiscreteScheduler,
        "ddim": sched_mod.DDIMScheduler,
        "ddpm": sched_mod.DDPMScheduler,
        "pndm": sched_mod.PNDMScheduler,
        "lms": sched_mod.LMSDiscreteScheduler,
        "dpm": sched_mod.DPMSolverMultistepScheduler,
        "dpm_solver": sched_mod.DPMSolverMultistepScheduler,
        "dpmsolver": sched_mod.DPMSolverMultistepScheduler,
        "dpmsolver_multistep": sched_mod.DPMSolverMultistepScheduler,
        "heun": sched_mod.HeunDiscreteScheduler,
        "unipc": sched_mod.UniPCMultistepScheduler,
    }
    flux_classes: Dict[str, Any] = {
        "flow_match": sched_mod.FlowMatchEulerDiscreteScheduler,
        "flux": sched_mod.FlowMatchEulerDiscreteScheduler,
        "euler": sched_mod.FlowMatchEulerDiscreteScheduler,
    }
    mapping = flux_classes if family == "flux" else ldm_classes
    cls = mapping.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown scheduler_type '{scheduler_type}' for family '{family}'. "
            f"Try one of: {', '.join(sorted(mapping.keys()))}"
        )

    if template is not None and hasattr(template, "config"):
        try:
            return cls.from_config(template.config)
        except Exception:
            pass

    repo = _FALLBACK_SCHEDULER_REPO.get(family, _FALLBACK_SCHEDULER_REPO["sd15"])
    return cls.from_pretrained(repo, subfolder="scheduler")


def _infer_hypergraph_device(graph: Any) -> Any:
    """Resolve target device for a diffusion graph.

    Order: ``metadata['device']`` (set by :meth:`Hypergraph.to`), then any
    non-cpu ``node._config['device']``, then ``cpu`` from config, then first
    module parameter device among common diffusion attributes.
    """
    meta = getattr(graph, "metadata", None) or {}
    d = meta.get("device")
    if d is not None:
        return d
    nodes = getattr(graph, "_nodes", None) or {}
    last_cpu: Any = None
    for node in nodes.values():
        cfg = getattr(node, "_config", None) or {}
        cd = cfg.get("device")
        if cd is None:
            continue
        if str(cd) != "cpu":
            return cd
        last_cpu = cd
    if last_cpu is not None:
        return last_cpu
    for node in nodes.values():
        for attr in (
            "_unet", "_vae", "_text_encoder", "_transformer", "_controlnet",
        ):
            mod = getattr(node, attr, None)
            if mod is None:
                continue
            try:
                p = next(mod.parameters(), None)
                if p is not None:
                    return p.device
            except Exception:
                continue
    return None


_META_IP_ADAPTER_SDS = "ip_adapter_accumulated_state_dicts"
_META_IP_ADAPTER_ORDER = "ip_adapter_weight_node_ids"


def _sync_ip_adapter_plus_token_embed_dims_from_unet(graph: Any, unet: Any) -> None:
    """Set each IP-Adapter node's Plus token width from the matching ``encoder_hid_proj`` layer."""
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    unet = resolve_if_lazy(unet)
    if unet is None:
        return
    wrap = getattr(unet, "encoder_hid_proj", None)
    layers = getattr(wrap, "image_projection_layers", None) if wrap is not None else None
    if not layers:
        return
    key = C.CFG_IP_ADAPTER_PLUS_TOKEN_EMBED_DIM
    order = list(graph.metadata.get(_META_IP_ADAPTER_ORDER) or [])
    if len(order) != len(layers):
        ip_sorted = [
            n
            for n in sorted(graph.node_ids)
            if (graph.get_node(n) is not None)
            and getattr(graph.get_node(n), "block_type", "") == "adapter/ip_adapter"
        ]
        if len(ip_sorted) == len(layers):
            order = ip_sorted
    for idx, layer in enumerate(layers):
        pin = getattr(layer, "proj_in", None)
        if pin is None or not hasattr(pin, "in_features"):
            continue
        dim = int(getattr(pin, "in_features", 0) or 0)
        if dim <= 0:
            continue
        nid = order[idx] if idx < len(order) else None
        if nid is None:
            continue
        node = graph.get_node(nid)
        if node is None or getattr(node, "block_type", "") != "adapter/ip_adapter":
            continue
        if not hasattr(node, "_config"):
            node._config = {}
        node._config[key] = dim


def _load_ip_adapter_weights_into_graph(
    graph: Any,
    *,
    pretrained: str,
    subfolder: str = "models",
    weight_name: str = "ip-adapter_sd15.bin",
    ip_adapter_scale: Optional[float] = None,
    adapter_node_id: str,
) -> None:
    """Append one IP-Adapter checkpoint and reload **all** accumulated weights on the UNet (diffusers API).

    Diffusers ``_load_ip_adapter_weights`` always rebuilds processors for the full ``state_dicts`` list,
    so each new ``add_component(..., sdxl.ipadapter, ...)`` must merge with previous checkpoints.
    Scales are applied at run time via ``ip_adapter_conditioning_scale``; *ip_adapter_scale* here is
    only forwarded when this is the **first** adapter on the graph (single-adapter backward compat).
    """
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    sd = _load_ip_adapter_state_dict(
        pretrained,
        subfolder=subfolder,
        weight_name=weight_name,
    )
    acc = graph.metadata.setdefault(_META_IP_ADAPTER_SDS, [])
    order = graph.metadata.setdefault(_META_IP_ADAPTER_ORDER, [])
    acc.append(sd)
    order.append(adapter_node_id)

    for nid in graph.node_ids:
        node = graph.get_node(nid)
        bt = getattr(node, "block_type", "") or ""
        if not (bt.endswith("/unet") or bt.endswith("/transformer")):
            continue
        unet = getattr(node, "_unet", None)
        if unet is None:
            continue
        resolved = resolve_if_lazy(unet)
        if resolved is not unet:
            node._unet = resolved
        reload_ip_adapter_weights_on_unet(resolved, acc, low_cpu_mem_usage=True)
        _sync_ip_adapter_plus_token_embed_dims_from_unet(graph, resolved)
        if ip_adapter_scale is not None and len(acc) == 1:
            from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import (
                _set_ip_adapter_scale_on_unet,
            )

            _set_ip_adapter_scale_on_unet(resolved, ip_adapter_scale)
        return
    import logging

    logging.getLogger(__name__).warning(
        "IP-Adapter weights not loaded: no UNet/transformer node in graph. "
        "Add sd15.unet (or sdxl.unet / flux.transformer) before sd15.ipadapter / sdxl.ipadapter."
    )


_IP_ADAPTER_MASK_PREP_NODE_ID = "ip_mask_prep"


def _graph_has_block_type(graph: Any, block_type: str) -> bool:
    for nid in graph.node_ids:
        node = graph.get_node(nid)
        if getattr(node, "block_type", "") == block_type:
            return True
    return False


def _find_backbone_node_id(graph: Any) -> Optional[str]:
    for nid in graph.node_ids:
        node = graph.get_node(nid)
        bt = getattr(node, "block_type", "") or ""
        if bt.endswith("/unet") or bt.endswith("/transformer"):
            return nid
    return None


def _ensure_ip_adapter_mask_prep(graph: Any) -> None:
    """Add ``ip_mask_prep`` → UNet when IP-Adapter is present; run.py pins ``None`` if no masks."""
    if _graph_has_block_type(graph, "common/ip_adapter_mask_prep"):
        return
    if not _graph_has_block_type(graph, "adapter/ip_adapter"):
        return
    unet_nid = _find_backbone_node_id(graph)
    if unet_nid is None:
        return
    if _IP_ADAPTER_MASK_PREP_NODE_ID in graph.node_ids:
        return
    from yggdrasill.foundation.registry import BlockRegistry
    from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect

    reg = BlockRegistry.global_registry()
    mask_node = reg.build({
        "type": "common/ip_adapter_mask_prep",
        "node_id": _IP_ADAPTER_MASK_PREP_NODE_ID,
        "config": {},
    })
    graph.add_node(_IP_ADAPTER_MASK_PREP_NODE_ID, mask_node)
    apply_port_name_auto_connect(graph, _IP_ADAPTER_MASK_PREP_NODE_ID, mask_node)
    getattr(graph, "metadata", {}).setdefault("ip_mask_prep_auto", True)


class DiffusionGraphBuilder:
    """Builds diffusion graphs by adding components with pretrained loading.

    Usage:
        builder = DiffusionGraphBuilder(Hypergraph())
        builder.add_component("unet", "sd15.unet", pretrained="runwayml/stable-diffusion-v1-5")
        builder.add_component("tokenizer", "sd15.tokenizer", ...)
        graph = builder.graph
    """

    def __init__(
        self,
        graph: Optional[Hypergraph] = None,
        *,
        graph_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        graph_id = name or graph_id or "diffusion_graph"
        self._graph = graph or Hypergraph(graph_id=graph_id)
        self._added_groups: Dict[str, str] = {}  # group -> node_id
        self._completed: bool = False

    @classmethod
    def from_template(cls, template_name: str, **kwargs: Any) -> "DiffusionGraphBuilder":
        """Wrap a graph built from a diffusion template (same kwargs as :meth:`Hypergraph.from_template`)."""
        graph = Hypergraph.from_template(template_name, **kwargs)
        return cls(graph)

    def _apply_graph_device(self) -> None:
        """Move all nodes (and schedulers) to the graph's inferred device."""
        dev = _infer_hypergraph_device(self._graph)
        if dev is not None and hasattr(self._graph, "to"):
            self._graph.to(dev)

    @property
    def graph(self) -> Hypergraph:
        self._ensure_text2img_complete()
        return self._graph

    def to(self, device: Any) -> "DiffusionGraphBuilder":
        """Move the graph to the target device. Returns self for chaining."""
        self.graph.to(device)
        return self

    def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the diffusion graph. Accepts prompt, negative_prompt, num_inference_steps,
        guidance_scale, seed, width, height, device, controlnet_image, ip_adapter_image,
        ip_adapter_image_embeds, controlnet_conditioning_scale, ip_adapter_conditioning_scale,
        (InstantStyle / per-layer scales: pass a dict with only ``down`` / ``up`` / ``mid`` keys, or a list
        of per–IP-Adapter configs as in diffusers ``set_ip_adapter_scale``). For **multiple** IP-Adapter
        nodes, ``ip_adapter_image`` may be a **list** in the same order as ``add_component`` calls that
        loaded weights (stored in graph metadata; falls back to sorted node ids if missing),
        etc. Returns DiffusionOutput."""
        from yggdrasill.integrations.diffusers.run import run as run_diffusion
        return run_diffusion(self.graph, inputs, wrap_output=True, **kwargs)

    def _ensure_text2img_complete(self) -> None:
        """Add latent_init, expose I/O, and metadata if this is an incomplete text2img topology."""
        if self._completed:
            return
        has_tokenizer = has_prompt_enc = has_unet = has_vae_decode = False
        has_sched_setup = has_sched_step = has_latent_init = False
        family = ""
        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            bt = getattr(node, "block_type", "") or ""
            if "tokenizer" in bt:
                has_tokenizer = True
            if "prompt_encoder" in bt:
                has_prompt_enc = True
            if bt.endswith("/unet") or bt.endswith("/transformer"):
                has_unet = True
            if "vae_decode" in bt:
                has_vae_decode = True
            if "scheduler_setup" in bt:
                has_sched_setup = True
            if "scheduler_step" in bt:
                has_sched_step = True
            if "latent_init" in bt:
                has_latent_init = True
            # Backbone family for latent_init fallback — must not be overwritten by
            # ``adapter/controlnet`` or ``adapter/ip_adapter`` (last nodes in typical builds).
            if "/" in bt:
                root = bt.split("/")[0]
                if root != "adapter":
                    family = root

        if not (
            has_tokenizer
            and has_prompt_enc
            and has_unet
            and has_vae_decode
            and (has_sched_setup or has_sched_step)
            and not has_latent_init
        ):
            # Graph is already complete (e.g. from_template). Still expose adapter ports
            # (control_image, ip_adapter_image) if ControlNet/IPAdapter nodes were added.
            self.expose_default_io()
            self._completed = True
            return

        # SD1.5 manual stack: one graph for text2img / img2img / inpaint (optional image, mask).
        if family == "sd15":
            from yggdrasill.integrations.diffusers.sd15.universal import (
                try_complete_sd15_universal_diffusion,
            )

            if try_complete_sd15_universal_diffusion(self._graph):
                for nid in self._graph.node_ids:
                    node = self._graph.get_node(nid)
                    bt = getattr(node, "block_type", "") or ""
                    if "scheduler_setup" in bt and hasattr(node, "_config"):
                        node._config = node._config or {}
                        node._config.setdefault("device", "cuda")
                        node._config.setdefault("num_inference_steps", 50)
                self.expose_default_io()
                self._graph.metadata.setdefault("num_loop_steps", 50)
                self._completed = True
                return

        if family == "sdxl":
            from yggdrasill.integrations.diffusers.sdxl.universal import (
                try_complete_sdxl_universal_diffusion,
            )

            if try_complete_sdxl_universal_diffusion(self._graph):
                for nid in self._graph.node_ids:
                    node = self._graph.get_node(nid)
                    bt = getattr(node, "block_type", "") or ""
                    if "scheduler_setup" in bt and hasattr(node, "_config"):
                        node._config = node._config or {}
                        node._config.setdefault("device", "cuda")
                        node._config.setdefault("num_inference_steps", 50)
                        node._config.setdefault("height", 1024)
                        node._config.setdefault("width", 1024)
                self.expose_default_io()
                self._graph.metadata.setdefault("num_loop_steps", 50)
                self._completed = True
                return

        # Fallback: text2img-only completion (no image/mask path).
        latent_type = f"{family}.latent_init" if family else "sd15.latent_init"
        cfg: Dict[str, Any] = {
            "height": 512,
            "width": 512,
            "device": "cuda",
            "dtype": "float16",
        }
        if family == "sdxl":
            cfg["height"] = 1024
            cfg["width"] = 1024
        self.add_component("LatentInit", latent_type, config=cfg)

        # Update scheduler_setup config for device and num_inference_steps
        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            bt = getattr(node, "block_type", "") or ""
            if "scheduler_setup" in bt and hasattr(node, "_config"):
                node._config = node._config or {}
                node._config.setdefault("device", "cuda")
                node._config.setdefault("num_inference_steps", 50)

        self.expose_default_io()
        self._graph.metadata["num_loop_steps"] = 50
        self._completed = True

    def add_component(
        self,
        node_id: str,
        component_type: str,
        *,
        pretrained: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        store: Optional[Any] = None,
        variant: str = "",
        torch_dtype: Optional[Any] = None,
        subfolder: Optional[str] = None,
        weight_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "DiffusionGraphBuilder":
        """Add a component node (or update an existing grouped node).

        Args:
            node_id: Graph node id.
            component_type: E.g. "sd15.unet", "sd15.controlnet", "sd15.ipadapter".
            pretrained: HF repo id or local path for loading.
            config: Node config overrides.
            store: ModelStore instance (optional).
            variant: Model variant (e.g. "fp16").
            torch_dtype: Target dtype for loaded models.
            subfolder: Subfolder in repo (e.g. "models" for IP-Adapter).
            weight_name: Weight filename (e.g. "ip-adapter_sd15.bin"). Passed to config.
            **kwargs: Additional overrides merged into config.

        Returns:
            self for chaining.
        """
        if not is_component_type(component_type):
            raise ValueError(
                f"Expected component type (e.g. 'sd15.unet'), got '{component_type}'"
            )

        spec = resolve_component_type(component_type)
        family = component_type.split(".", 1)[0]
        load_family = getattr(spec, "load_family", None) or family
        family_spec = get_family_spec(family)

        cfg = dict(config or {})
        if subfolder is not None:
            cfg["subfolder"] = subfolder
        if weight_name is not None:
            cfg["weight_name"] = weight_name
        # Default IP-Adapter config when adding ipadapter with h94/IP-Adapter
        if (
            pretrained
            and "h94/IP-Adapter" in str(pretrained)
            and subfolder is None
            and weight_name is None
        ):
            if component_type == "sdxl.ipadapter":
                cfg.setdefault("subfolder", "sdxl_models")
                cfg.setdefault("weight_name", "ip-adapter_sdxl.bin")
            elif component_type in ("sd15.ipadapter", "adapter.ip_adapter"):
                cfg.setdefault("subfolder", "models")
                cfg.setdefault("weight_name", "ip-adapter_sd15.bin")
        cfg.update(kwargs)
        # IP-Adapter Plus / Plus-Face: UNet projection expects CLIP vision hidden states, not pooled
        # image_embeds (see diffusers SDXL prepare_ip_adapter_image_embeds / encode_image).
        wn = cfg.get("weight_name")
        if wn is not None and spec.block_types and any(
            "ip_adapter" in str(bt) for bt in spec.block_types
        ):
            wn_blob = " ".join(str(x).lower() for x in wn) if isinstance(wn, (list, tuple)) else str(wn).lower()
            if "plus" in wn_blob:
                cfg.setdefault("ip_adapter_use_hidden_states", True)
        if pretrained is not None:
            cfg.setdefault("pretrained", str(pretrained))
        if "controlnet" in component_type:
            if component_type.startswith("sdxl."):
                cfg.setdefault("width", 1024)
                cfg.setdefault("height", 1024)
            else:
                cfg.setdefault("width", 512)
                cfg.setdefault("height", 512)

        # Resolve torch_dtype: use family default when loading pretrained
        dtype_to_load = torch_dtype
        if dtype_to_load is None and pretrained and spec.load_keys:
            dtype_str = family_spec.torch_dtype_default
            dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
            dtype_to_load = dtype_map.get(dtype_str, torch.float16)

        # Load pretrained components if requested
        components_loaded: Dict[str, Any] = {}
        if pretrained and spec.load_keys:
            load_variant = variant if variant else ("fp16" if dtype_to_load == torch.float16 else "")
            pretrained_map = getattr(spec, "load_pretrained_map", None)
            subfolder_map = getattr(spec, "load_subfolder_map", None)
            variant_map = getattr(spec, "load_variant_map", None)
            components_loaded = load_components_from_pretrained(
                spec.load_keys,
                pretrained,
                family=load_family,
                store=store,
                torch_dtype=dtype_to_load,
                variant=load_variant,
                pretrained_map=pretrained_map,
                subfolder_map=subfolder_map,
                variant_map=variant_map,
            )

        # Build constructor kwargs from constructor_map + loaded components
        from yggdrasill.foundation.registry import BlockRegistry
        reg = BlockRegistry.global_registry()

        for block_type in spec.block_types:
            const_map = spec.constructor_map.get(block_type, {})
            kwargs: Dict[str, Any] = {}
            for ctor_kwarg, load_key in const_map.items():
                val = components_loaded.get(load_key)
                if val is not None:
                    kwargs[ctor_kwarg] = val

            if config:
                kwargs["config"] = dict(cfg)

            if spec.group and spec.group in self._added_groups:
                # Update existing grouped node (e.g. prompt_encoder with text_encoder_2)
                existing_id = self._added_groups[spec.group]
                existing_node = self._graph._nodes.get(existing_id)
                if existing_node is not None and hasattr(existing_node, "update_from_components"):
                    existing_node.update_from_components(kwargs)
                    self._apply_graph_device()
                    return self

            if len(spec.block_types) == 1:
                nid = node_id
            else:
                # e.g. scheduler_setup -> _setup, scheduler_step -> _step
                suffix = "_" + block_type.split("/")[-1].split("_", 1)[-1]
                nid = f"{node_id}{suffix}"
            build_cfg: Dict[str, Any] = {
                "type": block_type,
                "node_id": nid,
                "config": kwargs.get("config", cfg),
            }
            for k, v in kwargs.items():
                if k not in ("config", "type", "node_id"):
                    build_cfg[k] = v

            node = reg.build(build_cfg)
            self._graph.add_node(nid, node)
            if spec.group:
                self._added_groups[spec.group] = nid
            from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect
            apply_port_name_auto_connect(self._graph, nid, node)

        # IP-Adapter: load weights into UNet when pretrained + weight config are provided
        if (
            "ip_adapter" in str(spec.block_types)
            and pretrained
            and (cfg.get("subfolder") or cfg.get("weight_name"))
        ):
            _load_ip_adapter_weights_into_graph(
                self._graph,
                pretrained=pretrained,
                subfolder=cfg.get("subfolder", "models"),
                weight_name=cfg.get("weight_name", "ip-adapter_sd15.bin"),
                ip_adapter_scale=cfg.get(C.CFG_IP_ADAPTER_SCALE),
                adapter_node_id=nid,
            )

        if "ip_adapter" in str(spec.block_types):
            _ensure_ip_adapter_mask_prep(self._graph)

        # After first .graph / .run, _ensure_text2img_complete() sets _completed and will
        # not call expose_default_io again; new ControlNet / IP-Adapter nodes must expose
        # control_image / ip_adapter_image or run-time dicts never reach EdgeBuffers.
        self.expose_default_io()
        self._apply_graph_device()
        return self

    def _find_unet_node(self) -> Optional[Any]:
        """Find the Backbone (UNet/transformer) node in the graph."""
        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            bt = getattr(node, "block_type", "") or ""
            if bt.endswith("/unet") or bt.endswith("/transformer"):
                return node
        return None

    def add_node(self, node_id: str, node: Any) -> "DiffusionGraphBuilder":
        """Add a pre-built node directly. Auto-connects by port names."""
        from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect
        self._graph.add_node(node_id, node)
        apply_port_name_auto_connect(self._graph, node_id, node)
        self.expose_default_io()
        self._apply_graph_device()
        return self

    def add_edge(self, source: str, source_port: str, target: str, target_port: str) -> "DiffusionGraphBuilder":
        """Add an edge. Uses contract port names when passed as strings."""
        self._graph.add_edge(Edge(source, source_port, target, target_port))
        return self

    def expose_default_io(self) -> "DiffusionGraphBuilder":
        """Expose standard diffusion inputs/outputs for text2img topologies.

        Finds tokenizer-like nodes and exposes prompt/negative_prompt;
        finds vae_decode nodes and exposes decoded_image as output_image.
        Safe to call multiple times; skips already-exposed ports.
        """
        from yggdrasill.foundation.node import AbstractGraphNode

        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            if not isinstance(node, AbstractGraphNode):
                continue
            bt = getattr(node, "block_type", "") or ""
            in_names = {p.name for p in node.get_input_ports()}
            out_names = {p.name for p in node.get_output_ports()}

            if "tokenizer" in bt:
                if C.PORT_PROMPT in in_names:
                    self._graph.expose_input(nid, C.PORT_PROMPT, C.PORT_PROMPT)
                if C.PORT_NEGATIVE_PROMPT in in_names:
                    self._graph.expose_input(
                        nid, C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT
                    )
            if "vae_decode" in bt and C.PORT_DECODED_IMAGE in out_names:
                self._graph.expose_output(nid, C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)
            if "adapter/ip_adapter" in bt and C.PORT_IP_ADAPTER_IMAGE in in_names:
                # Use node-scoped key so multi-IPAdapter graphs work with ip_adapter_image={node_id: img}
                self._graph.expose_input(nid, C.PORT_IP_ADAPTER_IMAGE, f"{nid}:{C.PORT_IP_ADAPTER_IMAGE}")
            if "adapter/ip_adapter" in bt and C.PORT_IP_ADAPTER_IMAGE_EMBEDS in in_names:
                self._graph.expose_input(
                    nid,
                    C.PORT_IP_ADAPTER_IMAGE_EMBEDS,
                    f"{nid}:{C.PORT_IP_ADAPTER_IMAGE_EMBEDS}",
                )
            if "adapter/controlnet" in bt and C.PORT_CONTROL_IMAGE in in_names:
                # Use node-scoped key so multi-ControlNet graphs work with controlnet_image={node_id: img}
                self._graph.expose_input(nid, C.PORT_CONTROL_IMAGE, f"{nid}:{C.PORT_CONTROL_IMAGE}")
            if "ip_adapter_mask_prep" in bt and C.PORT_IP_ADAPTER_MASK_IMAGES in in_names:
                self._graph.expose_input(
                    nid, C.PORT_IP_ADAPTER_MASK_IMAGES, C.PORT_IP_ADAPTER_MASK_IMAGES,
                )

        return self

    def _resolve_scheduler_base_id(self) -> Optional[str]:
        """Return base id (e.g. ``sched``) for ``sched_setup`` / ``sched_step`` pair."""
        for nid in sorted(self._graph.node_ids):
            node = self._graph.get_node(nid)
            if node is None:
                continue
            bt = getattr(node, "block_type", "") or ""
            if "scheduler_setup" in bt and nid.endswith("_setup"):
                return nid[: -len("_setup")]
        return None

    def _resolve_role_to_node_id(self, role_or_id: str) -> str:
        """Resolve canonical role name (e.g. 'Backbone') to actual graph node id."""
        if role_or_id in ("Scheduler", "scheduler"):
            base = self._resolve_scheduler_base_id()
            if base is not None:
                return base
        role_map = {
            "Backbone": ("unet", "transformer"),
            "Conjector": ("prompt_encoder",),
        }
        aliases = role_map.get(role_or_id)
        if aliases is None:
            return role_or_id
        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            bt = getattr(node, "block_type", "") or ""
            if any(a in bt for a in aliases):
                return nid
        return role_or_id

    def replace_component(
        self,
        node_id: str,
        component_type: str,
        *,
        pretrained: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        store: Optional[Any] = None,
        variant: str = "",
        torch_dtype: Optional[Any] = None,
        use_safetensors: Optional[bool] = None,
        **kwargs: Any,
    ) -> "DiffusionGraphBuilder":
        """Replace an existing node with a new component from *component_type*.

        For single-block components (unet, tokenizer, vae_decode), replaces
        the given node_id. For multi-block (e.g. scheduler), pass the base
        id (e.g. "sched") to replace both setup and step nodes.

        Supports canonical role names: ``Backbone`` → unet/transformer node,
        ``Conjector`` → prompt_encoder node, ``Scheduler`` → scheduler pair
        (``sched_setup`` / ``sched_step``).

        Args:
            node_id: Graph node id to replace (or role name: Backbone, Conjector, Scheduler).
            component_type: E.g. "sd15.unet", "sd15.scheduler".
        pretrained: HF repo id or local path for loading.
        config: Node config overrides.
        store: ModelStore instance (optional).
        variant: Model variant (e.g. "fp16").
        torch_dtype: Target dtype for loaded models.
        use_safetensors: If False, load .bin instead of .safetensors (needed for
            repos like Lykon/DreamShaper that have only diffusion_pytorch_model.bin).
        **kwargs: Merged into node config; use ``scheduler_type="euler"`` (etc.) to swap
            the Diffusers scheduler class without reloading the whole repo (SD/SDXL/FLUX).
            Note: ``stabilityai/stable-diffusion-xl-base-1.0`` already ships an Euler scheduler;
            replacing with ``scheduler_type="euler"`` only re-instantiates Euler (no visual change).
            Try ``"dpm_solver"``, ``"ddim"``, ``"unipc"``, etc. to see a different sampler.

        Returns:
            self for chaining.
        """
        cfg = dict(config or {})
        cfg.update(kwargs)
        scheduler_type = cfg.pop("scheduler_type", None)

        node_id = self._resolve_role_to_node_id(node_id)
        if not is_component_type(component_type):
            raise ValueError(
                f"Expected component type (e.g. 'sd15.unet'), got '{component_type}'"
            )

        spec = resolve_component_type(component_type)
        family = component_type.split(".", 1)[0]
        load_family = getattr(spec, "load_family", None) or family

        components_loaded: Dict[str, Any] = {}
        if pretrained and spec.load_keys:
            components_loaded = load_components_from_pretrained(
                spec.load_keys,
                pretrained,
                family=load_family,
                store=store,
                torch_dtype=torch_dtype,
                variant=variant if variant else "",
                use_safetensors=use_safetensors,
            )
            missing = [k for k in spec.load_keys if components_loaded.get(k) is None]
            if missing:
                raise RuntimeError(
                    f"Failed to load {missing} from {pretrained}. "
                    "Check repo structure (unet/, diffusion_pytorch_model.safetensors or .fp16.safetensors or .bin)."
                )

        if scheduler_type and spec.load_keys and "scheduler" in spec.load_keys:
            template = components_loaded.get("scheduler")
            if template is None:
                setup_nid = f"{node_id}_setup"
                old_setup = self._graph.get_node(setup_nid)
                template = getattr(old_setup, "_scheduler", None) if old_setup else None
            components_loaded["scheduler"] = _instantiate_scheduler_by_type(
                scheduler_type, template, family
            )
        elif (
            spec.load_keys == ["scheduler"]
            and not components_loaded
            and not scheduler_type
        ):
            raise RuntimeError(
                f"replace_component({component_type!r}) needs pretrained=... "
                f"and/or scheduler_type=... (e.g. scheduler_type='euler')."
            )

        from yggdrasill.foundation.registry import BlockRegistry
        reg = BlockRegistry.global_registry()

        replaced_ids: list[str] = []

        for block_type in spec.block_types:
            const_map = spec.constructor_map.get(block_type, {})
            ctor_kwargs: Dict[str, Any] = {}
            for ctor_kwarg, load_key in const_map.items():
                val = components_loaded.get(load_key)
                if val is not None:
                    ctor_kwargs[ctor_kwarg] = val

            if len(spec.block_types) == 1:
                nid = node_id
            else:
                suffix = "_" + block_type.split("/")[-1].split("_", 1)[-1]
                nid = f"{node_id}{suffix}"

            if nid not in self._graph.node_ids:
                continue

            # Preserve node config from the replaced node (device, num_inference_steps,
            # denoising_*, etc.). Otherwise replace_component(..., scheduler_type=...) alone
            # wipes _config and scheduler defaults to cpu / wrong schedule vs latents.
            old_node = self._graph.get_node(nid)
            merged_cfg = dict(cfg)
            if old_node is not None:
                prev = getattr(old_node, "_config", None) or {}
                merged_cfg = {**dict(prev), **merged_cfg}
            ctor_kwargs["config"] = merged_cfg

            build_cfg: Dict[str, Any] = {
                "block_type": block_type,
                "node_id": nid,
                "config": ctor_kwargs.get("config", merged_cfg),
            }
            for k, v in ctor_kwargs.items():
                if k not in ("config", "block_type", "node_id"):
                    build_cfg[k] = v

            new_node = reg.build(build_cfg)
            self._graph.replace_node(nid, node=new_node)
            replaced_ids.append(nid)

        if len(spec.block_types) > 1 and not replaced_ids:
            expected = [
                f"{node_id}_" + bt.split("/")[-1].split("_", 1)[-1]
                for bt in spec.block_types
            ]
            raise ValueError(
                f"replace_component: no scheduler nodes matched base id {node_id!r}. "
                f"Expected graph node ids like {expected!r} (from add_component(\"sched\", ...) "
                f"→ sched_setup / sched_step), or pass that base explicitly. "
                f"Role name 'Scheduler' only resolves if a *scheduler_setup* node id ends with '_setup'."
            )
        if len(spec.block_types) > 1 and len(replaced_ids) != len(spec.block_types):
            raise ValueError(
                f"replace_component: partial scheduler replace — matched {replaced_ids!r}, "
                f"expected {len(spec.block_types)} nodes for {component_type!r}."
            )

        _meta = getattr(self._graph, "metadata", None) or {}
        if (
            component_type in ("sd15.unet", "sd15.backbone")
            and (
                self._graph.graph_id == "sd15_inpaint"
                or _meta.get("sd15_universal")
            )
        ):
            from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
            from yggdrasill.integrations.diffusers.presets.sd15 import (
                reconfigure_sd15_inpaint_for_unet_in_channels,
            )

            un_n = self._graph.get_node(node_id)
            inner = getattr(un_n, "_unet", None) if un_n is not None else None
            inner = resolve_if_lazy(inner) if inner is not None else None
            uc = getattr(inner, "config", None) if inner is not None else None
            in_ch = int(getattr(uc, "in_channels", 4)) if uc is not None else 4
            reconfigure_sd15_inpaint_for_unet_in_channels(self._graph, in_channels=in_ch)

        if (
            component_type in ("sdxl.unet", "sdxl.backbone")
            and (
                self._graph.graph_id == "sdxl_inpaint"
                or _meta.get("sdxl_universal")
            )
        ):
            from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
            from yggdrasill.integrations.diffusers.presets.sdxl import (
                reconfigure_sdxl_inpaint_for_unet_in_channels,
            )

            un_n = self._graph.get_node(node_id)
            inner = getattr(un_n, "_unet", None) if un_n is not None else None
            inner = resolve_if_lazy(inner) if inner is not None else None
            uc = getattr(inner, "config", None) if inner is not None else None
            in_ch = int(getattr(uc, "in_channels", 4)) if uc is not None else 4
            reconfigure_sdxl_inpaint_for_unet_in_channels(self._graph, in_channels=in_ch)

        if component_type in _BACKBONE_REPLACE_COMPONENT_TYPES and replaced_ids:
            _reclaim_cuda_after_backbone_replace()

        if "ip_adapter" in str(spec.block_types):
            _ensure_ip_adapter_mask_prep(self._graph)

        self.expose_default_io()
        self._apply_graph_device()
        return self
