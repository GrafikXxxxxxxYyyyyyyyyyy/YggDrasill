"""DiffusionGraphBuilder: high-level API for adding components to diffusion graphs.

Resolves component types via FamilyRegistry, loads from ModelStore,
adds implicit nodes, and delegates to graph.add_node with ready nodes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.components import (
    is_component_type,
    load_components_from_pretrained,
    resolve_component_type,
)
from yggdrasill.integrations.diffusers.family_registry import get_family_spec


class DiffusionGraphBuilder:
    """Builds diffusion graphs by adding components with pretrained loading.

    Usage:
        builder = DiffusionGraphBuilder(Hypergraph())
        builder.add_component("unet", "sd15.unet", pretrained="runwayml/stable-diffusion-v1-5")
        builder.add_component("tokenizer", "sd15.tokenizer", ...)
        graph = builder.graph
    """

    def __init__(self, graph: Optional[Hypergraph] = None, *, graph_id: Optional[str] = None) -> None:
        self._graph = graph or Hypergraph(graph_id=graph_id or "diffusion_graph")
        self._added_groups: Dict[str, str] = {}  # group -> node_id
        self._completed: bool = False

    @property
    def graph(self) -> Hypergraph:
        self._ensure_text2img_complete()
        return self._graph

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
            if "/" in bt:
                family = bt.split("/")[0] or family

        if not (
            has_tokenizer
            and has_prompt_enc
            and has_unet
            and has_vae_decode
            and (has_sched_setup or has_sched_step)
            and not has_latent_init
        ):
            self._completed = True
            return

        # Add latent_init and complete the pipeline (matches from_template defaults)
        latent_type = f"{family}.latent_init" if family else "sd15.latent_init"
        cfg: Dict[str, Any] = {
            "height": 512,
            "width": 512,
            "device": "cuda",
            "dtype": "float32",
        }
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
    ) -> "DiffusionGraphBuilder":
        """Add a component node (or update an existing grouped node).

        Args:
            node_id: Graph node id.
            component_type: E.g. "sd15.unet", "sdxl.tokenizer".
            pretrained: HF repo id or local path for loading.
            config: Node config overrides.
            store: ModelStore instance (optional).
            variant: Model variant (e.g. "fp16").
            torch_dtype: Target dtype for loaded models.

        Returns:
            self for chaining.
        """
        if not is_component_type(component_type):
            raise ValueError(
                f"Expected component type (e.g. 'sd15.unet'), got '{component_type}'"
            )

        spec = resolve_component_type(component_type)
        family = component_type.split(".", 1)[0]
        family_spec = get_family_spec(family)

        cfg = dict(config or {})

        # Load pretrained components if requested
        components_loaded: Dict[str, Any] = {}
        if pretrained and spec.load_keys:
            components_loaded = load_components_from_pretrained(
                spec.load_keys,
                pretrained,
                family=family,
                store=store,
                torch_dtype=torch_dtype,
                variant=variant if variant else "",
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

        return self

    def add_node(self, node_id: str, node: Any) -> "DiffusionGraphBuilder":
        """Add a pre-built node directly. Auto-connects by port names."""
        from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect
        self._graph.add_node(node_id, node)
        apply_port_name_auto_connect(self._graph, node_id, node)
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

        return self

    def _resolve_role_to_node_id(self, role_or_id: str) -> str:
        """Resolve canonical role name (e.g. 'Backbone') to actual graph node id."""
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
    ) -> "DiffusionGraphBuilder":
        """Replace an existing node with a new component from *component_type*.

        For single-block components (unet, tokenizer, vae_decode), replaces
        the given node_id. For multi-block (e.g. scheduler), pass the base
        id (e.g. "sched") to replace both setup and step nodes.

        Supports canonical role names: ``Backbone`` → unet/transformer node,
        ``Conjector`` → prompt_encoder node.

        Args:
            node_id: Graph node id to replace (or role name: Backbone, Conjector).
            component_type: E.g. "sd15.unet", "sd15.scheduler".
        pretrained: HF repo id or local path for loading.
        config: Node config overrides.
        store: ModelStore instance (optional).
        variant: Model variant (e.g. "fp16").
        torch_dtype: Target dtype for loaded models.
        use_safetensors: If False, load .bin instead of .safetensors (needed for
            repos like Lykon/DreamShaper that have only diffusion_pytorch_model.bin).

        Returns:
            self for chaining.
        """
        node_id = self._resolve_role_to_node_id(node_id)
        if not is_component_type(component_type):
            raise ValueError(
                f"Expected component type (e.g. 'sd15.unet'), got '{component_type}'"
            )

        spec = resolve_component_type(component_type)
        family = component_type.split(".", 1)[0]
        family_spec = get_family_spec(family)

        cfg = dict(config or {})

        components_loaded: Dict[str, Any] = {}
        if pretrained and spec.load_keys:
            components_loaded = load_components_from_pretrained(
                spec.load_keys,
                pretrained,
                family=family,
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

            if len(spec.block_types) == 1:
                nid = node_id
            else:
                suffix = "_" + block_type.split("/")[-1].split("_", 1)[-1]
                nid = f"{node_id}{suffix}"

            if nid not in self._graph.node_ids:
                continue

            build_cfg: Dict[str, Any] = {
                "block_type": block_type,
                "node_id": nid,
                "config": kwargs.get("config", cfg),
            }
            for k, v in kwargs.items():
                if k not in ("config", "block_type", "node_id"):
                    build_cfg[k] = v

            new_node = reg.build(build_cfg)
            self._graph.replace_node(nid, node=new_node)

        return self
