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

    @property
    def graph(self) -> Hypergraph:
        return self._graph

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
                variant=variant or family_spec.torch_dtype_default,
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

        return self

    def add_node(self, node_id: str, node: Any) -> "DiffusionGraphBuilder":
        """Add a pre-built node directly."""
        self._graph.add_node(node_id, node)
        return self

    def add_edge(self, source: str, source_port: str, target: str, target_port: str) -> "DiffusionGraphBuilder":
        """Add an edge. Uses contract port names when passed as strings."""
        self._graph.add_edge(Edge(source, source_port, target, target_port))
        return self
