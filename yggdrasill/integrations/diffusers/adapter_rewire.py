"""Re-wire adapter nodes after the graph gains a latent / scheduler loop.

``DiffusionGraphBuilder`` often adds ControlNet / IP-Adapter *before*
``try_complete_*_universal_diffusion`` injects ``latent_init`` and loop edges.
Port-name auto-connect only sees nodes present at add time, so we re-run it
once the latent stack exists.
"""
from __future__ import annotations

from typing import Any

from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect


def rewire_adapters_after_latent_stack(graph: Any) -> None:
    """Connect ``adapter/controlnet`` and ``adapter/ip_adapter`` to new latent/timestep sources."""
    for nid in list(getattr(graph, "node_ids", ()) or ()):
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if not isinstance(node, AbstractGraphNode):
            continue
        bt = getattr(node, "block_type", "") or ""
        if bt in ("adapter/controlnet", "adapter/ip_adapter"):
            apply_port_name_auto_connect(graph, nid, node)
