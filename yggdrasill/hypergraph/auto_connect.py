"""Auto-connect: automatically create edges when adding a task-node.

Based on PHASE_4 §9 and Canon 02 §12.2.

Two strategies are provided:

1. **Role-based** (:func:`apply_auto_connect`) — uses the canonical role
   edge rules (``ROLE_EDGE_RULES``) with generic port names like
   ``"condition"`` / ``"latent"``.  Best for custom non-diffusion graphs.

2. **Port-name-based** (:func:`apply_port_name_auto_connect`) — matches
   output ports to input ports by *exact name* (with a small alias
   table for known mismatches).  Best for diffusion graphs whose ports
   follow the canonical contract names.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from yggdrasill.engine.edge import Edge
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import PortDirection
from yggdrasill.task_nodes.role_rules import suggest_edges_for_new_node
from yggdrasill.task_nodes.roles import role_from_block_type


# ── Role-based auto-connect (original) ─────────────────────────────────

def apply_auto_connect(
    hypergraph: Any,
    new_node_id: str,
    new_block: Any,
) -> int:
    """Create edges for *new_node_id* based on role rules.

    Returns the number of edges actually added.
    """
    bt = getattr(new_block, "block_type", None)
    if bt is None:
        return 0
    new_role = role_from_block_type(bt)
    if new_role is None:
        return 0

    existing: Dict[str, str] = {}
    for nid in hypergraph.node_ids:
        if nid == new_node_id:
            continue
        node = hypergraph.get_node(nid)
        node_bt = getattr(node, "block_type", None)
        if node_bt is None:
            continue
        r = role_from_block_type(node_bt)
        if r is not None:
            existing[nid] = r

    suggestions = suggest_edges_for_new_node(new_node_id, new_role, existing)
    added = 0
    for src_nid, src_port, tgt_nid, tgt_port in suggestions:
        src_node = hypergraph.get_node(src_nid)
        tgt_node = hypergraph.get_node(tgt_nid)
        src_port_obj = None
        tgt_port_obj = None
        if isinstance(src_node, AbstractGraphNode):
            src_port_obj = src_node.get_port(src_port)
            if src_port_obj is None:
                continue
        if isinstance(tgt_node, AbstractGraphNode):
            tgt_port_obj = tgt_node.get_port(tgt_port)
            if tgt_port_obj is None:
                continue
        if src_port_obj is not None and tgt_port_obj is not None:
            if not src_port_obj.compatible_with(tgt_port_obj):
                continue
        edge = Edge(src_nid, src_port, tgt_nid, tgt_port)
        try:
            hypergraph.add_edge(edge)
            added += 1
        except (ValueError, KeyError):
            pass
    return added


# ── Port-name-based auto-connect (diffusion-aware) ─────────────────────

PORT_ALIASES: Dict[str, List[str]] = {
    "next_latent": ["latents"],
    "next_timestep": ["timestep"],
}


def _allows_self_connect(block: Any) -> bool:
    """Nodes with next_timestep out and timestep in can self-connect for the denoising loop."""
    if not isinstance(block, AbstractGraphNode):
        return False
    out_names = {p.name for p in block.get_output_ports() if p.direction == PortDirection.OUT}
    in_names = {p.name for p in block.get_input_ports() if p.direction == PortDirection.IN}
    return "next_timestep" in out_names and "timestep" in in_names


def _get_existing_edges(hypergraph: Any) -> Set[Tuple[str, str, str, str]]:
    """Return a set of (src_node, src_port, tgt_node, tgt_port) tuples."""
    return {
        (e.source_node, e.source_port, e.target_node, e.target_port)
        for e in hypergraph.get_edges()
    }


def _is_latent_init(block: Any) -> bool:
    """True if block produces initial noisy latents (for the denoising loop only)."""
    bt = getattr(block, "block_type", "") or ""
    return "latent_init" in bt


def _is_vae_decode(block: Any) -> bool:
    """True if block decodes final latents to image."""
    bt = getattr(block, "block_type", "") or ""
    return "vae_decode" in bt


def _should_skip_edge(
    src_block: Any,
    src_port: str,
    tgt_block: Any,
    tgt_port: str,
) -> bool:
    """Skip edges that are semantically wrong despite matching port names.

    latent_init outputs noisy latents for the denoising loop; vae_decode must
    receive only the final denoised latents from the scheduler step.
    """
    if src_port not in ("latents", "next_latent"):
        return False
    if _is_latent_init(src_block) and _is_vae_decode(tgt_block):
        return True
    return False


def apply_port_name_auto_connect(
    hypergraph: Any,
    new_node_id: str,
    new_block: Any,
) -> int:
    """Wire *new_node_id* to existing nodes by matching port names.

    For every **output** port on the new node, looks for existing nodes
    that have an **input** port with the same name (or a known alias).
    For every **input** port on the new node, looks for existing nodes
    that have an **output** port with the same name (or alias).

    Returns the number of edges added.
    """
    if not isinstance(new_block, AbstractGraphNode):
        return 0

    existing_edges = _get_existing_edges(hypergraph)
    added = 0

    new_out_ports = [p for p in new_block.get_output_ports() if p.direction == PortDirection.OUT]
    new_in_ports = [p for p in new_block.get_input_ports() if p.direction == PortDirection.IN]

    for nid in list(hypergraph.node_ids):
        other = hypergraph.get_node(nid)
        if nid == new_node_id and not _allows_self_connect(new_block):
            continue
        if not isinstance(other, AbstractGraphNode):
            continue

        other_in_ports = {p.name: p for p in other.get_input_ports() if p.direction == PortDirection.IN}
        other_out_ports = {p.name: p for p in other.get_output_ports() if p.direction == PortDirection.OUT}

        for out_port in new_out_ports:
            target_names = [out_port.name] + PORT_ALIASES.get(out_port.name, [])
            for tgt_name in target_names:
                if tgt_name in other_in_ports:
                    in_port = other_in_ports[tgt_name]
                    if not out_port.compatible_with(in_port):
                        continue
                    if _should_skip_edge(new_block, out_port.name, other, tgt_name):
                        continue
                    edge_key = (new_node_id, out_port.name, nid, tgt_name)
                    if edge_key in existing_edges:
                        continue
                    edge = Edge(new_node_id, out_port.name, nid, tgt_name)
                    try:
                        hypergraph.add_edge(edge)
                        existing_edges.add(edge_key)
                        added += 1
                    except (ValueError, KeyError):
                        pass

        for in_port in new_in_ports:
            target_names = [in_port.name]
            reverse_aliases = [
                alias for alias, targets in PORT_ALIASES.items()
                if in_port.name in targets
            ]
            target_names.extend(reverse_aliases)
            for src_name in target_names:
                if src_name in other_out_ports:
                    out_port_obj = other_out_ports[src_name]
                    if not out_port_obj.compatible_with(in_port):
                        continue
                    if _should_skip_edge(other, src_name, new_block, in_port.name):
                        continue
                    edge_key = (nid, src_name, new_node_id, in_port.name)
                    if edge_key in existing_edges:
                        continue
                    edge = Edge(nid, src_name, new_node_id, in_port.name)
                    try:
                        hypergraph.add_edge(edge)
                        existing_edges.add(edge_key)
                        added += 1
                    except (ValueError, KeyError):
                        pass

    return added


# ── Public helpers ──────────────────────────────────────────────────────

def use_task_node_auto_connect(hypergraph: Any) -> None:
    """Enable role-based auto-connect on *hypergraph*.

    After calling this, ``hypergraph.add_node_from_config(..., auto_connect=True)``
    will automatically create edges based on role rules.
    """
    hypergraph._auto_connect_fn = apply_auto_connect
