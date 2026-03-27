"""Training execution plan: forward node order, loss sink, post-backward helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

@dataclass(frozen=True)
class TrainingPlan:
    forward_node_ids: Tuple[str, ...]
    loss_node_id: str
    loss_port: str
    post_backward_node_ids: Tuple[str, ...]


def _parse_training_spec(structure: Any) -> Dict[str, Any]:
    meta = getattr(structure, "metadata", None) or {}
    spec = meta.get("training")
    if not isinstance(spec, dict) or not spec:
        raise ValueError(
            "Training graphs require structure.metadata['training'] to be a non-empty dict "
            "with at least 'loss_node_id'."
        )
    return spec


def _topo_sort_forward(
    structure: Any,
    *,
    nodes: Set[str],
) -> List[str]:
    """Topological order of *nodes* using edges with both ends in *nodes*."""
    edges = structure.get_edges()
    preds: Dict[str, Set[str]] = {n: set() for n in nodes}
    succ: Dict[str, Set[str]] = {n: set() for n in nodes}
    for e in edges:
        if e.source_node in nodes and e.target_node in nodes:
            preds[e.target_node].add(e.source_node)
            succ[e.source_node].add(e.target_node)

    ready = sorted(n for n in nodes if not preds[n])
    order: List[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for nxt in sorted(succ.get(nid, ())):
            preds[nxt].discard(nid)
            if not preds[nxt] and nxt not in order and nxt not in ready:
                ready.append(nxt)
        ready.sort()

    if len(order) != len(nodes):
        remaining = nodes - set(order)
        raise ValueError(
            f"Training forward subgraph has a cycle or unresolved order among {sorted(remaining)}"
        )
    return order


def build_training_plan(structure: Any) -> TrainingPlan:
    """Derive forward order, loss sink, and post-backward helper chain from metadata."""
    spec = _parse_training_spec(structure)
    loss_nid = spec.get("loss_node_id")
    if not loss_nid or not isinstance(loss_nid, str):
        raise ValueError("metadata['training']['loss_node_id'] must be a non-empty string")
    loss_port = spec.get("loss_port") or "loss"
    if loss_nid not in structure.node_ids:
        raise ValueError(f"loss_node_id {loss_nid!r} is not in the graph")

    post_raw = spec.get("post_backward_node_ids") or ()
    post_ids = tuple(str(x) for x in post_raw)
    for nid in post_ids:
        if nid not in structure.node_ids:
            raise ValueError(f"post_backward node {nid!r} is not in the graph")

    post_set = set(post_ids)
    if loss_nid in post_set:
        raise ValueError("loss_node_id must not appear in post_backward_node_ids")

    explicit = spec.get("forward_node_ids")
    if explicit is not None:
        fwd = [str(x) for x in explicit]
        for nid in fwd:
            if nid not in structure.node_ids:
                raise ValueError(f"forward_node_ids entry {nid!r} is not in the graph")
            if nid in post_set:
                raise ValueError(f"forward_node_ids entry {nid!r} must not be a post_backward node")
        if loss_nid not in fwd:
            raise ValueError("forward_node_ids must include loss_node_id")
        if fwd[-1] != loss_nid:
            raise ValueError("forward_node_ids must end with loss_node_id (loss is the forward sink)")
        forward = tuple(fwd)
    else:
        forward_nodes = {n for n in structure.node_ids if n not in post_set}
        order = _topo_sort_forward(structure, nodes=forward_nodes)
        if order[-1] != loss_nid:
            # Allow loss not last in topo if user should use explicit forward_node_ids
            raise ValueError(
                f"Topological order ends with {order[-1]!r} but loss_node_id is {loss_nid!r}. "
                "Set metadata['training']['forward_node_ids'] explicitly (must end with loss node)."
            )
        forward = tuple(order)

    return TrainingPlan(
        forward_node_ids=forward,
        loss_node_id=loss_nid,
        loss_port=str(loss_port),
        post_backward_node_ids=post_ids,
    )


def training_plan_signature(structure: Any) -> str:
    """Short string for checkpoint tagging (graph id + training slice)."""
    gid = getattr(structure, "graph_id", "graph")
    p = build_training_plan(structure)
    return f"{gid}:{','.join(p.forward_node_ids)}:{p.loss_node_id}"
