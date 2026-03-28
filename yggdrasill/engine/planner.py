from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

_MAX_CACHE_SIZE = 256
_plan_cache: OrderedDict[Tuple[str, int, int], List[Tuple[str, Any]]] = OrderedDict()


@dataclass(frozen=True)
class TrainingPlan:
    """Forward order through the loss sink and post-backward helper nodes."""

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


def _nodes_reaching_loss(
    structure: Any,
    *,
    loss_nid: str,
    candidates: Set[str],
) -> Set[str]:
    """Nodes in *candidates* that lie on a directed path to *loss_nid* (including *loss_nid*)."""
    preds: Dict[str, Set[str]] = {n: set() for n in candidates}
    for e in structure.get_edges():
        if e.source_node in candidates and e.target_node in candidates:
            preds[e.target_node].add(e.source_node)
    if loss_nid not in candidates:
        return set()
    reachable: Set[str] = {loss_nid}
    stack = [loss_nid]
    while stack:
        cur = stack.pop()
        for p in preds.get(cur, ()):
            if p not in reachable:
                reachable.add(p)
                stack.append(p)
    return reachable


def _topo_sort_forward(
    structure: Any,
    *,
    nodes: Set[str],
) -> List[str]:
    """Topological order of *nodes* using edges with both ends in *nodes*."""
    preds: Dict[str, Set[str]] = {n: set() for n in nodes}
    succ: Dict[str, Set[str]] = {n: set() for n in nodes}
    for e in structure.get_edges():
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
    """Derive forward order, loss sink, and post-backward helper chain from metadata.

    Lives next to :func:`build_plan` so inference and training planning stay co-located.
    When ``forward_node_ids`` is omitted, forward nodes default to those on paths to the
    loss node (subgraph toward ``loss_node_id``), then topologically sorted.
    """
    spec = _parse_training_spec(structure)
    loss_nid = spec.get("loss_node_id")
    if not loss_nid or not isinstance(loss_nid, str):
        raise ValueError("metadata['training']['loss_node_id'] must be a non-empty string")
    loss_port = spec.get("loss_port") or "loss"
    if loss_nid not in structure.node_ids:
        raise ValueError(f"loss_node_id {loss_nid!r} is not in the graph")

    tni = spec.get("trainable_node_ids")
    if tni is not None:
        for nid in tni:
            s = str(nid)
            if s not in structure.node_ids:
                raise ValueError(f"trainable_node_ids entry {s!r} is not in the graph")

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
        toward_loss = _nodes_reaching_loss(structure, loss_nid=loss_nid, candidates=forward_nodes)
        if toward_loss and toward_loss != forward_nodes:
            forward_nodes = toward_loss
        order = _topo_sort_forward(structure, nodes=forward_nodes)
        if order[-1] != loss_nid:
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


def build_plan(
    structure: Any,
    *,
    subgraph_node_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, Any]]:
    """Build an execution plan for the given structure.

    Training-phase plans (forward / loss / backward / optim): see
    :func:`build_training_plan` in this module.

    Returns a list of steps:
      ("node", node_id)               -- execute one node once
      ("cycle", (rep, frozenset(ids))) -- execute nodes K times (K from options at run-time)
      ("agent_loop", node_id)          -- agent with tool_calls sub-loop

    Agent nodes are detected from three sources (any match is sufficient):
      1. ``metadata["agent_node_ids"]`` — explicit list.
      2. Per-node ``is_agent`` attribute (duck-typed).
      3. ``graph_kind == "agent"`` on the structure — backbone-role nodes
         become agents automatically.

    If *subgraph_node_ids* is provided the plan is restricted to that subset.
    Subgraph plans are **not** cached.
    """
    use_cache = subgraph_node_ids is None

    instance_id = getattr(structure, "_instance_id", id(structure))
    type_tag = type(structure).__name__
    cache_key = (type_tag, instance_id, structure.execution_version)
    if use_cache and cache_key in _plan_cache:
        return _plan_cache[cache_key]

    meta = getattr(structure, "metadata", {}) or {}

    tool_map: Dict[str, str] = meta.get("tool_id_to_node_id", {})
    tool_nids = set(tool_map.values())

    agent_ids = set(meta.get("agent_node_ids", []))

    graph_kind = getattr(structure, "graph_kind", None) or meta.get("graph_kind")

    for nid in structure.node_ids:
        if nid in tool_nids:
            continue
        node = structure.get_node(nid)
        if node is None:
            continue
        if getattr(node, "is_agent", False):
            agent_ids.add(nid)
        if graph_kind == "agent":
            role = getattr(node, "role", None)
            if role is not None and getattr(role, "value", None) == "backbone":
                agent_ids.add(nid)

    node_ids = sorted(nid for nid in structure.node_ids if nid not in tool_nids)

    if subgraph_node_ids is not None:
        allowed = subgraph_node_ids - tool_nids
        node_ids = [nid for nid in node_ids if nid in allowed]

    edges = structure.get_edges()

    adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
    for edge in edges:
        if edge.source_node in adj and edge.target_node in adj:
            adj[edge.source_node].add(edge.target_node)

    sccs = _tarjan(node_ids, adj)

    scc_order = _topo_sort_sccs(sccs, adj)

    has_self_loop = set()
    for edge in edges:
        if edge.source_node == edge.target_node:
            has_self_loop.add(edge.source_node)

    plan: List[Tuple[str, Any]] = []
    for comp in scc_order:
        if len(comp) == 1:
            nid = next(iter(comp))
            if nid in agent_ids:
                plan.append(("agent_loop", nid))
            elif nid in has_self_loop:
                plan.append(("cycle", (nid, frozenset(comp))))
            else:
                plan.append(("node", nid))
        else:
            rep = sorted(comp)[0]
            plan.append(("cycle", (rep, frozenset(comp))))

    if use_cache:
        if len(_plan_cache) >= _MAX_CACHE_SIZE:
            _plan_cache.popitem(last=False)
        _plan_cache[cache_key] = plan
    return plan


def clear_plan_cache() -> None:
    _plan_cache.clear()


# ---------------------------------------------------------------------------
# Tarjan's SCC algorithm
# ---------------------------------------------------------------------------

def _tarjan(node_ids: List[str], adj: Dict[str, Set[str]]) -> List[Set[str]]:
    index_counter = [0]
    stack: List[str] = []
    on_stack: Set[str] = set()
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    result: List[Set[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            comp: Set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.add(w)
                if w == v:
                    break
            result.append(comp)

    for v in node_ids:
        if v not in index:
            strongconnect(v)

    return result


def _topo_sort_sccs(
    sccs: List[Set[str]], adj: Dict[str, Set[str]],
) -> List[Set[str]]:
    """Topologically sort SCCs (sources first)."""
    scc_of: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for nid in comp:
            scc_of[nid] = i

    n = len(sccs)
    scc_adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    in_deg: Dict[int, int] = {i: 0 for i in range(n)}

    for src, targets in adj.items():
        si = scc_of.get(src)
        if si is None:
            continue
        for tgt in targets:
            ti = scc_of.get(tgt)
            if ti is None or ti == si:
                continue
            if ti not in scc_adj[si]:
                scc_adj[si].add(ti)
                in_deg[ti] += 1

    queue = sorted([i for i in range(n) if in_deg[i] == 0])
    order: List[int] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in sorted(scc_adj[cur]):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                bisect.insort(queue, nxt)

    return [sccs[i] for i in order]
