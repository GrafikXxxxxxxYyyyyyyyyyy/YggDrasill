from __future__ import annotations

import bisect
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

_MAX_CACHE_SIZE = 256
_plan_cache: OrderedDict[Tuple[str, int, int], List[Tuple[str, Any]]] = OrderedDict()


def build_plan(
    structure: Any,
    *,
    subgraph_node_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, Any]]:
    """Build an execution plan for the given structure.

    Training-phase plans (forward / loss / backward / optim) are separate:
    see :func:`yggdrasill.training.plan.build_training_plan`.

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
