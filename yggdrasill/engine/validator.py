from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortDirection, PortType


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


def validate(structure: Any) -> ValidationResult:
    """Validate an executable structure (Hypergraph, Workflow, etc.)."""
    result = ValidationResult()
    node_ids = structure.node_ids
    edges = structure.get_edges()

    for edge in edges:
        if edge.source_node not in node_ids:
            result.errors.append(
                f"Edge references unknown source node '{edge.source_node}'"
            )
            continue
        if edge.target_node not in node_ids:
            result.errors.append(
                f"Edge references unknown target node '{edge.target_node}'"
            )
            continue

        src_node = structure.get_node(edge.source_node)
        dst_node = structure.get_node(edge.target_node)

        src_port = _find_port(src_node, edge.source_port, PortDirection.OUT)
        if src_port is None:
            result.errors.append(
                f"Output port '{edge.source_port}' not found on node '{edge.source_node}'"
            )
        dst_port = _find_port(dst_node, edge.target_port, PortDirection.IN)
        if dst_port is None:
            result.errors.append(
                f"Input port '{edge.target_port}' not found on node '{edge.target_node}'"
            )

        if src_port is not None and dst_port is not None:
            if not src_port.compatible_with(dst_port):
                result.errors.append(
                    f"Incompatible types: {edge.source_node}.{edge.source_port} "
                    f"({src_port.dtype}) -> {edge.target_node}.{edge.target_port} "
                    f"({dst_port.dtype})"
                )

    # Check required inputs are covered
    for nid in node_ids:
        node = structure.get_node(nid)
        if not isinstance(node, AbstractGraphNode):
            continue
        in_edges = structure.get_edges_in(nid)
        covered_ports = {e.target_port for e in in_edges}
        exposed_ports = set()
        for entry in structure.get_input_spec():
            entry_nid = entry.get("node_id") or entry.get("graph_id")
            if entry_nid == nid:
                exposed_ports.add(entry["port_name"])

        for port in node.get_input_ports():
            if not port.optional and port.name not in covered_ports and port.name not in exposed_ports:
                result.errors.append(
                    f"Required input port '{port.name}' on node '{nid}' "
                    f"has no incoming edge and is not an exposed input"
                )

    for entry in structure.get_input_spec():
        nid = entry.get("node_id") or entry.get("graph_id")
        pname = entry.get("port_name")
        if nid not in node_ids:
            result.errors.append(f"Exposed input references unknown node '{nid}'")
            continue
        node = structure.get_node(nid)
        port = _find_port(node, pname, PortDirection.IN)
        if port is None:
            result.errors.append(
                f"Exposed input port '{pname}' not found as input on node '{nid}'"
            )

    for entry in structure.get_output_spec():
        nid = entry.get("node_id") or entry.get("graph_id")
        pname = entry.get("port_name")
        if nid not in node_ids:
            result.errors.append(f"Exposed output references unknown node '{nid}'")
            continue
        node = structure.get_node(nid)
        port = _find_port(node, pname, PortDirection.OUT)
        if port is None:
            result.errors.append(
                f"Exposed output port '{pname}' not found as output on node '{nid}'"
            )

    _check_cycles(structure, result)

    _validate_training_graph_metadata(structure, result)

    return result


def _validate_training_graph_metadata(structure: Any, result: ValidationResult) -> None:
    meta = getattr(structure, "metadata", None) or {}
    spec = meta.get("training")
    if spec is None:
        return
    if not isinstance(spec, dict):
        result.errors.append("metadata['training'] must be a dict when present")
        return
    loss_nid = spec.get("loss_node_id")
    if not loss_nid or not isinstance(loss_nid, str):
        result.errors.append("metadata['training']['loss_node_id'] must be a non-empty string")
        return
    if loss_nid not in structure.node_ids:
        result.errors.append(
            f"metadata['training']['loss_node_id'] {loss_nid!r} is not a graph node"
        )
    post = spec.get("post_backward_node_ids") or ()
    if not isinstance(post, (list, tuple)):
        result.errors.append("metadata['training']['post_backward_node_ids'] must be a list/tuple")
        return
    for nid in post:
        if str(nid) not in structure.node_ids:
            result.errors.append(
                f"metadata['training']['post_backward_node_ids'] entry {nid!r} is not a graph node"
            )
    fwd = spec.get("forward_node_ids")
    if fwd is not None:
        if not isinstance(fwd, (list, tuple)) or not fwd:
            result.errors.append("metadata['training']['forward_node_ids'] must be a non-empty list")
            return
        for nid in fwd:
            if str(nid) not in structure.node_ids:
                result.errors.append(
                    f"metadata['training']['forward_node_ids'] entry {nid!r} is not a graph node"
                )
        if str(fwd[-1]) != str(loss_nid):
            result.errors.append("metadata['training']['forward_node_ids'] must end with loss_node_id")


def _check_cycles(structure: Any, result: ValidationResult) -> None:
    """Detect cycles (including self-loops) via Tarjan SCC; warn if num_loop_steps is unset."""
    from yggdrasill.engine.planner import _tarjan

    node_ids = sorted(structure.node_ids)
    edges = structure.get_edges()
    adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
    self_loop_nodes: Set[str] = set()
    for edge in edges:
        if edge.source_node in adj:
            adj[edge.source_node].add(edge.target_node)
        if edge.source_node == edge.target_node:
            self_loop_nodes.add(edge.source_node)

    all_sccs = _tarjan(node_ids, adj)
    sccs = [list(scc) for scc in all_sccs if len(scc) > 1]

    for nid in self_loop_nodes:
        if not any(nid in scc for scc in sccs):
            sccs.append([nid])

    if sccs:
        meta = getattr(structure, "metadata", {})
        if meta.get("num_loop_steps") is None:
            for scc in sccs:
                result.warnings.append(
                    f"Cycle detected among nodes {sorted(scc)} but "
                    f"metadata['num_loop_steps'] is not set"
                )


def _find_port(node: Any, port_name: str, direction: PortDirection):
    """Try to find a port on a node, supporting both task-nodes and hypergraph-level nodes."""
    if isinstance(node, AbstractGraphNode):
        port = node.get_port(port_name)
        if port is not None and port.direction == direction:
            return port
        return None

    # For workflow-level nodes (Hypergraph objects), check spec
    if direction == PortDirection.OUT:
        spec = getattr(node, "get_output_spec", lambda **_k: [])(include_dtype=True)
    else:
        spec = getattr(node, "get_input_spec", lambda **_k: [])(include_dtype=True)

    for entry in spec:
        key = entry.get("port_name") or entry.get("name", "")
        if key == port_name:
            dtype_str = entry.get("dtype", "any")
            try:
                dtype = PortType(dtype_str)
            except ValueError:
                dtype = PortType.ANY
            return Port(name=port_name, direction=direction, dtype=dtype)
    return None
