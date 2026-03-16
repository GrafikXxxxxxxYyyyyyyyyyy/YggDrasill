from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional

from yggdrasill.engine.buffers import EdgeBuffers

# Port name convention for cycle-based loops (scheduler emits num_loop_steps)
_SCHEDULER_STATE_PORT = "scheduler_state"
from yggdrasill.engine.planner import build_plan
from yggdrasill.engine.validator import validate
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import PortAggregation


class ValidationError(Exception):
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")


@dataclass
class RunResult:
    """Extended result from :func:`run` when interrupt/resume is active."""

    outputs: Dict[str, Any]
    suspended: bool = False
    run_data: Optional[Dict[str, Dict[str, Any]]] = None


def run(
    structure: Any,
    inputs: Dict[str, Any],
    *,
    training: bool = False,
    num_loop_steps: Optional[int] = None,
    device: Optional[Any] = None,
    callbacks: Optional[List[Callable[..., None]]] = None,
    dry_run: bool = False,
    validate_before: bool = True,
    max_steps: Optional[int] = None,
    pin_data: Optional[Dict[str, Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    run_data: Optional[Dict[str, Dict[str, Any]]] = None,
    destination_node_id: Optional[str] = None,
    dirty_node_ids: Optional[List[str]] = None,
    interrupt_on: Optional[List[str]] = None,
) -> Dict[str, Any] | RunResult:
    """Execute the structure (Hypergraph, Workflow, etc.) and return outputs.

    Returns a plain ``dict`` unless *interrupt_on* fires, in which case a
    :class:`RunResult` is returned with ``suspended=True`` and a *run_data*
    snapshot that can be fed back to resume execution.
    """
    if validate_before:
        result = validate(structure)
        if not result.valid:
            raise ValidationError(result.errors)

    K = num_loop_steps
    if K is None:
        meta = getattr(structure, "metadata", {}) or {}
        K = meta.get("num_loop_steps", 1)

    agent_max = max_steps
    if agent_max is None:
        meta = getattr(structure, "metadata", {}) or {}
        agent_max = meta.get("max_steps", 10)

    input_spec = structure.get_input_spec()
    output_spec = structure.get_output_spec()

    buf = EdgeBuffers.init_from_inputs(input_spec, inputs)

    if run_data:
        for nid, port_values in run_data.items():
            if isinstance(port_values, dict):
                for pname, val in port_values.items():
                    buf.write(nid, pname, val)

    _prepare_nodes(structure, training, device, seed)

    plan = build_plan(structure)

    skip = _compute_skip_set(structure, plan, run_data, dirty_node_ids)

    cbs = callbacks or []
    pin = pin_data or {}
    interrupt_set = set(interrupt_on) if interrupt_on else set()

    for step_type, step_data in plan:
        if step_type == "node":
            nid = step_data
            if nid in skip:
                continue
            if nid in interrupt_set:
                return _make_suspended(buf, output_spec)
            _execute_node(structure, nid, buf, input_spec, dry_run, cbs, pin, training)
            if destination_node_id and nid == destination_node_id:
                break

        elif step_type == "cycle":
            rep, comp = step_data
            node_order = _cycle_node_order(structure, comp)
            K_use = _resolve_cycle_steps(structure, buf, K)
            _fire_callbacks(cbs, "loop_start", {"nodes": node_order, "steps": K_use})
            for step_idx in range(K_use):
                for nid in node_order:
                    if nid in skip:
                        continue
                    if nid in interrupt_set:
                        return _make_suspended(buf, output_spec)
                    _execute_node(structure, nid, buf, input_spec, dry_run, cbs, pin, training)
                _fire_callbacks(cbs, "cycle_step", {"step": step_idx, "total": K_use})
            _fire_callbacks(cbs, "loop_end", {"nodes": node_order, "steps": K_use})

        elif step_type == "agent_loop":
            nid = step_data
            if nid in skip:
                continue
            if nid in interrupt_set:
                return _make_suspended(buf, output_spec)
            _execute_agent_loop(
                structure, nid, buf, input_spec, dry_run, cbs, agent_max, pin,
                training,
            )
            if destination_node_id and nid == destination_node_id:
                break

    return _collect_outputs(output_spec, buf)


def run_stream(
    structure: Any,
    inputs: Dict[str, Any],
    *,
    training: bool = False,
    num_loop_steps: Optional[int] = None,
    device: Optional[Any] = None,
    callbacks: Optional[List[Callable[..., None]]] = None,
    dry_run: bool = False,
    validate_before: bool = True,
    max_steps: Optional[int] = None,
    pin_data: Optional[Dict[str, Dict[str, Any]]] = None,
    seed: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Streaming variant of :func:`run`.

    Yields intermediate output snapshots after each executed step.
    The **last** yielded value equals the normal ``run()`` return.
    """
    if validate_before:
        result = validate(structure)
        if not result.valid:
            raise ValidationError(result.errors)

    K = num_loop_steps
    if K is None:
        meta = getattr(structure, "metadata", {}) or {}
        K = meta.get("num_loop_steps", 1)

    agent_max = max_steps
    if agent_max is None:
        meta = getattr(structure, "metadata", {}) or {}
        agent_max = meta.get("max_steps", 10)

    input_spec = structure.get_input_spec()
    output_spec = structure.get_output_spec()

    buf = EdgeBuffers.init_from_inputs(input_spec, inputs)
    _prepare_nodes(structure, training, device, seed)

    plan = build_plan(structure)
    cbs = callbacks or []
    pin = pin_data or {}

    for step_type, step_data in plan:
        if step_type == "node":
            _execute_node(structure, step_data, buf, input_spec, dry_run, cbs, pin, training)
            yield _collect_outputs(output_spec, buf)

        elif step_type == "cycle":
            rep, comp = step_data
            node_order = _cycle_node_order(structure, comp)
            K_use = _resolve_cycle_steps(structure, buf, K)
            _fire_callbacks(cbs, "loop_start", {"nodes": node_order, "steps": K_use})
            for _it in range(K_use):
                for nid in node_order:
                    _execute_node(structure, nid, buf, input_spec, dry_run, cbs, pin, training)
                yield _collect_outputs(output_spec, buf)
            _fire_callbacks(cbs, "loop_end", {"nodes": node_order, "steps": K_use})

        elif step_type == "agent_loop":
            _execute_agent_loop(
                structure, step_data, buf, input_spec, dry_run, cbs, agent_max, pin,
                training,
            )
            yield _collect_outputs(output_spec, buf)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_cycle_steps(structure: Any, buf: EdgeBuffers, default: int) -> int:
    """Use scheduler_state.num_loop_steps when available (e.g. PNDM has 51 steps for 50 inference)."""
    for edge in structure.get_edges():
        if edge.source_port != _SCHEDULER_STATE_PORT:
            continue
        if buf.has(edge.source_node, _SCHEDULER_STATE_PORT):
            state = buf.read(edge.source_node, _SCHEDULER_STATE_PORT)
            if isinstance(state, dict):
                k = state.get("num_loop_steps")
                if isinstance(k, int) and k > 0:
                    return k
    return default


def _cycle_node_order(structure: Any, comp: Any) -> List[str]:
    """Order cycle nodes so dependencies run first.

    Ignores feedback edges (source_port starts with ``next_``) so that within
    a denoising loop, backbone (UNet) runs before solver (scheduler_step).
    """
    comp = set(comp)
    if len(comp) <= 1:
        return sorted(comp)
    edges = structure.get_edges()
    pred: Dict[str, Set[str]] = {n: set() for n in comp}
    for e in edges:
        if e.source_node not in comp or e.target_node not in comp or e.source_node == e.target_node:
            continue
        if e.source_port.startswith("next_"):
            continue
        pred[e.target_node].add(e.source_node)
    order: List[str] = []
    remaining = set(comp)
    while remaining:
        ready = [n for n in remaining if pred[n] <= set(order)]
        if not ready:
            break
        order.append(min(ready))
        remaining.discard(order[-1])
    if not remaining:
        return order
    order.extend(sorted(remaining))
    return order


def _prepare_nodes(
    structure: Any, training: bool, device: Any, seed: Optional[int] = None,
) -> None:
    for nid in structure.node_ids:
        node = structure.get_node(nid)
        if node is None:
            continue
        if hasattr(node, "train") and callable(node.train):
            node.train(training)
        if device is not None and hasattr(node, "to") and callable(node.to):
            node.to(device)
        if seed is not None and hasattr(node, "seed"):
            try:
                node.seed = seed
            except (AttributeError, TypeError):
                pass


def _gather_node_inputs(
    structure: Any,
    node_id: str,
    buf: EdgeBuffers,
    input_spec: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collect all inputs for *node_id* from edge buffers and exposed input spec."""
    node = structure.get_node(node_id)
    if node is None:
        return {}

    node_inputs: Dict[str, Any] = {}

    in_edges = structure.get_edges_in(node_id)

    port_edge_counts: Dict[str, int] = {}
    for edge in in_edges:
        port_edge_counts[edge.target_port] = port_edge_counts.get(edge.target_port, 0) + 1

    for target_port, count in port_edge_counts.items():
        if count == 1:
            edge = next(e for e in in_edges if e.target_port == target_port)
            if buf.has(edge.source_node, edge.source_port):
                node_inputs[target_port] = buf.read(edge.source_node, edge.source_port)
        else:
            agg_policy = _get_aggregation(node, target_port)
            buf.clear_multi(node_id, target_port)
            # Sort edges so feedback (next_*) sources come last for SINGLE aggregation.
            # Denoising loop: latent_init provides initial; sched_step provides next_latent.
            # values[-1] must be next_* when both exist.
            port_edges = [e for e in in_edges if e.target_port == target_port]
            port_edges = sorted(
                port_edges,
                key=lambda e: (1 if e.source_port.startswith("next_") else 0, e.source_node),
            )
            for edge in port_edges:
                if buf.has(edge.source_node, edge.source_port):
                    buf.append(
                        node_id, target_port,
                        buf.read(edge.source_node, edge.source_port),
                        source_node=edge.source_node,
                    )
            if buf.has_multi(node_id, target_port):
                node_inputs[target_port] = buf.aggregate(node_id, target_port, agg_policy)

    for entry in input_spec:
        nid = entry.get("node_id") or entry.get("graph_id")
        if nid == node_id:
            pname = entry["port_name"]
            if pname not in node_inputs:
                if buf.has(nid, pname):
                    node_inputs[pname] = buf.read(nid, pname)

    return node_inputs


def _execute_node(
    structure: Any,
    node_id: str,
    buf: EdgeBuffers,
    input_spec: List[Dict[str, Any]],
    dry_run: bool,
    callbacks: List[Callable[..., None]],
    pin_data: Dict[str, Dict[str, Any]],
    training: bool = False,
) -> None:
    node = structure.get_node(node_id)
    if node is None:
        return

    if node_id in pin_data:
        for pname, val in pin_data[node_id].items():
            buf.write(node_id, pname, val)
        _fire_callbacks(callbacks, "pinned", {"node_id": node_id})
        return

    node_inputs = _gather_node_inputs(structure, node_id, buf, input_spec)

    _fire_callbacks(callbacks, "before", {"node_id": node_id})

    if dry_run:
        node_outputs: Dict[str, Any] = {}
        if hasattr(node, "get_output_ports"):
            for port in node.get_output_ports():
                node_outputs[port.name] = None
        elif hasattr(node, "get_output_spec"):
            for entry in node.get_output_spec():
                node_outputs[entry.get("name") or entry["port_name"]] = None
    else:
        if training:
            node_outputs = node.run(node_inputs)
        else:
            import torch
            with torch.inference_mode():
                node_outputs = node.run(node_inputs)

    for port_name, value in node_outputs.items():
        buf.write(node_id, port_name, value)

    _fire_callbacks(callbacks, "after", {"node_id": node_id})


def _execute_agent_loop(
    structure: Any,
    node_id: str,
    buf: EdgeBuffers,
    input_spec: List[Dict[str, Any]],
    dry_run: bool,
    callbacks: List[Callable[..., None]],
    max_steps: int,
    pin_data: Dict[str, Dict[str, Any]],
    training: bool = False,
) -> None:
    """Run an agent node in a tool_calls sub-loop."""
    meta = getattr(structure, "metadata", {}) or {}
    tool_map: Dict[str, str] = meta.get("tool_id_to_node_id", {})

    node = structure.get_node(node_id)
    if node is None:
        return

    if node_id in pin_data:
        for pname, val in pin_data[node_id].items():
            buf.write(node_id, pname, val)
        _fire_callbacks(callbacks, "pinned", {"node_id": node_id})
        return

    base_inputs = _gather_node_inputs(structure, node_id, buf, input_spec)

    steps_taken = 0
    for step in range(max_steps):
        steps_taken += 1
        _fire_callbacks(callbacks, "agent_step", {"node_id": node_id, "step": step})

        if dry_run:
            outputs: Dict[str, Any] = {}
            if hasattr(node, "get_output_ports"):
                for port in node.get_output_ports():
                    outputs[port.name] = None
        else:
            outputs = node.run(base_inputs)

        for pname, val in outputs.items():
            buf.write(node_id, pname, val)

        tool_calls = outputs.get("tool_calls")
        if not tool_calls:
            break

        tool_results: List[Dict[str, Any]] = []
        for tc in tool_calls:
            tid = tc.get("tool_id") or tc.get("function", {}).get("name", "")
            tool_nid = tool_map.get(tid)
            if tool_nid is None:
                continue
            tool_node = structure.get_node(tool_nid)
            if tool_node is None:
                continue
            tool_args = tc.get("arguments") or tc.get("args", {})

            _fire_callbacks(callbacks, "tool_call", {"tool_id": tid, "node_id": tool_nid})

            if dry_run:
                tool_out: Dict[str, Any] = {}
            else:
                tool_out = tool_node.run(tool_args)

            tool_results.append({
                "tool_call_id": tc.get("id", tid),
                "content": tool_out,
            })

        base_inputs["tool_results"] = tool_results

    _fire_callbacks(callbacks, "agent_loop_done", {
        "node_id": node_id, "steps": steps_taken,
    })


def _compute_skip_set(
    structure: Any,
    plan: List[tuple],
    run_data: Optional[Dict[str, Dict[str, Any]]],
    dirty_node_ids: Optional[List[str]],
) -> set:
    """Determine which nodes to skip during partial run."""
    if dirty_node_ids is None:
        return set()

    dirty = set(dirty_node_ids)
    edges = structure.get_edges()
    adj: Dict[str, set] = {nid: set() for nid in structure.node_ids}
    for edge in edges:
        adj.setdefault(edge.source_node, set()).add(edge.target_node)

    to_run: set = set()
    queue = list(dirty)
    while queue:
        nid = queue.pop()
        if nid in to_run:
            continue
        to_run.add(nid)
        for dep in adj.get(nid, ()):
            if dep not in to_run:
                queue.append(dep)

    all_plan_nodes: set = set()
    for _, data in plan:
        if isinstance(data, str):
            all_plan_nodes.add(data)
        elif isinstance(data, tuple) and len(data) == 2:
            _, comp = data
            if isinstance(comp, frozenset):
                all_plan_nodes |= comp

    return all_plan_nodes - to_run


def _make_suspended(buf: EdgeBuffers, output_spec: List[Dict[str, Any]]) -> RunResult:
    """Create a suspended RunResult with buffer snapshot."""
    return RunResult(
        outputs=_collect_outputs(output_spec, buf),
        suspended=True,
        run_data=buf.snapshot(),
    )


def _collect_outputs(
    output_spec: List[Dict[str, Any]], buf: EdgeBuffers,
) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    for entry in output_spec:
        key = _spec_key(entry)
        nid = entry.get("node_id") or entry.get("graph_id")
        pname = entry["port_name"]
        outputs[key] = buf.read(nid, pname)
    return outputs


def _get_aggregation(node: Any, port_name: str) -> PortAggregation:
    """Retrieve the aggregation policy for *port_name* on *node*."""
    if isinstance(node, AbstractGraphNode):
        port = node.get_port(port_name)
        if port is not None:
            return port.aggregation
    return PortAggregation.SINGLE


def _spec_key(entry: Dict[str, Any]) -> str:
    if "name" in entry and entry["name"] is not None:
        return entry["name"]
    nid = entry.get("node_id") or entry.get("graph_id", "")
    return f"{nid}:{entry['port_name']}"


def _fire_callbacks(
    callbacks: List[Callable[..., None]], phase: str, info: Dict[str, Any],
) -> None:
    for cb in callbacks:
        try:
            cb(phase, info)
        except Exception as exc:
            warnings.warn(
                f"Callback raised {type(exc).__name__}: {exc} (phase={phase!r})",
                UserWarning,
                stacklevel=2,
            )
