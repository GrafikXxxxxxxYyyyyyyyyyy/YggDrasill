"""Advanced engine feature tests — agent detection, subgraph plans, pin_data
edge cases, interrupt/resume, streaming, seed, partial run, and buffer snapshots.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from yggdrasill.engine.buffers import EdgeBuffers
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.executor import RunResult, ValidationError, run, run_stream
from yggdrasill.engine.planner import build_plan, clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortDirection, PortType
from tests.engine.helpers import make_chain, make_cycle
from tests.foundation.helpers import IdentityTaskNode


# ---------------------------------------------------------------------------
# Stub: node with is_agent attribute
# ---------------------------------------------------------------------------

class _AgentNode(AbstractBaseBlock, AbstractGraphNode):
    is_agent = True

    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self._call_count = 0

    @property
    def block_type(self) -> str:
        return "test/agent_node"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            return {
                "output": "thinking",
                "tool_calls": [{"tool_id": "t1", "args": {"x": 1}}],
            }
        return {"output": f"done(calls={self._call_count})"}


class _BackboneForAgentKind(AbstractBaseBlock, AbstractGraphNode):
    """Backbone-role node for graph_kind='agent' detection."""

    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self._call_count = 0

    @property
    def block_type(self) -> str:
        return "test/backbone_agent"

    @property
    def role(self):
        from yggdrasill.task_nodes.roles import Role
        return Role.BACKBONE

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            return {
                "output": "backbone_thinking",
                "tool_calls": [{"tool_id": "t1", "args": {"q": "hi"}}],
            }
        return {"output": f"backbone_done(calls={self._call_count})"}


class _ToolNode(AbstractBaseBlock, AbstractGraphNode):
    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def block_type(self) -> str:
        return "test/tool"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"tool_result({inputs})"}


class _InspectingAgentNode(AbstractBaseBlock, AbstractGraphNode):
    is_agent = True

    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self._call_count = 0

    @property
    def block_type(self) -> str:
        return "test/inspecting_agent"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            return {
                "output": "thinking",
                "tool_calls": [{"tool_id": "t1", "args": {"x": 1}}],
            }
        return {
            "output": {
                "tool_results": inputs.get("tool_results"),
                "keys": sorted(inputs.keys()),
            },
        }


class _CounterNode(AbstractBaseBlock, AbstractGraphNode):
    """Node that counts how many times it's been called."""

    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self.call_count = 0

    @property
    def block_type(self) -> str:
        return "test/counter"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        return {"out": inputs.get("in")}


class _SeedableNode(AbstractBaseBlock, AbstractGraphNode):
    """Node that accepts seed assignment."""

    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self.seed: int | None = None

    @property
    def block_type(self) -> str:
        return "test/seedable"

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("in")}


class _ReadOnlySeedNode(AbstractBaseBlock, AbstractGraphNode):
    """Node where seed is a read-only property."""

    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def block_type(self) -> str:
        return "test/readonly_seed"

    @property
    def seed(self) -> None:
        return None

    def declare_ports(self) -> List[Port]:
        return [
            Port("in", PortDirection.IN, PortType.ANY),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("in")}


class _PersistentAgentNode(AbstractBaseBlock, AbstractGraphNode):
    """Agent that always returns tool_calls."""
    is_agent = True

    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def block_type(self) -> str:
        return "test/persistent_agent"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "output": "still_thinking",
            "tool_calls": [{"tool_id": "t1", "args": {}}],
        }


# ============================================================================
# Planner: agent detection tests
# ============================================================================

class TestPlannerAgentDetection:
    def test_agent_loop_via_metadata_agent_node_ids(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        h.metadata = {"agent_node_ids": ["A"]}
        plan = build_plan(h)
        assert plan == [("agent_loop", "A")]

    def test_agent_loop_via_is_agent_attribute(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("ag", _AgentNode("ag"))
        h.add_node("t1", _ToolNode("t1"))
        h.expose_input("ag", "input", "q")
        h.expose_output("ag", "output", "a")
        h.metadata = {"tool_id_to_node_id": {"t1": "t1"}}
        plan = build_plan(h)
        assert len(plan) == 1
        assert plan[0] == ("agent_loop", "ag")

    def test_agent_loop_via_graph_kind_agent(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("bb", _BackboneForAgentKind("bb"))
        h.add_node("t1", _ToolNode("t1"))
        h.expose_input("bb", "input", "q")
        h.expose_output("bb", "output", "a")
        h.graph_kind = "agent"
        h.metadata = {"tool_id_to_node_id": {"t1": "t1"}}
        plan = build_plan(h)
        assert len(plan) == 1
        assert plan[0] == ("agent_loop", "bb")

    def test_agent_mixed_detection_sources(self):
        """Both is_agent attribute and metadata agree — no duplicate."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("ag", _AgentNode("ag"))
        h.expose_input("ag", "input", "q")
        h.expose_output("ag", "output", "a")
        h.metadata = {"agent_node_ids": ["ag"]}
        plan = build_plan(h)
        assert plan == [("agent_loop", "ag")]

    def test_tool_nodes_excluded_from_plan(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("T", IdentityTaskNode(node_id="T"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        h.metadata = {
            "agent_node_ids": ["A"],
            "tool_id_to_node_id": {"tool1": "T"},
        }
        plan = build_plan(h)
        plan_nids = set()
        for _, data in plan:
            if isinstance(data, str):
                plan_nids.add(data)
        assert "T" not in plan_nids

    def test_non_backbone_node_not_promoted_by_graph_kind(self):
        """graph_kind='agent' only promotes backbone-role nodes."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("N", IdentityTaskNode(node_id="N"))
        h.expose_input("N", "in", "x")
        h.expose_output("N", "out", "y")
        h.graph_kind = "agent"
        plan = build_plan(h)
        assert plan == [("node", "N")]


# ============================================================================
# Planner: subgraph_node_ids tests
# ============================================================================

class TestPlannerSubgraph:
    def test_subgraph_restricts_plan(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        plan = build_plan(h, subgraph_node_ids={"A", "B"})
        ids = [s[1] for s in plan]
        assert "A" in ids
        assert "B" in ids
        assert "C" not in ids

    def test_subgraph_with_cycle(self):
        """Cycle members within subgraph stay as a cycle step."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_node("C", IdentityTaskNode(node_id="C"))
        h.add_edge(Edge("A", "out", "B", "in"))
        h.add_edge(Edge("B", "out", "A", "in"))
        h.add_edge(Edge("B", "out", "C", "in"))
        h.expose_input("A", "in", "x")
        h.expose_output("C", "out", "y")
        plan = build_plan(h, subgraph_node_ids={"A", "B"})
        assert len(plan) == 1
        assert plan[0][0] == "cycle"

    def test_subgraph_not_cached(self):
        """Subgraph plans must not pollute the plan cache."""
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        _ = build_plan(h)
        full_plan = build_plan(h)
        sub_plan = build_plan(h, subgraph_node_ids={"A"})
        assert len(sub_plan) == 1
        full_plan_2 = build_plan(h)
        assert full_plan_2 is full_plan

    def test_subgraph_empty_yields_empty_plan(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        plan = build_plan(h, subgraph_node_ids=set())
        assert plan == []

    def test_subgraph_excludes_tool_nodes(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("T", IdentityTaskNode(node_id="T"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        h.metadata = {"tool_id_to_node_id": {"t1": "T"}}
        plan = build_plan(h, subgraph_node_ids={"A", "T"})
        ids = [s[1] for s in plan if isinstance(s[1], str)]
        assert "A" in ids
        assert "T" not in ids


# ============================================================================
# Executor: agent_loop with is_agent detection (end-to-end)
# ============================================================================

class TestAgentLoopIsAgentE2E:
    def test_is_agent_node_runs_agent_loop(self):
        clear_plan_cache()
        h = Hypergraph()
        agent = _AgentNode("ag")
        tool = _ToolNode("t1")
        h.add_node("ag", agent)
        h.add_node("t1", tool)
        h.expose_input("ag", "input", "q")
        h.expose_output("ag", "output", "a")
        h.metadata = {"tool_id_to_node_id": {"t1": "t1"}}
        out = run(h, {"q": "hi"}, validate_before=False)
        assert "done" in out["a"]

    def test_graph_kind_agent_runs_agent_loop(self):
        clear_plan_cache()
        h = Hypergraph()
        bb = _BackboneForAgentKind("bb")
        tool = _ToolNode("t1")
        h.add_node("bb", bb)
        h.add_node("t1", tool)
        h.expose_input("bb", "input", "q")
        h.expose_output("bb", "output", "a")
        h.graph_kind = "agent"
        h.metadata = {"tool_id_to_node_id": {"t1": "t1"}}
        out = run(h, {"q": "hi"}, validate_before=False)
        assert "backbone_done" in out["a"]


# ============================================================================
# Executor: pin_data edge cases
# ============================================================================

class TestPinDataEdgeCases:
    def test_pin_multiple_nodes(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        log: list = []
        out = run(
            h, {"x": 1},
            pin_data={"A": {"out": "PA"}, "B": {"out": "PB"}},
            validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        assert ("pinned", "A") in log
        assert ("pinned", "B") in log
        assert ("before", "A") not in log
        assert ("before", "B") not in log
        assert ("before", "C") in log
        assert out["y"] == "PB"

    def test_pin_in_cycle(self):
        """Pinned node inside a cycle skips execution every iteration."""
        clear_plan_cache()
        h = make_cycle("A", "B")
        counter_a = _CounterNode("A")
        counter_b = _CounterNode("B")
        h._nodes["A"] = counter_a
        h._nodes["B"] = counter_b
        run(
            h, {"x": 1},
            pin_data={"A": {"out": "pinA"}},
            num_loop_steps=3,
            validate_before=False,
        )
        assert counter_a.call_count == 0
        assert counter_b.call_count == 3

    def test_pin_empty_dict_no_effect(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        log: list = []
        out = run(
            h, {"x": 42}, pin_data={},
            validate_before=False,
            callbacks=[lambda p, i: log.append(p)],
        )
        assert out["y"] == 42
        assert "pinned" not in log


# ============================================================================
# Executor: interrupt/resume edge cases
# ============================================================================

class TestInterruptResumeEdgeCases:
    def test_interrupt_at_first_node(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        result = run(h, {"x": 1}, interrupt_on=["A"], validate_before=False)
        assert isinstance(result, RunResult)
        assert result.suspended
        assert result.outputs["y"] is None

    def test_interrupt_at_last_node(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        result = run(h, {"x": 1}, interrupt_on=["B"], validate_before=False)
        assert isinstance(result, RunResult)
        assert result.suspended

    def test_no_interrupt_returns_plain_dict(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        out = run(h, {"x": 1}, interrupt_on=["NONEXISTENT"], validate_before=False)
        assert isinstance(out, dict)
        assert out["y"] == 1

    def test_resume_produces_correct_output(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        r1 = run(h, {"x": 99}, interrupt_on=["B"], validate_before=False)
        assert isinstance(r1, RunResult)
        final = run(h, {"x": 99}, run_data=r1.run_data, validate_before=False)
        assert final["y"] == 99

    def test_interrupt_preserves_partial_buffer(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        r1 = run(h, {"x": 7}, interrupt_on=["C"], validate_before=False)
        assert isinstance(r1, RunResult)
        assert "B" in r1.run_data
        assert r1.run_data["B"]["out"] == 7


# ============================================================================
# Executor: partial run edge cases
# ============================================================================

class TestPartialRunEdgeCases:
    def test_run_data_prepopulates_without_dirty(self):
        """run_data without dirty_node_ids should prepopulate buffers, all nodes run."""
        clear_plan_cache()
        h = make_chain("A", "B")
        log: list = []
        run(
            h, {"x": 1},
            run_data={"A": {"out": "preloaded"}},
            validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        executed = [nid for p, nid in log if p == "before"]
        assert "A" in executed
        assert "B" in executed

    def test_dirty_no_downstream(self):
        """Dirty node with no downstream: only that node runs."""
        clear_plan_cache()
        h = make_chain("A", "B")
        log: list = []
        run(
            h, {"x": 1},
            run_data={"A": {"out": 1}},
            dirty_node_ids=["B"],
            validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        executed = [nid for p, nid in log if p == "before"]
        assert "A" not in executed
        assert "B" in executed

    def test_destination_first_node(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        log: list = []
        run(
            h, {"x": 1}, destination_node_id="A",
            validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        executed = [nid for p, nid in log if p == "before"]
        assert executed == ["A"]

    def test_destination_last_node_runs_all(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        log: list = []
        run(
            h, {"x": 1}, destination_node_id="B",
            validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        executed = [nid for p, nid in log if p == "before"]
        assert "A" in executed
        assert "B" in executed

    def test_dirty_and_destination_combined(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        log: list = []
        run(
            h, {"x": 1},
            run_data={"A": {"out": 1}},
            dirty_node_ids=["B"],
            destination_node_id="B",
            validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        executed = [nid for p, nid in log if p == "before"]
        assert "A" not in executed
        assert "B" in executed
        assert "C" not in executed


# ============================================================================
# Executor: streaming edge cases
# ============================================================================

class TestStreamEdgeCases:
    def test_stream_final_equals_run(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        normal = run(h, {"x": 42}, validate_before=False)
        clear_plan_cache()
        snapshots = list(run_stream(h, {"x": 42}, validate_before=False))
        assert snapshots[-1] == normal

    def test_stream_validation_error(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(ValidationError):
            list(run_stream(h, {}))

    def test_stream_empty_graph(self):
        clear_plan_cache()
        h = Hypergraph()
        snapshots = list(run_stream(h, {}, validate_before=False))
        assert snapshots == []

    def test_stream_yields_per_step(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        snapshots = list(run_stream(h, {"x": 1}, validate_before=False))
        assert len(snapshots) == 2

    def test_stream_supports_partial_run_arguments(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        snapshots = list(
            run_stream(
                h,
                {"x": 1},
                run_data={"A": {"out": 1}},
                dirty_node_ids=["B"],
                destination_node_id="B",
                validate_before=False,
            )
        )
        assert len(snapshots) == 1
        assert snapshots[0]["y"] is None


# ============================================================================
# Executor: seed edge cases
# ============================================================================

class TestSeedEdgeCases:
    def test_seed_set_via_attribute(self):
        clear_plan_cache()
        h = Hypergraph()
        node = _SeedableNode("S")
        h.add_node("S", node)
        h.expose_input("S", "in", "x")
        h.expose_output("S", "out", "y")
        run(h, {"x": 1}, seed=123, validate_before=False)
        assert node.seed == 123

    def test_seed_none_skips_assignment(self):
        clear_plan_cache()
        h = Hypergraph()
        node = _SeedableNode("S")
        h.add_node("S", node)
        h.expose_input("S", "in", "x")
        h.expose_output("S", "out", "y")
        run(h, {"x": 1}, validate_before=False)
        assert node.seed is None

    def test_seed_readonly_property_tolerated(self):
        """If seed is a read-only property, setting it should not crash."""
        clear_plan_cache()
        h = Hypergraph()
        node = _ReadOnlySeedNode("R")
        h.add_node("R", node)
        h.expose_input("R", "in", "x")
        h.expose_output("R", "out", "y")
        out = run(h, {"x": 42}, seed=99, validate_before=False)
        assert out["y"] == 42


# ============================================================================
# Executor: agent_loop edge cases
# ============================================================================

class TestAgentLoopEdgeCases:
    def _build_agent_graph(self, agent_node, tool_nodes=None, tool_map=None):
        h = Hypergraph()
        h.add_node("ag", agent_node)
        tool_map = tool_map or {}
        for tid, tnid in (tool_nodes or {}).items():
            h.add_node(tnid, _ToolNode(tnid))
            tool_map[tid] = tnid
        h.expose_input("ag", "input", "q")
        h.expose_output("ag", "output", "a")
        h.metadata = {
            "agent_node_ids": ["ag"],
            "tool_id_to_node_id": tool_map,
        }
        return h

    def test_agent_loop_no_tool_map(self):
        """Agent returns tool_calls but no tool_map — loop ends after first step."""
        clear_plan_cache()
        h = Hypergraph()
        agent = _AgentNode("ag")
        h.add_node("ag", agent)
        h.expose_input("ag", "input", "q")
        h.expose_output("ag", "output", "a")
        h.metadata = {"agent_node_ids": ["ag"]}
        out = run(h, {"q": "hi"}, validate_before=False)
        assert out["a"] is not None

    def test_agent_loop_tool_not_in_map(self):
        """Agent requests a tool not in the tool_map — continues gracefully."""
        clear_plan_cache()
        agent = _AgentNode("ag")
        h = self._build_agent_graph(agent, tool_nodes={"other_tool": "ot"})
        out = run(h, {"q": "hi"}, validate_before=False)
        assert out["a"] is not None

    def test_agent_loop_missing_tool_emits_callback(self):
        clear_plan_cache()
        agent = _AgentNode("ag")
        h = self._build_agent_graph(agent, tool_nodes={"other_tool": "ot"})
        phases: list[tuple[str, dict]] = []
        run(
            h, {"q": "hi"},
            validate_before=False,
            callbacks=[lambda p, i: phases.append((p, dict(i)))],
        )
        missing = [info for phase, info in phases if phase == "tool_missing"]
        assert missing
        assert missing[0]["tool_id"] == "t1"

    def test_agent_loop_missing_tool_policy_error(self):
        clear_plan_cache()
        agent = _AgentNode("ag")
        h = self._build_agent_graph(agent, tool_nodes={"other_tool": "ot"})
        h.metadata["missing_tool_policy"] = "error"
        with pytest.raises(KeyError):
            run(h, {"q": "hi"}, validate_before=False)

    def test_agent_loop_dry_run(self):
        clear_plan_cache()
        h = self._build_agent_graph(
            _AgentNode("ag"),
            tool_nodes={"t1": "tool1"},
        )
        out = run(h, {"q": "hi"}, dry_run=True, validate_before=False)
        assert "a" in out

    def test_agent_loop_max_steps_limits(self):
        clear_plan_cache()
        h = self._build_agent_graph(
            _PersistentAgentNode("ag"),
            tool_nodes={"t1": "tool1"},
        )
        log: list = []
        run(
            h, {"q": "hi"}, max_steps=3,
            validate_before=False,
            callbacks=[lambda p, i: log.append(p)],
        )
        agent_steps = [p for p in log if p == "agent_step"]
        assert len(agent_steps) == 3

    def test_agent_loop_callbacks_phases(self):
        clear_plan_cache()
        agent = _AgentNode("ag")
        h = self._build_agent_graph(agent, tool_nodes={"t1": "tool1"})
        phases: list = []
        run(
            h, {"q": "hi"},
            validate_before=False,
            callbacks=[lambda p, i: phases.append(p)],
        )
        assert "agent_step" in phases
        assert "tool_call" in phases
        assert "agent_loop_done" in phases

    def test_agent_loop_pin_data_applies_to_tool_nodes(self):
        clear_plan_cache()
        h = self._build_agent_graph(
            _InspectingAgentNode("ag"),
            tool_nodes={"t1": "tool1"},
        )
        out = run(
            h,
            {"q": "hi"},
            pin_data={"tool1": {"result": "PINNED_TOOL"}},
            validate_before=False,
        )
        assert out["a"]["tool_results"] == [{"tool_call_id": "t1", "content": {"result": "PINNED_TOOL"}}]
        assert out["a"]["keys"] == ["input", "tool_results"]

    def test_agent_max_steps_from_metadata(self):
        clear_plan_cache()
        h = self._build_agent_graph(
            _PersistentAgentNode("ag"),
            tool_nodes={"t1": "tool1"},
        )
        h.metadata["max_steps"] = 2
        log: list = []
        run(
            h, {"q": "hi"},
            validate_before=False,
            callbacks=[lambda p, i: log.append(p)],
        )
        agent_steps = [p for p in log if p == "agent_step"]
        assert len(agent_steps) == 2


# ============================================================================
# EdgeBuffers: snapshot tests
# ============================================================================

class TestEdgeBuffersSnapshot:
    def test_snapshot_empty(self):
        buf = EdgeBuffers()
        assert buf.snapshot() == {}

    def test_snapshot_with_data(self):
        buf = EdgeBuffers()
        buf.write("A", "out", 42)
        buf.write("B", "x", "hello")
        snap = buf.snapshot()
        assert snap == {"A": {"out": 42}, "B": {"x": "hello"}}

    def test_snapshot_multiple_ports_same_node(self):
        buf = EdgeBuffers()
        buf.write("N", "p1", 1)
        buf.write("N", "p2", 2)
        snap = buf.snapshot()
        assert snap == {"N": {"p1": 1, "p2": 2}}

    def test_snapshot_does_not_include_multi(self):
        """snapshot only captures single-write data, not multi-edge accumulators."""
        buf = EdgeBuffers()
        buf.append("A", "data", 10, source_node="S1")
        snap = buf.snapshot()
        assert "A" not in snap


# ============================================================================
# Executor: run with all new params at defaults
# ============================================================================

class TestRunDefaultParams:
    def test_all_new_params_default(self):
        """All new params (pin_data, seed, etc.) default to None — normal execution."""
        clear_plan_cache()
        h = make_chain("A", "B")
        out = run(
            h, {"x": 42},
            max_steps=None, pin_data=None, seed=None,
            run_data=None, destination_node_id=None,
            dirty_node_ids=None, interrupt_on=None,
            validate_before=False,
        )
        assert out["y"] == 42
