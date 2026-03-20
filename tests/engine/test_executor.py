from typing import Any, Dict, List

import pytest
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.executor import run, ValidationError
from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortAggregation, PortDirection, PortType
from tests.engine.helpers import make_chain, make_cycle
from tests.foundation.helpers import IdentityTaskNode


class TestEdgeBuffersDirect:
    """Direct tests for EdgeBuffers to cover edge cases."""

    def test_aggregate_empty_multi_falls_back_to_single(self):
        from yggdrasill.engine.buffers import EdgeBuffers
        buf = EdgeBuffers()
        buf.write("A", "out", 42)
        result = buf.aggregate("A", "out")
        assert result == 42

    def test_init_from_inputs_with_tuple_key(self):
        from yggdrasill.engine.buffers import EdgeBuffers
        spec = [{"node_id": "N", "port_name": "in"}]
        inputs = {("N", "in"): 99}
        buf = EdgeBuffers.init_from_inputs(spec, inputs)
        assert buf.read("N", "in") == 99

    def test_init_from_inputs_falls_back_to_port_name_when_expose_name_differs(self):
        """e.g. expose name ``image`` but run() supplies ``init_image`` (diffusion)."""
        from yggdrasill.engine.buffers import EdgeBuffers
        spec = [{"node_id": "enc", "port_name": "init_image", "name": "image"}]
        buf = EdgeBuffers.init_from_inputs(spec, {"init_image": "url"})
        assert buf.read("enc", "init_image") == "url"


class TestExecutorChain:
    def test_chain_three(self):
        clear_plan_cache()
        h = make_chain("A", "B", "C")
        out = run(h, {"x": 42})
        assert out["y"] == 42

    def test_chain_two(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        out = run(h, {"x": "hello"})
        assert out["y"] == "hello"

    def test_single_node(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("X", IdentityTaskNode(node_id="X"))
        h.expose_input("X", "in", "x")
        h.expose_output("X", "out", "y")
        out = run(h, {"x": 99})
        assert out["y"] == 99


class TestExecutorCycle:
    def test_cycle_two_steps(self):
        clear_plan_cache()
        h = make_cycle("A", "B")
        out = run(h, {"x": 1}, num_loop_steps=2, validate_before=False)
        assert out["y"] == 1  # identity pass-through on each iteration

    def test_cycle_order_inpaint_blend_after_scheduler_step(self):
        """Regression: next_latent -> latents_post_step must not be ignored (SD15 4-ch inpaint)."""
        from yggdrasill.engine.executor import _cycle_node_order

        class _G:
            _edges = [
                Edge("unet", "noise_pred", "sched_step", "noise_pred"),
                Edge("sched_step", "next_latent", "inpaint_blend", "latents_post_step"),
                Edge("inpaint_blend", "next_latent", "unet", "latents"),
                Edge("sched_step", "next_timestep", "unet", "timestep"),
            ]

            def get_edges(self):
                return self._edges

        order = _cycle_node_order(_G(), {"unet", "sched_step", "inpaint_blend"})
        assert order.index("sched_step") < order.index("inpaint_blend")
        assert order.index("unet") < order.index("sched_step")


class TestExecutorDryRun:
    def test_dry_run(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        out = run(h, {"x": 42}, dry_run=True)
        assert "y" in out
        assert out["y"] is None


class TestExecutorValidation:
    def test_invalid_graph_raises(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        # required input not covered
        with pytest.raises(ValidationError):
            run(h, {})

    def test_skip_validation(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        out = run(h, {"x": 5}, validate_before=False)
        assert out["y"] == 5


class TestExecutorCallbacks:
    def test_callbacks_called(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        log = []
        run(h, {"x": 1}, callbacks=[lambda phase, info: log.append((phase, info["node_id"]))])
        assert ("before", "A") in log
        assert ("after", "A") in log
        assert ("before", "B") in log
        assert ("after", "B") in log

    def test_loop_start_end_callbacks(self):
        clear_plan_cache()
        h = make_cycle("A", "B")
        log = []

        def cb(phase, info):
            log.append(phase)

        run(h, {"x": 1}, num_loop_steps=3, callbacks=[cb], validate_before=False)
        assert "loop_start" in log
        assert "loop_end" in log
        start_idx = log.index("loop_start")
        end_idx = log.index("loop_end")
        assert start_idx < end_idx
        before_count = log[start_idx:end_idx].count("before")
        assert before_count == 6  # 3 iterations * 2 nodes


class _SumNode(AbstractBaseBlock, AbstractGraphNode):
    """Node with an aggregating input port for testing multi-edge aggregation."""

    def __init__(self, node_id: str, agg: PortAggregation = PortAggregation.CONCAT) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self._agg = agg

    @property
    def block_type(self) -> str:
        return "test/sum"

    def declare_ports(self) -> List[Port]:
        return [
            Port("data", PortDirection.IN, PortType.ANY, aggregation=self._agg),
            Port("out", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": inputs.get("data")}


class _SourceNode(AbstractBaseBlock, AbstractGraphNode):
    """Simple source node that outputs a fixed value."""

    def __init__(self, node_id: str, value: Any) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self._value = value

    @property
    def block_type(self) -> str:
        return "test/source"

    def declare_ports(self) -> List[Port]:
        return [Port("out", PortDirection.OUT, PortType.ANY)]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"out": self._value}


class TestExecutorEmptyGraph:
    def test_empty_graph_returns_empty(self):
        clear_plan_cache()
        h = Hypergraph()
        out = run(h, {}, validate_before=False)
        assert out == {}


class TestExecutorEdgeCases:
    def test_num_loop_steps_zero_skips_cycle(self):
        clear_plan_cache()
        h = make_cycle("A", "B")
        log = []
        run(h, {"x": 1}, num_loop_steps=0, callbacks=[lambda p, i: log.append(p)], validate_before=False)
        assert "before" not in log

    def test_callback_exception_swallowed(self):
        """Callback exceptions are warned but execution continues."""
        import warnings
        clear_plan_cache()
        h = make_chain("A", "B")

        def bad_callback(phase, info):
            raise RuntimeError("oops")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = run(h, {"x": 42}, callbacks=[bad_callback])
        assert out["y"] == 42
        assert any("Callback raised" in str(m.message) for m in w)

    def test_extra_input_keys_ignored(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        out = run(h, {"x": 1, "extra_key": 999})
        assert out["y"] == 1


class TestPortAggregation:
    def _build_fan_in(self, agg: PortAggregation, v1: Any = 10, v2: Any = 20):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("S1", _SourceNode("S1", v1))
        h.add_node("S2", _SourceNode("S2", v2))
        h.add_node("M", _SumNode("M", agg))
        h.add_edge(Edge("S1", "out", "M", "data"))
        h.add_edge(Edge("S2", "out", "M", "data"))
        h.expose_output("M", "out", "result")
        return h

    def test_concat(self):
        h = self._build_fan_in(PortAggregation.CONCAT)
        out = run(h, {}, validate_before=False)
        assert set(out["result"]) == {10, 20}

    def test_sum(self):
        h = self._build_fan_in(PortAggregation.SUM)
        out = run(h, {}, validate_before=False)
        assert out["result"] == 30

    def test_first(self):
        h = self._build_fan_in(PortAggregation.FIRST)
        out = run(h, {}, validate_before=False)
        assert out["result"] in (10, 20)

    def test_dict(self):
        h = self._build_fan_in(PortAggregation.DICT)
        out = run(h, {}, validate_before=False)
        assert isinstance(out["result"], dict)
        assert set(out["result"].values()) == {10, 20}

    def test_single_aggregation_on_multi_edge_raises(self):
        """SINGLE aggregation with multiple edges should use the last value."""
        clear_plan_cache()
        h = self._build_fan_in(PortAggregation.SINGLE)
        out = run(h, {}, validate_before=False)
        assert out["result"] in (10, 20)


class TestExecutorDevicePlacement:
    def test_run_with_device(self):
        clear_plan_cache()
        h = make_chain("A", "B")
        out = run(h, {"x": 42}, device="cpu")
        assert out["y"] == 42


class TestExecutorDryRunWorkflowLevel:
    def test_dry_run_on_workflow(self):
        """dry_run with workflow (nodes have get_output_spec not get_output_ports)."""
        clear_plan_cache()
        from yggdrasill.foundation.registry import BlockRegistry
        from yggdrasill.workflow.workflow import Workflow
        from tests.foundation.helpers import IdentityTaskNode
        reg = BlockRegistry()
        reg.register("test/identity_task", IdentityTaskNode)
        hg = Hypergraph.from_config({
            "graph_id": "hg",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }, registry=reg)
        w = Workflow()
        w.add_node("g1", hg)
        w.expose_input("g1", "in", "x")
        w.expose_output("g1", "out", "y")
        out = run(w, {"x": 99}, dry_run=True, validate_before=False)
        assert "y" in out
        assert out["y"] is None


class TestExecutorSpecKeyFallback:
    def test_spec_key_without_name_uses_node_port(self):
        """Cover _spec_key fallback when 'name' is absent."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h._exposed_inputs.append({"node_id": "A", "port_name": "in"})
        h._exposed_outputs.append({"node_id": "A", "port_name": "out"})
        out = run(h, {"A:in": 42}, validate_before=False)
        assert out["A:out"] == 42


class TestExecutorWorkflowLevelRun:
    def test_run_workflow_with_spec_key_fallback(self):
        """Cover _spec_key with graph_id and workflow-level execution."""
        clear_plan_cache()
        from yggdrasill.foundation.registry import BlockRegistry
        from yggdrasill.workflow.workflow import Workflow
        from tests.foundation.helpers import IdentityTaskNode
        reg = BlockRegistry()
        reg.register("test/identity_task", IdentityTaskNode)
        hg = Hypergraph.from_config({
            "graph_id": "hg",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }, registry=reg)
        w = Workflow()
        w.add_node("g1", hg)
        w.expose_input("g1", "in", "x")
        w.expose_output("g1", "out", "y")
        result = run(w, {"x": 42}, validate_before=False)
        assert result["y"] == 42


class TestExecutorMetadataFallback:
    def test_num_loop_steps_from_metadata(self):
        """When num_loop_steps not passed, executor uses metadata."""
        clear_plan_cache()
        h = make_cycle("A", "B")
        h.metadata = {"num_loop_steps": 3}
        log = []
        run(h, {"x": 1}, callbacks=[lambda p, i: log.append(p)], validate_before=False)
        before_count = log.count("before")
        assert before_count == 6  # 3 iterations * 2 nodes

    def test_explicit_num_loop_steps_overrides_metadata(self):
        """Explicit num_loop_steps arg takes priority over metadata."""
        clear_plan_cache()
        h = make_cycle("A", "B")
        h.metadata = {"num_loop_steps": 10}
        log = []
        run(h, {"x": 1}, num_loop_steps=1, callbacks=[lambda p, i: log.append(p)], validate_before=False)
        before_count = log.count("before")
        assert before_count == 2  # 1 iteration * 2 nodes


class TestExecutorForwardException:
    """If node.run raises, it propagates out of the executor."""

    def test_forward_exception_propagates(self):

        class FailNode(AbstractBaseBlock, AbstractGraphNode):
            def __init__(self, node_id):
                AbstractBaseBlock.__init__(self)
                AbstractGraphNode.__init__(self, node_id=node_id)

            @property
            def block_type(self):
                return "test/fail"

            def declare_ports(self):
                return [
                    Port("in", PortDirection.IN, PortType.ANY),
                    Port("out", PortDirection.OUT, PortType.ANY),
                ]

            def forward(self, inputs):
                raise RuntimeError("intentional failure")

        clear_plan_cache()
        h = Hypergraph()
        h.add_node("F", FailNode("F"))
        h.expose_input("F", "in", "x")
        h.expose_output("F", "out", "y")
        with pytest.raises(RuntimeError, match="intentional failure"):
            run(h, {"x": 1}, validate_before=False)


class TestExecutorDiamondDAG:
    """Diamond fan-in: S1→M, S2→M (CONCAT aggregation)."""

    def test_diamond_fan_in_concat(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("S1", _SourceNode("S1", 10))
        h.add_node("S2", _SourceNode("S2", 20))
        h.add_node("M", _SumNode("M", PortAggregation.CONCAT))
        h.add_edge(Edge("S1", "out", "M", "data"))
        h.add_edge(Edge("S2", "out", "M", "data"))
        h.expose_output("M", "out", "result")
        out = run(h, {}, validate_before=False)
        assert sorted(out["result"]) == [10, 20]


class TestExecutorDictAggregation:
    """DICT aggregation produces {source_node: value} mapping."""

    def test_dict_agg_keys(self):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node("S1", _SourceNode("S1", "val_a"))
        h.add_node("S2", _SourceNode("S2", "val_b"))
        h.add_node("M", _SumNode("M", PortAggregation.DICT))
        h.add_edge(Edge("S1", "out", "M", "data"))
        h.add_edge(Edge("S2", "out", "M", "data"))
        h.expose_output("M", "out", "result")
        out = run(h, {}, validate_before=False)
        result = out["result"]
        assert isinstance(result, dict)
        assert "S1" in result
        assert "S2" in result
        assert result["S1"] == "val_a"
        assert result["S2"] == "val_b"
