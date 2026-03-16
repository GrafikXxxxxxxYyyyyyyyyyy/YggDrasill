"""PHASE_4 §12: auto-connect tests."""
from __future__ import annotations

import pytest

from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.hypergraph.auto_connect import apply_auto_connect, use_task_node_auto_connect
from yggdrasill.task_nodes.role_rules import (
    get_rule_edges,
    suggest_edges_for_new_node,
)
from yggdrasill.task_nodes.stubs import (
    IdentityBackbone,
    IdentityConjector,
)


class TestRoleRules:
    def test_conjector_to_backbone(self):
        rules = get_rule_edges("conjector", "backbone")
        assert ("condition", "condition") in rules

    def test_backbone_to_inner_module(self):
        rules = get_rule_edges("backbone", "inner_module")
        assert ("pred", "pred") in rules

    def test_helper_to_backbone(self):
        rules = get_rule_edges("helper", "backbone")
        assert ("result", "latent") in rules

    def test_unknown_pair_empty(self):
        assert get_rule_edges("backbone", "backbone") == []

    def test_injector_to_inner_module_uses_control_port(self):
        """Audit B-2 fix: rule should use 'control' not 'condition'."""
        rules = get_rule_edges("injector", "inner_module")
        assert ("adapted", "control") in rules


class TestSuggestEdges:
    def test_basic_suggestions(self):
        existing = {"bb": "backbone"}
        suggestions = suggest_edges_for_new_node("cj", "conjector", existing)
        assert any(
            s == ("cj", "condition", "bb", "condition") for s in suggestions
        )

    def test_bidirectional_inner_module(self):
        existing = {"bb": "backbone"}
        suggestions = suggest_edges_for_new_node("im", "inner_module", existing)
        src_edges = [s for s in suggestions if s[0] == "bb"]
        tgt_edges = [s for s in suggestions if s[0] == "im"]
        assert len(src_edges) >= 1  # backbone -> inner_module
        assert len(tgt_edges) >= 1  # inner_module -> backbone


class TestApplyAutoConnect:
    @pytest.fixture(autouse=True)
    def _clear(self):
        clear_plan_cache()

    def test_auto_connect_creates_edges(self):
        h = Hypergraph()
        bb = IdentityBackbone(node_id="bb")
        h.add_node("bb", bb)

        cj = IdentityConjector(node_id="cj")
        h.add_node("cj", cj)

        added = apply_auto_connect(h, "cj", cj)
        assert added >= 1
        edges = h.get_edges()
        edge_tuples = [(e.source_node, e.source_port, e.target_node, e.target_port) for e in edges]
        assert ("cj", "condition", "bb", "condition") in edge_tuples

    def test_auto_connect_no_duplicate_edges(self):
        h = Hypergraph()
        bb = IdentityBackbone(node_id="bb")
        h.add_node("bb", bb)
        cj = IdentityConjector(node_id="cj")
        h.add_node("cj", cj)
        apply_auto_connect(h, "cj", cj)
        count1 = len(h.get_edges())
        apply_auto_connect(h, "cj", cj)
        assert len(h.get_edges()) == count1


class TestApplyAutoConnectEdgeCases:
    @pytest.fixture(autouse=True)
    def _clear(self):
        clear_plan_cache()

    def test_node_without_block_type_returns_zero(self):
        """If block has no block_type, auto_connect should skip it."""
        from yggdrasill.foundation.node import AbstractGraphNode
        from yggdrasill.foundation.port import Port, PortDirection

        class NoTypeNode(AbstractGraphNode):
            def declare_ports(self):
                return [Port("x", PortDirection.IN)]

        h = Hypergraph()
        h.add_node("bb", IdentityBackbone(node_id="bb"))
        n = NoTypeNode(node_id="raw")
        h.add_node("raw", n)
        added = apply_auto_connect(h, "raw", n)
        assert added == 0

    def test_node_with_unknown_role_returns_zero(self):
        """If block_type doesn't map to a known role, return 0."""
        from tests.foundation.helpers import IdentityTaskNode
        h = Hypergraph()
        h.add_node("bb", IdentityBackbone(node_id="bb"))
        fake = IdentityTaskNode(node_id="fake")
        h.add_node("fake", fake)
        added = apply_auto_connect(h, "fake", fake)
        assert added == 0

    def test_existing_node_without_block_type_skipped(self):
        """Existing nodes without block_type should be ignored."""
        from yggdrasill.foundation.node import AbstractGraphNode
        from yggdrasill.foundation.port import Port, PortDirection

        class RawNode(AbstractGraphNode):
            def declare_ports(self):
                return [Port("x", PortDirection.IN)]

        h = Hypergraph()
        h.add_node("raw", RawNode(node_id="raw"))
        cj = IdentityConjector(node_id="cj")
        h.add_node("cj", cj)
        added = apply_auto_connect(h, "cj", cj)
        assert added == 0

    def test_missing_port_on_target_skips(self):
        """If suggested port doesn't exist on the node, skip without error."""
        from yggdrasill.foundation.block import AbstractBaseBlock
        from yggdrasill.foundation.node import AbstractGraphNode
        from yggdrasill.foundation.port import Port, PortDirection, PortType

        class WeirdBackbone(AbstractBaseBlock, AbstractGraphNode):
            """Has backbone/ prefix but non-standard ports."""
            def __init__(self, node_id):
                AbstractBaseBlock.__init__(self)
                AbstractGraphNode.__init__(self, node_id=node_id)

            @property
            def block_type(self):
                return "backbone/weird"

            def declare_ports(self):
                return [
                    Port("x_in", PortDirection.IN, PortType.ANY),
                    Port("x_out", PortDirection.OUT, PortType.ANY),
                ]

            def forward(self, inputs):
                return {"x_out": inputs["x_in"]}

        h = Hypergraph()
        h.add_node("bb", WeirdBackbone(node_id="bb"))
        cj = IdentityConjector(node_id="cj")
        h.add_node("cj", cj)
        added = apply_auto_connect(h, "cj", cj)
        assert added == 0


class TestHelperToHelperAutoConnect:
    @pytest.fixture(autouse=True)
    def _clear(self):
        clear_plan_cache()

    def test_helper_to_helper_uses_result_query(self):
        """B-2: HELPER→HELPER should connect result→query."""
        from yggdrasill.task_nodes.stubs import IdentityHelper
        h = Hypergraph()
        h1 = IdentityHelper(node_id="h1")
        h2 = IdentityHelper(node_id="h2")
        h.add_node("h1", h1)
        h.add_node("h2", h2)
        added = apply_auto_connect(h, "h2", h2)
        assert added >= 1
        edges = h.get_edges()
        edge_tuples = [(e.source_node, e.source_port, e.target_node, e.target_port) for e in edges]
        assert ("h1", "result", "h2", "query") in edge_tuples


class TestAutoConnectIncompatiblePorts:
    @pytest.fixture(autouse=True)
    def _clear(self):
        clear_plan_cache()

    def test_incompatible_dtypes_skip(self):
        """If suggested ports have incompatible dtypes, the edge is not created."""
        from yggdrasill.foundation.block import AbstractBaseBlock
        from yggdrasill.foundation.node import AbstractGraphNode
        from yggdrasill.foundation.port import Port, PortDirection, PortType

        class TypedBackbone(AbstractBaseBlock, AbstractGraphNode):
            def __init__(self, node_id):
                AbstractBaseBlock.__init__(self)
                AbstractGraphNode.__init__(self, node_id=node_id)

            @property
            def block_type(self):
                return "backbone/typed"

            def declare_ports(self):
                return [
                    Port("latent", PortDirection.IN, PortType.IMAGE),
                    Port("pred", PortDirection.OUT, PortType.TEXT),
                ]

            def forward(self, inputs):
                return {"pred": inputs["latent"]}

        class TypedConjector(AbstractBaseBlock, AbstractGraphNode):
            def __init__(self, node_id):
                AbstractBaseBlock.__init__(self)
                AbstractGraphNode.__init__(self, node_id=node_id)

            @property
            def block_type(self):
                return "conjector/typed"

            def declare_ports(self):
                return [
                    Port("pred", PortDirection.OUT, PortType.IMAGE),
                    Port("latent", PortDirection.IN, PortType.ANY),
                ]

            def forward(self, inputs):
                return {"pred": inputs["latent"]}

        h = Hypergraph()
        h.add_node("bb", TypedBackbone(node_id="bb"))
        cj = TypedConjector(node_id="cj")
        h.add_node("cj", cj)
        added = apply_auto_connect(h, "cj", cj)
        assert added == 0


class TestUseTaskNodeAutoConnect:
    @pytest.fixture(autouse=True)
    def _clear(self):
        clear_plan_cache()

    def test_auto_connect_via_add_node_from_config(self):
        h = Hypergraph()
        bb = IdentityBackbone(node_id="bb")
        h.add_node("bb", bb)
        use_task_node_auto_connect(h)
        h.add_node_from_config(
            "cj", "conjector/identity", auto_connect=True,
        )
        edges = h.get_edges()
        edge_tuples = [(e.source_node, e.source_port, e.target_node, e.target_port) for e in edges]
        assert ("cj", "condition", "bb", "condition") in edge_tuples
