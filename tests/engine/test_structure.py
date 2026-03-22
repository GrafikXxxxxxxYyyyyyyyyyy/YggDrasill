from typing import Any, Dict, List

import pytest
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortDirection, PortType
from tests.foundation.helpers import IdentityTaskNode


class TestHypergraphNodes:
    def test_add_and_get_node(self):
        h = Hypergraph()
        n = IdentityTaskNode(node_id="A")
        h.add_node("A", n)
        assert h.get_node("A") is n
        assert "A" in h.node_ids

    def test_remove_node(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_edge(Edge("A", "out", "B", "in"))
        h.expose_input("A", "in", "x")
        h.remove_node("A")
        assert "A" not in h.node_ids
        assert h.get_edges() == []
        assert all(e.get("node_id") != "A" for e in h.get_input_spec())

    def test_remove_node_clears_diffusion_module_refs(self):
        """Heavy modules on removed nodes must be dropped so CUDA memory can be freed."""

        class Dummy:
            pass

        h = Hypergraph()
        d = Dummy()
        d._unet = object()
        d._vae = object()
        h.add_node("d", d)
        h.remove_node("d")
        assert d._unet is None
        assert d._vae is None

    def test_empty_node_id_raises(self):
        h = Hypergraph()
        with pytest.raises(ValueError):
            h.add_node("", IdentityTaskNode(node_id="A"))


class TestHypergraphEdges:
    def test_add_edge(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_edge(Edge("A", "out", "B", "in"))
        assert len(h.get_edges()) == 1

    def test_add_edge_idempotent(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        e = Edge("A", "out", "B", "in")
        h.add_edge(e)
        v1 = h.execution_version
        h.add_edge(e)
        assert len(h.get_edges()) == 1
        assert h.execution_version == v1  # no increment on duplicate

    def test_add_edge_unknown_node(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(ValueError, match="Target node"):
            h.add_edge(Edge("A", "out", "Z", "in"))

    def test_add_edge_unknown_port(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        with pytest.raises(ValueError, match="Port"):
            h.add_edge(Edge("A", "nonexistent", "B", "in"))

    def test_get_edges_in_out(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.add_edge(Edge("A", "out", "B", "in"))
        assert len(h.get_edges_out("A")) == 1
        assert len(h.get_edges_in("B")) == 1
        assert len(h.get_edges_in("A")) == 0


class TestHypergraphExposed:
    def test_expose_input_output(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_output("A", "out", "y")
        spec_in = h.get_input_spec()
        spec_out = h.get_output_spec()
        assert len(spec_in) == 1
        assert spec_in[0]["name"] == "x"
        assert len(spec_out) == 1
        assert spec_out[0]["name"] == "y"

    def test_expose_input_no_dup(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        h.expose_input("A", "in", "x")
        assert len(h.get_input_spec()) == 1

    def test_include_dtype(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_input("A", "in", "x")
        spec = h.get_input_spec(include_dtype=True)
        assert "dtype" in spec[0]


class TestAddEdgeIncompatiblePorts:
    """Verify that add_edge rejects edges between incompatible port types."""

    class _TypedNode(AbstractBaseBlock, AbstractGraphNode):
        def __init__(self, nid: str, in_dt: PortType, out_dt: PortType) -> None:
            AbstractBaseBlock.__init__(self)
            AbstractGraphNode.__init__(self, node_id=nid)
            self._in_dt, self._out_dt = in_dt, out_dt

        @property
        def block_type(self) -> str:
            return "test/typed"

        def declare_ports(self) -> List[Port]:
            return [
                Port("in", PortDirection.IN, self._in_dt),
                Port("out", PortDirection.OUT, self._out_dt),
            ]

        def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {"out": inputs.get("in")}

    def test_incompatible_raises(self):
        h = Hypergraph()
        h.add_node("A", self._TypedNode("A", PortType.ANY, PortType.TENSOR))
        h.add_node("B", self._TypedNode("B", PortType.IMAGE, PortType.ANY))
        with pytest.raises(ValueError, match="Incompatible"):
            h.add_edge(Edge("A", "out", "B", "in"))

    def test_compatible_ok(self):
        h = Hypergraph()
        h.add_node("A", self._TypedNode("A", PortType.ANY, PortType.TENSOR))
        h.add_node("B", self._TypedNode("B", PortType.TENSOR, PortType.ANY))
        h.add_edge(Edge("A", "out", "B", "in"))
        assert len(h.get_edges()) == 1


class TestExecutionVersion:
    def test_increments(self):
        h = Hypergraph()
        v0 = h.execution_version
        h.add_node("A", IdentityTaskNode(node_id="A"))
        assert h.execution_version > v0
        v1 = h.execution_version
        h.add_node("B", IdentityTaskNode(node_id="B"))
        assert h.execution_version > v1
        v2 = h.execution_version
        h.add_edge(Edge("A", "out", "B", "in"))
        assert h.execution_version > v2

    def test_increments_on_remove_edge(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        e = Edge("A", "out", "B", "in")
        h.add_edge(e)
        v = h.execution_version
        h.remove_edge(e)
        assert h.execution_version > v

    def test_increments_on_remove_node(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        v = h.execution_version
        h.remove_node("A")
        assert h.execution_version > v


class TestMetadataMutability:
    def test_metadata_in_place_mutation(self):
        h = Hypergraph()
        h.metadata["num_loop_steps"] = 5
        assert h.metadata.get("num_loop_steps") == 5

    def test_metadata_full_replacement(self):
        h = Hypergraph()
        h.metadata = {"key": "value"}
        assert h.metadata["key"] == "value"


class TestHypergraphAddEdgeDirectionChecks:
    def test_add_edge_source_port_wrong_direction(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        with pytest.raises(ValueError, match="not an output"):
            h.add_edge(Edge("A", "in", "B", "in"))

    def test_add_edge_target_port_wrong_direction(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        with pytest.raises(ValueError, match="not an input"):
            h.add_edge(Edge("A", "out", "B", "out"))

    def test_add_edge_target_port_not_found(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        with pytest.raises(ValueError, match="not found"):
            h.add_edge(Edge("A", "out", "B", "nonexistent"))

    def test_add_edge_source_not_in_graph(self):
        h = Hypergraph()
        h.add_node("B", IdentityTaskNode(node_id="B"))
        with pytest.raises(ValueError, match="Source node"):
            h.add_edge(Edge("GHOST", "out", "B", "in"))


class TestHypergraphExposeValidation:
    def test_expose_input_unknown_node(self):
        h = Hypergraph()
        with pytest.raises(ValueError, match="not in graph"):
            h.expose_input("GHOST", "in")

    def test_expose_input_unknown_port(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(ValueError, match="not found"):
            h.expose_input("A", "nonexistent")

    def test_expose_input_wrong_direction(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(ValueError, match="not an input"):
            h.expose_input("A", "out")

    def test_expose_output_unknown_node(self):
        h = Hypergraph()
        with pytest.raises(ValueError, match="not in graph"):
            h.expose_output("GHOST", "out")

    def test_expose_output_unknown_port(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(ValueError, match="not found"):
            h.expose_output("A", "nonexistent")

    def test_expose_output_wrong_direction(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(ValueError, match="not an output"):
            h.expose_output("A", "in")

    def test_expose_output_duplicate_ignored(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.expose_output("A", "out", "y")
        v = h.execution_version
        h.expose_output("A", "out", "y")
        assert h.execution_version == v


class TestHypergraphRemoveEdge:
    def test_remove_nonexistent_edge_no_error(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.add_node("B", IdentityTaskNode(node_id="B"))
        h.remove_edge(Edge("A", "out", "B", "in"))

    def test_remove_nonexistent_node_no_error(self):
        h = Hypergraph()
        h.remove_node("GHOST")


class TestHypergraphGraphKind:
    def test_graph_kind_setter_getter(self):
        h = Hypergraph()
        assert h.graph_kind is None
        h.graph_kind = "diffusion"
        assert h.graph_kind == "diffusion"


class TestHypergraphSaveCheckpoint:
    def test_save_checkpoint(self, tmp_path):
        from yggdrasill.foundation.registry import BlockRegistry
        from tests.foundation.helpers import AddTaskNode
        r = BlockRegistry()
        r.register("test/add_task", AddTaskNode)
        h = Hypergraph()
        h.add_node("A", AddTaskNode(node_id="A", config={"offset": 10}))
        h.expose_input("A", "a", "a")
        h.expose_input("A", "b", "b")
        h.expose_output("A", "out", "y")
        h.save_checkpoint(tmp_path / "ckpt")
        assert (tmp_path / "ckpt" / "checkpoint.pkl").exists()


class TestHypergraphLoadStateStrictError:
    def test_extra_key_raises(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(KeyError, match="not in graph"):
            h.load_state_dict({"GHOST": {}}, strict=True)


class TestHypergraphInferExposedWithNonNode:
    def test_non_graph_node_skipped_by_infer(self):
        """infer_exposed_ports skips nodes not inheriting AbstractGraphNode."""
        h = Hypergraph()

        class PlainObj:
            def run(self, inputs):
                return inputs

        h.add_node("plain", PlainObj())
        h.add_node("A", IdentityTaskNode(node_id="A"))
        h.infer_exposed_ports()
        exposed_nids = {e["node_id"] for e in h.get_input_spec()}
        assert "plain" not in exposed_nids
        assert "A" in exposed_nids


class TestHypergraphStateDictNoMethod:
    def test_node_without_state_dict_skipped(self):
        """state_dict skips nodes that don't have state_dict method."""
        h = Hypergraph()

        class Bare:
            pass

        h.add_node("bare", Bare())
        h.add_node("A", IdentityTaskNode(node_id="A"))
        sd = h.state_dict()
        assert "bare" not in sd


class TestHypergraphSpecKey:
    def test_spec_key_with_name(self):
        assert Hypergraph._spec_key({"node_id": "A", "port_name": "in", "name": "x"}) == "x"

    def test_spec_key_without_name(self):
        assert Hypergraph._spec_key({"node_id": "A", "port_name": "in"}) == "A:in"

    def test_spec_key_with_graph_id(self):
        assert Hypergraph._spec_key({"graph_id": "g1", "port_name": "out"}) == "g1:out"

    def test_spec_key_name_none_falls_back(self):
        assert Hypergraph._spec_key({"node_id": "A", "port_name": "in", "name": None}) == "A:in"


class TestHypergraphDuplicateNodeId:
    def test_add_node_from_config_duplicate_raises(self):
        h = Hypergraph()
        h.add_node("A", IdentityTaskNode(node_id="A"))
        with pytest.raises(ValueError, match="already exists"):
            h.add_node_from_config("A", "test/identity_task")


class TestHypergraphFromConfigKeyShadowing:
    """B-3: Config keys don't shadow block_type/node_id/block_id."""

    def test_config_with_shadowing_keys(self):
        from yggdrasill.foundation.registry import BlockRegistry
        reg = BlockRegistry()
        reg.register("test/identity_task", IdentityTaskNode)
        cfg = {
            "graph_id": "g",
            "nodes": [{
                "node_id": "A",
                "block_type": "test/identity_task",
                "config": {"block_type": "malicious", "node_id": "evil"},
            }],
            "edges": [],
            "exposed_inputs": [{"node_id": "A", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"node_id": "A", "port_name": "out", "name": "y"}],
        }
        h = Hypergraph.from_config(cfg, registry=reg)
        node = h.get_node("A")
        assert node.block_type == "test/identity_task"
        assert node._node_id == "A"


class TestHypergraphPretrainedPath:
    """S-1: pretrained accepts both dict and string path."""

    def test_pretrained_dict(self):
        from yggdrasill.foundation.registry import BlockRegistry
        from tests.foundation.helpers import AddTaskNode
        reg = BlockRegistry()
        reg.register("test/add_task", AddTaskNode)
        h = Hypergraph()
        h.add_node_from_config("A", "test/add_task", registry=reg, pretrained={"offset": 42})
        node = h.get_node("A")
        assert node.offset == 42

    def test_pretrained_path(self, tmp_path):
        import pickle
        from yggdrasill.foundation.registry import BlockRegistry
        from tests.foundation.helpers import AddTaskNode
        reg = BlockRegistry()
        reg.register("test/add_task", AddTaskNode)
        ckpt = tmp_path / "weights.pkl"
        with open(ckpt, "wb") as f:
            pickle.dump({"offset": 99}, f)
        h = Hypergraph()
        h.add_node_from_config("A", "test/add_task", registry=reg, pretrained=str(ckpt))
        node = h.get_node("A")
        assert node.offset == 99


class _IpImageInOnly(AbstractBaseBlock, AbstractGraphNode):
    """Minimal node exposing ``ip_adapter_image`` for routing tests."""

    def __init__(self, node_id: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def block_type(self) -> str:
        return "test/ip_image_in"

    def declare_ports(self) -> List[Port]:
        return [
            Port("ip_adapter_image", PortDirection.IN, PortType.IMAGE, optional=True),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class TestResolveRunKwargsMultiExposeSamePort:
    """Bare port names must not route kwargs when several nodes share that port (multi IP-Adapter)."""

    def test_bare_ip_adapter_image_not_in_resolved_when_two_nodes_expose_it(self):
        h = Hypergraph()
        h.add_node("style_ip", _IpImageInOnly("style_ip"))
        h.add_node("face_ip", _IpImageInOnly("face_ip"))
        h.expose_input("style_ip", "ip_adapter_image", "style_ip:ip_adapter_image")
        h.expose_input("face_ip", "ip_adapter_image", "face_ip:ip_adapter_image")

        resolved, _ = h._resolve_run_kwargs(
            {"style_ip:ip_adapter_image": "S", "face_ip:ip_adapter_image": "F"},
            {"ip_adapter_image": "WRONG"},
        )
        assert resolved["style_ip:ip_adapter_image"] == "S"
        assert resolved["face_ip:ip_adapter_image"] == "F"
        assert "ip_adapter_image" not in resolved

    def test_bare_ip_adapter_image_routed_when_single_node_exposes_it(self):
        h = Hypergraph()
        h.add_node("ip", _IpImageInOnly("ip"))
        h.expose_input("ip", "ip_adapter_image", "custom_name")
        resolved, _ = h._resolve_run_kwargs({}, {"ip_adapter_image": "Z"})
        assert resolved["ip_adapter_image"] == "Z"
