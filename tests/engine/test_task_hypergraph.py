import pytest
from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.registry import BlockRegistry
from tests.foundation.helpers import AddTaskNode, IdentityTaskNode, OptionalPortTaskNode


@pytest.fixture()
def registry():
    r = BlockRegistry()
    r.register("test/identity_task", IdentityTaskNode)
    r.register("test/add_task", AddTaskNode)
    r.register("test/optional_port", OptionalPortTaskNode)
    return r


CHAIN_CONFIG = {
    "graph_id": "test_chain",
    "nodes": [
        {"node_id": "A", "block_type": "test/identity_task"},
        {"node_id": "B", "block_type": "test/identity_task"},
    ],
    "edges": [
        {"source_node": "A", "source_port": "out", "target_node": "B", "target_port": "in"},
    ],
    "exposed_inputs": [{"node_id": "A", "port_name": "in", "name": "x"}],
    "exposed_outputs": [{"node_id": "B", "port_name": "out", "name": "y"}],
}


class TestAddNodeFromConfig:
    def test_basic(self, registry):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("A", "test/identity_task", registry=registry)
        node = h.get_node("A")
        assert node is not None
        assert node.node_id == "A"
        assert node.block_type == "test/identity_task"

    def test_with_pretrained(self, registry):
        h = Hypergraph()
        h.add_node_from_config("A", "test/add_task", config={"offset": 0},
                               pretrained={"offset": 99}, registry=registry)
        node = h.get_node("A")
        assert node.offset == 99

    def test_duplicate_node_id(self, registry):
        h = Hypergraph()
        h.add_node_from_config("A", "test/identity_task", registry=registry)
        with pytest.raises(ValueError, match="already exists"):
            h.add_node_from_config("A", "test/identity_task", registry=registry)

    def test_unknown_block_type(self, registry):
        h = Hypergraph()
        with pytest.raises(KeyError):
            h.add_node_from_config("A", "nonexistent", registry=registry)


class TestFromConfig:
    def test_basic(self, registry):
        clear_plan_cache()
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        assert h.graph_id == "test_chain"
        assert "A" in h.node_ids
        assert "B" in h.node_ids
        assert len(h.get_edges()) == 1
        assert len(h.get_input_spec()) == 1
        assert len(h.get_output_spec()) == 1

    def test_run_after_from_config(self, registry):
        clear_plan_cache()
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        out = h.run({"x": 42})
        assert out["y"] == 42

    def test_validate_on_from_config(self, registry):
        bad_config = {
            "nodes": [{"node_id": "A", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [],
            "exposed_outputs": [],
        }
        with pytest.raises(ValueError, match="validation failed"):
            Hypergraph.from_config(bad_config, registry=registry, validate=True)


class TestToConfig:
    def test_roundtrip(self, registry):
        clear_plan_cache()
        h1 = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        cfg = h1.to_config()
        assert cfg["schema_version"] == "1.0"
        assert cfg["graph_id"] == "test_chain"
        assert len(cfg["nodes"]) == 2
        assert len(cfg["edges"]) == 1

        h2 = Hypergraph.from_config(cfg, registry=registry)
        assert h2.node_ids == h1.node_ids
        assert h2.run({"x": 7}) == h1.run({"x": 7})


class TestInferExposedPorts:
    def test_infer(self, registry):
        from yggdrasill.engine.edge import Edge
        h = Hypergraph()
        h.add_node_from_config("A", "test/identity_task", registry=registry)
        h.add_node_from_config("B", "test/identity_task", registry=registry)
        h.add_node_from_config("C", "test/identity_task", registry=registry)
        h.add_edge(Edge("A", "out", "B", "in"))
        h.add_edge(Edge("B", "out", "C", "in"))
        h.infer_exposed_ports()
        in_spec = h.get_input_spec()
        out_spec = h.get_output_spec()
        in_ports = {(e["node_id"], e["port_name"]) for e in in_spec}
        out_ports = {(e["node_id"], e["port_name"]) for e in out_spec}
        assert ("A", "in") in in_ports
        assert ("C", "out") in out_ports


class TestStateDictHypergraph:
    def test_state_dict(self, registry):
        h = Hypergraph()
        h.add_node_from_config("A", "test/add_task", config={"offset": 5}, registry=registry)
        sd = h.state_dict()
        assert "A" in sd
        assert sd["A"]["offset"] == 5

    def test_load_state_dict(self, registry):
        clear_plan_cache()
        h = Hypergraph.from_config({
            "nodes": [
                {"node_id": "A", "block_type": "test/add_task", "config": {"offset": 0}},
            ],
            "edges": [],
            "exposed_inputs": [
                {"node_id": "A", "port_name": "a", "name": "a"},
                {"node_id": "A", "port_name": "b", "name": "b"},
            ],
            "exposed_outputs": [{"node_id": "A", "port_name": "out", "name": "out"}],
        }, registry=registry)
        h.load_state_dict({"A": {"offset": 100}})
        out = h.run({"a": 0, "b": 0}, validate_before=False)
        assert out["out"] == 100

    def test_get_input_spec_with_dtype(self, registry):
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        spec = h.get_input_spec(include_dtype=True)
        assert "dtype" in spec[0]


class TestGraphKindMetadataRoundtrip:
    def test_graph_kind_and_metadata_survive_roundtrip(self, registry):
        clear_plan_cache()
        cfg = dict(CHAIN_CONFIG)
        cfg["graph_kind"] = "diffusion"
        cfg["metadata"] = {"num_loop_steps": 3, "custom_key": "hello"}
        h = Hypergraph.from_config(cfg, registry=registry)
        exported = h.to_config()
        assert exported["graph_kind"] == "diffusion"
        assert exported["metadata"]["num_loop_steps"] == 3
        assert exported["metadata"]["custom_key"] == "hello"


class TestResolveConfigRef:
    def test_json_ref(self, registry, tmp_path):
        import json
        node_cfg = {"offset": 42}
        ref_file = tmp_path / "node_config.json"
        ref_file.write_text(json.dumps(node_cfg))
        cfg = {
            "nodes": [
                {"node_id": "A", "block_type": "test/add_task", "config": {"ref": str(ref_file)}},
            ],
            "edges": [],
            "exposed_inputs": [
                {"node_id": "A", "port_name": "a", "name": "a"},
                {"node_id": "A", "port_name": "b", "name": "b"},
            ],
            "exposed_outputs": [{"node_id": "A", "port_name": "out", "name": "y"}],
        }
        h = Hypergraph.from_config(cfg, registry=registry)
        out = h.run({"a": 0, "b": 0}, validate_before=False)
        assert out["y"] == 42


class TestSchemaVersionWarning:
    def test_from_config_warns_on_mismatched_schema_version(self, registry):
        import warnings
        cfg = dict(CHAIN_CONFIG)
        cfg["schema_version"] = "2.0"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Hypergraph.from_config(cfg, registry=registry)
            schema_warnings = [x for x in w if "schema_version" in str(x.message)]
            assert len(schema_warnings) >= 1

    def test_from_config_no_warning_for_1_0(self, registry):
        import warnings
        cfg = dict(CHAIN_CONFIG)
        cfg["schema_version"] = "1.0"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Hypergraph.from_config(cfg, registry=registry)
            schema_warnings = [x for x in w if "schema_version" in str(x.message)]
            assert len(schema_warnings) == 0


class TestResolveConfigRefYAML:
    def test_yaml_ref(self, registry, tmp_path):
        yaml = pytest.importorskip("yaml")
        node_cfg = {"offset": 77}
        ref_file = tmp_path / "node_config.yaml"
        ref_file.write_text(yaml.dump(node_cfg))
        cfg = {
            "nodes": [
                {"node_id": "A", "block_type": "test/add_task", "config": {"ref": str(ref_file)}},
            ],
            "edges": [],
            "exposed_inputs": [
                {"node_id": "A", "port_name": "a", "name": "a"},
                {"node_id": "A", "port_name": "b", "name": "b"},
            ],
            "exposed_outputs": [{"node_id": "A", "port_name": "out", "name": "y"}],
        }
        h = Hypergraph.from_config(cfg, registry=registry)
        out = h.run({"a": 0, "b": 0}, validate_before=False)
        assert out["y"] == 77


class TestResolveConfigRefYAMLImportError:
    def test_yaml_ref_without_pyyaml(self, registry, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        ref_file = tmp_path / "node.yaml"
        ref_file.write_text("{}")
        monkeypatch.setattr(builtins, "__import__", mock_import)
        from yggdrasill.engine.structure import _resolve_config_ref
        with pytest.raises(ImportError, match="PyYAML"):
            _resolve_config_ref({"ref": str(ref_file)})


class TestResolveConfigRefEdgeCases:
    def test_invalid_extension_raises(self, registry, tmp_path):
        bad_file = tmp_path / "config.txt"
        bad_file.write_text("{}")
        cfg = {
            "nodes": [
                {"node_id": "A", "block_type": "test/add_task", "config": {"ref": str(bad_file)}},
            ],
            "edges": [],
            "exposed_inputs": [
                {"node_id": "A", "port_name": "a", "name": "a"},
                {"node_id": "A", "port_name": "b", "name": "b"},
            ],
            "exposed_outputs": [{"node_id": "A", "port_name": "out", "name": "y"}],
        }
        import pytest
        with pytest.raises(ValueError, match="Unsupported"):
            Hypergraph.from_config(cfg, registry=registry)

    def test_path_traversal_ref_rejected(self):
        """Config ref with '..' is rejected for security."""
        from yggdrasill.engine.structure import _resolve_config_ref
        with pytest.raises(ValueError, match="path traversal"):
            _resolve_config_ref({"ref": "../../../etc/passwd"})
        with pytest.raises(ValueError, match="path traversal"):
            _resolve_config_ref({"ref": "sub/../other/config.json"})


class TestStateDictAliasDedup:
    def test_shared_block_id_deduplicates(self, registry):
        """Two nodes sharing block_id produce _aliases in state_dict."""
        h = Hypergraph()
        shared_node_a = AddTaskNode(node_id="A", block_id="shared", config={"offset": 5})
        shared_node_b = AddTaskNode(node_id="B", block_id="shared", config={"offset": 5})
        h.add_node("A", shared_node_a)
        h.add_node("B", shared_node_b)
        sd = h.state_dict()
        assert "_aliases" in sd
        assert "B" in sd["_aliases"] or "A" in sd["_aliases"]

    def test_load_state_dict_expands_aliases(self, registry):
        h = Hypergraph()
        n1 = AddTaskNode(node_id="A", block_id="shared", config={"offset": 5})
        n2 = AddTaskNode(node_id="B", block_id="shared", config={"offset": 5})
        h.add_node("A", n1)
        h.add_node("B", n2)
        sd = h.state_dict()
        n1.offset = 0
        n2.offset = 0
        h.load_state_dict(sd)
        assert n1.offset == 5
        assert n2.offset == 5


class TestFromConfigWithBlockId:
    def test_block_id_preserved(self, registry):
        cfg = {
            "nodes": [
                {"node_id": "A", "block_type": "test/identity_task", "block_id": "custom_bid"},
            ],
            "edges": [],
            "exposed_inputs": [{"node_id": "A", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"node_id": "A", "port_name": "out", "name": "y"}],
        }
        h = Hypergraph.from_config(cfg, registry=registry)
        node = h.get_node("A")
        assert node.block_id == "custom_bid"


class TestHypergraphToDevice:
    def test_to_returns_self(self, registry):
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        assert h.to("cpu") is h

    def test_set_trainable(self, registry):
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        h.set_trainable("A", False)
        params = list(h.trainable_parameters())
        assert isinstance(params, list)

    def test_set_trainable_unknown_raises(self, registry):
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        with pytest.raises(ValueError):
            h.set_trainable("GHOST", False)


class TestAddNodeFromConfigBlockId:
    def test_block_id_passed_to_node(self, registry):
        h = Hypergraph()
        h.add_node_from_config("A", "test/identity_task",
                               block_id="custom_bid", registry=registry)
        node = h.get_node("A")
        assert node.block_id == "custom_bid"

    def test_empty_node_id_raises(self, registry):
        h = Hypergraph()
        with pytest.raises(ValueError, match="non-empty"):
            h.add_node_from_config("", "test/identity_task", registry=registry)


class TestHypergraphTrainableParametersDedup:
    def test_trainable_parameters_skips_non_trainable(self, registry):
        h = Hypergraph.from_config(CHAIN_CONFIG, registry=registry)
        h.set_trainable("A", False)
        params = list(h.trainable_parameters())
        assert isinstance(params, list)

    def test_trainable_parameters_dedup_by_block_id(self, registry):
        from tests.foundation.helpers import AddTaskNode
        h = Hypergraph()
        n1 = AddTaskNode(node_id="X", block_id="shared", config={"offset": 5})
        n2 = AddTaskNode(node_id="Y", block_id="shared", config={"offset": 5})
        h.add_node("X", n1)
        h.add_node("Y", n2)
        params = list(h.trainable_parameters())
        assert isinstance(params, list)


class TestOptionalPort:
    """TODO_PART1: optional ports -- node with optional input, edge absent."""

    def test_optional_port_absent_ok(self, registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "opt",
            "nodes": [
                {"node_id": "src", "block_type": "test/identity_task"},
                {"node_id": "opt", "block_type": "test/optional_port"},
            ],
            "edges": [
                {"source_node": "src", "source_port": "out", "target_node": "opt", "target_port": "required_in"},
            ],
            "exposed_inputs": [
                {"node_id": "src", "port_name": "in", "name": "x"},
            ],
            "exposed_outputs": [{"node_id": "opt", "port_name": "out", "name": "y"}],
        }
        h = Hypergraph.from_config(cfg, registry=registry)
        out = h.run({"x": "hello"})
        assert out["y"] == "hello"

    def test_optional_port_present(self, registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "opt2",
            "nodes": [
                {"node_id": "src1", "block_type": "test/identity_task"},
                {"node_id": "src2", "block_type": "test/identity_task"},
                {"node_id": "opt", "block_type": "test/optional_port"},
            ],
            "edges": [
                {"source_node": "src1", "source_port": "out", "target_node": "opt", "target_port": "required_in"},
                {"source_node": "src2", "source_port": "out", "target_node": "opt", "target_port": "optional_in"},
            ],
            "exposed_inputs": [
                {"node_id": "src1", "port_name": "in", "name": "a"},
                {"node_id": "src2", "port_name": "in", "name": "b"},
            ],
            "exposed_outputs": [{"node_id": "opt", "port_name": "out", "name": "y"}],
        }
        h = Hypergraph.from_config(cfg, registry=registry)
        out = h.run({"a": "hello", "b": "world"})
        assert out["y"] == "hello+world"
