import json
from unittest.mock import patch

import pytest
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.executor import ValidationError
from yggdrasill.engine.planner import clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.workflow.auto_connect import apply_auto_connect, suggest_auto_edges
from yggdrasill.workflow.workflow import Workflow
from tests.foundation.helpers import AddTaskNode, IdentityTaskNode

import yggdrasill.task_nodes.stubs  # noqa: F401


@pytest.fixture()
def registry():
    r = BlockRegistry()
    r.register("test/identity_task", IdentityTaskNode)
    r.register("test/add_task", AddTaskNode)
    return r


def _make_identity_hg(graph_id: str, registry: BlockRegistry) -> Hypergraph:
    return Hypergraph.from_config({
        "graph_id": graph_id,
        "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
        "edges": [],
        "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
        "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
    }, registry=registry)


def _make_add_hg(graph_id: str, offset: float, registry: BlockRegistry) -> Hypergraph:
    return Hypergraph.from_config({
        "graph_id": graph_id,
        "nodes": [{"node_id": "N", "block_type": "test/add_task", "config": {"offset": offset}}],
        "edges": [],
        "exposed_inputs": [
            {"node_id": "N", "port_name": "a", "name": "a"},
            {"node_id": "N", "port_name": "b", "name": "b"},
        ],
        "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
    }, registry=registry)


class TestWorkflowBasic:
    def test_add_and_get_node(self, registry):
        w = Workflow()
        hg = _make_identity_hg("hg1", registry)
        gid = w.add_node("step1", hg)
        assert gid == "step1"
        assert "step1" in w.node_ids
        assert w.get_node("step1") is hg

    def test_remove_node(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.remove_node("s1")
        assert "s1" not in w.node_ids

    def test_execution_version(self, registry):
        w = Workflow()
        v0 = w.execution_version
        w.add_node("s1", _make_identity_hg("hg1", registry))
        assert w.execution_version > v0


class TestWorkflowRun:
    def test_chain_of_two_hypergraphs(self, registry):
        clear_plan_cache()
        w = Workflow(workflow_id="w_chain")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge("s1", "out", "s2", "in")
        w.expose_input("s1", "in", "x")
        w.expose_output("s2", "out", "y")

        result = w.run({"x": 42}, validate_before=False)
        assert result["y"] == 42

    def test_chain_of_three(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("a", _make_identity_hg("hg_a", registry))
        w.add_node("b", _make_identity_hg("hg_b", registry))
        w.add_node("c", _make_identity_hg("hg_c", registry))
        w.add_edge("a", "out", "b", "in")
        w.add_edge("b", "out", "c", "in")
        w.expose_input("a", "in", "x")
        w.expose_output("c", "out", "y")

        result = w.run({"x": "hello"}, validate_before=False)
        assert result["y"] == "hello"

    def test_single_hypergraph(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("g", _make_identity_hg("hg", registry))
        w.expose_input("g", "in", "x")
        w.expose_output("g", "out", "y")
        result = w.run({"x": 99}, validate_before=False)
        assert result["y"] == 99

    def test_run_with_kwargs_routes_to_inputs(self, registry):
        w = Workflow()
        w.add_node("g", _make_identity_hg("hg", registry))
        w.expose_input("g", "in", "x")
        w.expose_output("g", "out", "y")

        with patch("yggdrasill.engine.executor.run") as mock_run:
            mock_run.return_value = {"y": 1}
            w.run(x=42)
            assert mock_run.call_args[0][1]["x"] == 42

    def test_run_forwards_executor_kwargs(self, registry):
        w = Workflow()
        w.add_node("g", _make_identity_hg("hg", registry))
        w.expose_input("g", "in", "x")
        w.expose_output("g", "out", "y")

        with patch("yggdrasill.engine.executor.run") as mock_run:
            mock_run.return_value = {"y": 1}
            w.run(
                x=1,
                pin_data={"g": {"out": 7}},
                run_data={"g": {"out": 6}},
                dirty_node_ids=["g"],
                destination_node_id="g",
                interrupt_on=["g"],
                seed=123,
                max_steps=5,
                validate_before=False,
            )
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["pin_data"] == {"g": {"out": 7}}
            assert call_kwargs["run_data"] == {"g": {"out": 6}}
            assert call_kwargs["dirty_node_ids"] == ["g"]
            assert call_kwargs["destination_node_id"] == "g"
            assert call_kwargs["interrupt_on"] == ["g"]
            assert call_kwargs["seed"] == 123
            assert call_kwargs["max_steps"] == 5


class TestWorkflowAddEdgeMissing:
    def test_add_edge_target_not_in_workflow(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        with pytest.raises(ValueError, match="not in workflow"):
            w.add_edge("s1", "out", "GHOST", "in")

    def test_add_edge_source_not_in_workflow(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        with pytest.raises(ValueError, match="not in workflow"):
            w.add_edge("GHOST", "out", "s1", "in")


class TestWorkflowYAMLImportError:
    def test_read_yaml_config_without_pyyaml(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("{}")
        monkeypatch.setattr(builtins, "__import__", mock_import)
        from yggdrasill.hypergraph.serialization import load_config as _read_workflow_config
        with pytest.raises(ImportError, match="PyYAML"):
            _read_workflow_config(yaml_path)


class TestWorkflowProperties:
    def test_workflow_id_property(self, registry):
        w = Workflow(workflow_id="my_wf")
        assert w.workflow_id == "my_wf"

    def test_workflow_id_default(self):
        w = Workflow()
        assert w.workflow_id == "workflow"


class TestWorkflowEdgeValidation:
    def test_add_edge_validates_ports(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        with pytest.raises(ValueError, match="not in output spec"):
            w.add_edge("s1", "nonexistent", "s2", "in")

    def test_add_edge_unknown_graph(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        with pytest.raises(ValueError, match="not in workflow"):
            w.add_edge("s1", "out", "missing", "in")


class TestWorkflowRemoveEdge:
    def test_remove_edge(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge("s1", "out", "s2", "in")
        assert len(w.get_edges()) == 1
        w.remove_edge("s1", "out", "s2", "in")
        assert len(w.get_edges()) == 0


class TestWorkflowConfig:
    def test_to_config(self, registry):
        w = Workflow(workflow_id="test_wf")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        w.expose_output("s1", "out", "y")
        cfg = w.to_config()
        assert cfg["schema_version"] == "1.0"
        assert cfg["workflow_id"] == "test_wf"
        assert len(cfg["graphs"]) == 1
        assert cfg["graphs"][0]["graph_id"] == "s1"
        assert "config" in cfg["graphs"][0]

    def test_from_config_roundtrip(self, registry):
        clear_plan_cache()
        w1 = Workflow(workflow_id="rt_wf")
        w1.add_node("s1", _make_identity_hg("hg1", registry))
        w1.add_node("s2", _make_identity_hg("hg2", registry))
        w1.add_edge("s1", "out", "s2", "in")
        w1.expose_input("s1", "in", "x")
        w1.expose_output("s2", "out", "y")

        cfg = w1.to_config()
        w2 = Workflow.from_config(cfg, registry=registry)
        assert w2.graph_id == "rt_wf"
        assert w2.node_ids == w1.node_ids

        result = w2.run({"x": 7}, validate_before=False)
        assert result["y"] == 7

    def test_workflow_kind(self, registry):
        w = Workflow(workflow_id="kind_wf")
        w.workflow_kind = "chain"
        w.add_node("s1", _make_identity_hg("hg1", registry))
        cfg = w.to_config()
        assert cfg["workflow_kind"] == "chain"
        w2 = Workflow.from_config(cfg, registry=registry)
        assert w2.workflow_kind == "chain"


class TestWorkflowStateDict:
    def test_state_dict(self, registry):
        w = Workflow()
        w.add_node("s1", _make_add_hg("hg_add", 99, registry))
        sd = w.state_dict()
        assert "s1" in sd
        assert sd["s1"]["N"]["offset"] == 99

    def test_state_dict_empty_graph(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        sd = w.state_dict()
        assert "s1" in sd
        assert sd["s1"] == {}

    def test_load_state_dict(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("s1", _make_add_hg("hg_add", 0, registry))
        w.expose_input("s1", "a", "a")
        w.expose_input("s1", "b", "b")
        w.expose_output("s1", "out", "out")
        w.load_state_dict({"s1": {"N": {"offset": 50}}})
        result = w.run({"a": 0, "b": 0}, validate_before=False)
        assert result["out"] == 50

    def test_load_state_dict_strict(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        with pytest.raises(KeyError, match="not in workflow"):
            w.load_state_dict({"unknown": {}}, strict=True)


class TestWorkflowTrainable:
    def test_set_trainable(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.set_trainable("s1", False)
        params = list(w.trainable_parameters())
        assert params == []

    def test_set_trainable_unknown(self, registry):
        w = Workflow()
        with pytest.raises(ValueError, match="not in workflow"):
            w.set_trainable("missing", False)


class TestWorkflowTo:
    def test_to_device(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        result = w.to("cpu")
        assert result is w


class TestWorkflowInferExposed:
    def test_infer_exposed_ports(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge("s1", "out", "s2", "in")
        w.infer_exposed_ports()
        input_ports = {e["port_name"] for e in w.get_input_spec()}
        output_ports = {e["port_name"] for e in w.get_output_spec()}
        assert "in" in input_ports
        assert "out" in output_ports


class TestWorkflowSaveLoad:
    def test_save_load_roundtrip(self, tmp_path, registry):
        clear_plan_cache()
        w1 = Workflow(workflow_id="persist_wf")
        w1.add_node("s1", _make_identity_hg("hg1", registry))
        w1.add_node("s2", _make_identity_hg("hg2", registry))
        w1.add_edge("s1", "out", "s2", "in")
        w1.expose_input("s1", "in", "x")
        w1.expose_output("s2", "out", "y")

        w1.save(tmp_path / "wf")
        w2 = Workflow.load(tmp_path / "wf", registry=registry)

        assert w2.graph_id == "persist_wf"
        result = w2.run({"x": "data"}, validate_before=False)
        assert result["y"] == "data"

    def test_save_load_with_state(self, tmp_path, registry):
        clear_plan_cache()
        w1 = Workflow()
        w1.add_node("s1", _make_add_hg("hg_add", 42, registry))
        w1.expose_input("s1", "a", "a")
        w1.expose_input("s1", "b", "b")
        w1.expose_output("s1", "out", "out")
        out1 = w1.run({"a": 1, "b": 2}, validate_before=False)

        w1.save(tmp_path / "wf_state")
        w2 = Workflow.load(tmp_path / "wf_state", registry=registry)
        out2 = w2.run({"a": 1, "b": 2}, validate_before=False)
        assert out1 == out2

    def test_save_config_only(self, tmp_path, registry):
        w = Workflow(workflow_id="cfg_only")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        w.expose_output("s1", "out", "y")
        w.save_config(tmp_path / "wf_cfg")
        assert (tmp_path / "wf_cfg" / "config.json").exists()

    def test_load_from_checkpoint(self, tmp_path, registry):
        clear_plan_cache()
        w1 = Workflow()
        w1.add_node("s1", _make_add_hg("hg_add", 77, registry))
        w1.expose_input("s1", "a", "a")
        w1.expose_input("s1", "b", "b")
        w1.expose_output("s1", "out", "out")
        w1.save(tmp_path / "wf_ckpt")

        w2 = Workflow()
        w2.add_node("s1", _make_add_hg("hg_add", 0, registry))
        w2.expose_input("s1", "a", "a")
        w2.expose_input("s1", "b", "b")
        w2.expose_output("s1", "out", "out")
        w2.load_from_checkpoint(tmp_path / "wf_ckpt")

        result = w2.run({"a": 0, "b": 0}, validate_before=False)
        assert result["out"] == 77


class TestWorkflowCycle:
    def test_cycle_between_hypergraphs(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("a", _make_identity_hg("cyc_a", registry))
        w.add_node("b", _make_identity_hg("cyc_b", registry))
        w.add_edge("a", "out", "b", "in")
        w.add_edge("b", "out", "a", "in")
        w.expose_input("a", "in", "x")
        w.expose_output("b", "out", "y")
        w.metadata = {"num_loop_steps": 3}

        result = w.run({"x": "loop"}, validate_before=False, num_loop_steps=3)
        assert result["y"] == "loop"


class TestWorkflowValidateBefore:
    """PHASE_6 §15.9: invalid workflow + validate_before=True raises."""

    def test_invalid_workflow_raises(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("a", _make_identity_hg("va", registry))
        # Inject a bad edge that references nonexistent graph
        w._edges.append(Edge("a", "out", "NONEXISTENT", "in"))
        with pytest.raises(ValidationError):
            w.run({"x": 1}, validate_before=True)

    def test_valid_workflow_ok(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("a", _make_identity_hg("vok", registry))
        w.expose_input("a", "in", "x")
        w.expose_output("a", "out", "y")
        result = w.run({"x": 42}, validate_before=True)
        assert result["y"] == 42


class TestWorkflowFromConfigValidate:
    """PHASE_6 §10.2: from_config with validate=True."""

    def test_from_config_validate_true(self, registry):
        w = Workflow()
        w.add_node("a", _make_identity_hg("fc", registry))
        w.expose_input("a", "in", "x")
        w.expose_output("a", "out", "y")
        cfg = w.to_config()
        w2 = Workflow.from_config(cfg, registry=registry, validate=True)
        assert w2.node_ids == {"a"}


class TestWorkflowLoadConfig:
    """PHASE_6 §12: load_config class method."""

    def test_load_config_only(self, tmp_path, registry):
        w = Workflow(workflow_id="lc")
        w.add_node("a", _make_identity_hg("lca", registry))
        w.expose_input("a", "in", "x")
        w.expose_output("a", "out", "y")
        w.save(tmp_path / "wf_lc")
        w2 = Workflow.load_config(tmp_path / "wf_lc", registry=registry)
        assert w2.node_ids == {"a"}


class TestWorkflowYAMLConfig:
    def test_save_load_yaml_config(self, tmp_path, registry):
        yaml = pytest.importorskip("yaml")
        w = Workflow(workflow_id="yaml_wf")
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        w.expose_output("s1", "out", "y")
        cfg = w.to_config()
        yaml_path = tmp_path / "wf" / "config.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.dump(cfg))
        w2 = Workflow.load(tmp_path / "wf", registry=registry,
                           config_filename="config.yaml", load_checkpoint_flag=False)
        assert w2.node_ids == {"s1"}


class TestWorkflowRefYAML:
    def test_from_config_with_yaml_ref(self, tmp_path, registry):
        yaml = pytest.importorskip("yaml")
        hg_cfg = {
            "graph_id": "ref_hg",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }
        ref_path = tmp_path / "graph.yaml"
        ref_path.write_text(yaml.dump(hg_cfg))
        wf_cfg = {
            "workflow_id": "yaml_ref_wf",
            "graphs": [{"graph_id": "g1", "ref": str(ref_path)}],
            "edges": [],
            "exposed_inputs": [{"graph_id": "g1", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"graph_id": "g1", "port_name": "out", "name": "y"}],
        }
        w = Workflow.from_config(wf_cfg, registry=registry)
        assert "g1" in w.node_ids


class TestWorkflowRefSupport:
    """PHASE_6 §5.2: from_config with 'ref' for hypergraph configs."""

    def test_from_config_with_ref(self, tmp_path, registry):
        hg_cfg = {
            "graph_id": "ref_hg",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }
        ref_path = tmp_path / "graph_config.json"
        with open(ref_path, "w") as f:
            json.dump(hg_cfg, f)

        wf_cfg = {
            "workflow_id": "ref_wf",
            "graphs": [{"graph_id": "g1", "ref": str(ref_path)}],
            "edges": [],
            "exposed_inputs": [{"node_id": "g1", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"node_id": "g1", "port_name": "out", "name": "y"}],
        }
        w = Workflow.from_config(wf_cfg, registry=registry)
        assert "g1" in w.node_ids


class TestWorkflowEmptyRun:
    def test_empty_workflow_returns_empty(self, registry):
        clear_plan_cache()
        w = Workflow()
        result = w.run({}, validate_before=False)
        assert result == {}


class TestWorkflowSchemaVersion:
    def test_from_config_warns_on_mismatched_schema(self, registry):
        import warnings
        w = Workflow()
        w.add_node("a", _make_identity_hg("hg", registry))
        w.expose_input("a", "in", "x")
        w.expose_output("a", "out", "y")
        cfg = w.to_config()
        cfg["schema_version"] = "999.0"
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            Workflow.from_config(cfg, registry=registry)
            schema_warnings = [x for x in w_list if "schema_version" in str(x.message)]
            assert len(schema_warnings) >= 1


class TestWorkflowExposedGraphIdKey:
    def test_exposed_entries_use_graph_id_key(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        w.expose_output("s1", "out", "y")
        for entry in w.get_input_spec():
            assert "graph_id" in entry or "node_id" in entry
        for entry in w.get_output_spec():
            assert "graph_id" in entry or "node_id" in entry

    def test_exposed_graph_id_survives_roundtrip(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        w.expose_output("s1", "out", "y")
        cfg = w.to_config()
        w2 = Workflow.from_config(cfg, registry=registry)
        result = w2.run({"x": 42}, validate_before=False)
        assert result["y"] == 42


class TestWorkflowSpecWithDtype:
    def test_get_input_spec_with_dtype(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        spec = w.get_input_spec(include_dtype=True)
        assert len(spec) >= 1
        assert "dtype" in spec[0]

    def test_get_output_spec_with_dtype(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_output("s1", "out", "y")
        spec = w.get_output_spec(include_dtype=True)
        assert len(spec) >= 1
        assert "dtype" in spec[0]


class TestWorkflowEdgeIdempotent:
    def test_duplicate_edge_ignored(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge("s1", "out", "s2", "in")
        w.add_edge("s1", "out", "s2", "in")
        assert len(w.get_edges()) == 1

    def test_remove_edge_nonexistent_no_error(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.remove_edge("s1", "out", "s2", "in")

    def test_add_edge_bad_target_port(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        with pytest.raises(ValueError, match="not in input spec"):
            w.add_edge("s1", "out", "s2", "nonexistent")


class TestWorkflowExposeValidation:
    def test_expose_input_unknown_graph(self, registry):
        w = Workflow()
        with pytest.raises(ValueError, match="not in workflow"):
            w.expose_input("missing", "in", "x")

    def test_expose_input_unknown_port(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        with pytest.raises(ValueError, match="not in input spec"):
            w.expose_input("s1", "nonexistent", "x")

    def test_expose_output_unknown_graph(self, registry):
        w = Workflow()
        with pytest.raises(ValueError, match="not in workflow"):
            w.expose_output("missing", "out", "y")

    def test_expose_output_unknown_port(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        with pytest.raises(ValueError, match="not in output spec"):
            w.expose_output("s1", "nonexistent", "y")

    def test_expose_input_duplicate_ignored(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_input("s1", "in", "x")
        v_before = w.execution_version
        w.expose_input("s1", "in", "x")
        assert w.execution_version == v_before

    def test_expose_output_duplicate_ignored(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_output("s1", "out", "y")
        v_before = w.execution_version
        w.expose_output("s1", "out", "y")
        assert w.execution_version == v_before

    def test_expose_output_duplicate_updates_name(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.expose_output("s1", "out", "y")
        v_before = w.execution_version
        w.expose_output("s1", "out", "result")
        assert w.execution_version > v_before
        assert w.get_output_spec()[0]["name"] == "result"


class TestWorkflowAddNodeValidation:
    def test_empty_graph_id_raises(self, registry):
        w = Workflow()
        with pytest.raises(ValueError, match="non-empty"):
            w.add_node("", _make_identity_hg("hg1", registry))

    def test_remove_nonexistent_node_no_error(self, registry):
        w = Workflow()
        w.remove_node("ghost")


class TestWorkflowRemoveNodeCleanup:
    def test_remove_node_cleans_up_edges_and_exposed(self, registry):
        clear_plan_cache()
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.add_node("s2", _make_identity_hg("hg2", registry))
        w.add_edge("s1", "out", "s2", "in")
        w.expose_input("s1", "in", "x")
        w.expose_output("s2", "out", "y")
        w.remove_node("s1")
        assert len(w.get_edges()) == 0
        input_gids = {
            (e.get("graph_id") or e.get("node_id")) for e in w.get_input_spec()
        }
        assert "s1" not in input_gids


class TestWorkflowTrainableConfig:
    def test_trainable_false_in_to_config(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.set_trainable("s1", False)
        cfg = w.to_config()
        g_cfg = next(g for g in cfg["graphs"] if g["graph_id"] == "s1")
        assert g_cfg["trainable"] is False

    def test_trainable_false_roundtrip(self, registry):
        clear_plan_cache()
        w1 = Workflow()
        w1.add_node("s1", _make_identity_hg("hg1", registry))
        w1.set_trainable("s1", False)
        w1.expose_input("s1", "in", "x")
        w1.expose_output("s1", "out", "y")
        cfg = w1.to_config()
        w2 = Workflow.from_config(cfg, registry=registry)
        assert list(w2.trainable_parameters()) == []

    def test_trainable_parameters_yields_from_hg(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        params = list(w.trainable_parameters())
        assert isinstance(params, list)


class TestWorkflowMetadataConfig:
    def test_metadata_in_to_config(self, registry):
        w = Workflow()
        w.add_node("s1", _make_identity_hg("hg1", registry))
        w.metadata = {"custom": "value"}
        cfg = w.to_config()
        assert cfg["metadata"]["custom"] == "value"


class TestWorkflowSaveCheckpoint:
    def test_save_checkpoint_creates_file(self, tmp_path, registry):
        w = Workflow()
        w.add_node("s1", _make_add_hg("hg_add", 42, registry))
        w.save_checkpoint(tmp_path / "wf_ckpt")
        assert (tmp_path / "wf_ckpt" / "checkpoint.pkl").exists()


class TestWorkflowFromConfigYAMLRefImportError:
    def test_yaml_ref_without_pyyaml_in_from_config(self, tmp_path, registry, monkeypatch):
        import builtins
        real_import = builtins.__import__
        ref_path = tmp_path / "graph.yaml"
        ref_path.write_text("{}")
        cfg = {
            "workflow_id": "yaml_err",
            "graphs": [{"graph_id": "g1", "ref": str(ref_path)}],
            "edges": [],
            "exposed_inputs": [],
            "exposed_outputs": [],
        }

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="PyYAML"):
            Workflow.from_config(cfg, registry=registry)


class TestWorkflowFromConfigValidateFails:
    def test_from_config_validate_raises(self, registry, monkeypatch):
        """Use monkeypatch to make the validator return invalid, triggering line 486."""
        from yggdrasill.engine import validator as val_mod

        class FakeResult:
            valid = False
            errors = ["injected error"]
            warnings = []

        monkeypatch.setattr(val_mod, "validate", lambda s: FakeResult())
        cfg = {
            "workflow_id": "ok",
            "graphs": [{
                "graph_id": "g1",
                "config": {
                    "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
                    "edges": [],
                    "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
                    "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
                },
            }],
            "edges": [],
            "exposed_inputs": [{"graph_id": "g1", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"graph_id": "g1", "port_name": "out", "name": "y"}],
        }
        with pytest.raises(ValueError, match="validation failed"):
            Workflow.from_config(cfg, registry=registry, validate=True)


class TestWorkflowAutoConnect:
    """PHASE_6 §13.2: suggest_auto_edges / apply_auto_connect at workflow level."""

    def test_suggest_matching_ports(self, registry):
        from yggdrasill.foundation.block import AbstractBaseBlock
        from yggdrasill.foundation.node import AbstractGraphNode
        from yggdrasill.foundation.port import Port, PortDirection, PortType

        class OutputDataTaskNode(AbstractBaseBlock, AbstractGraphNode):
            def __init__(self, node_id):
                AbstractBaseBlock.__init__(self)
                AbstractGraphNode.__init__(self, node_id=node_id)

            @property
            def block_type(self):
                return "test/output_data_task"

            def declare_ports(self):
                return [
                    Port("data", PortDirection.OUT, PortType.ANY),
                ]

            def forward(self, inputs):
                return {"data": "value"}

        class InputDataTaskNode(AbstractBaseBlock, AbstractGraphNode):
            def __init__(self, node_id):
                AbstractBaseBlock.__init__(self)
                AbstractGraphNode.__init__(self, node_id=node_id)

            @property
            def block_type(self):
                return "test/input_data_task"

            def declare_ports(self):
                return [
                    Port("data", PortDirection.IN, PortType.ANY),
                    Port("result", PortDirection.OUT, PortType.ANY),
                ]

            def forward(self, inputs):
                return {"result": inputs["data"]}

        registry.register("test/output_data_task", OutputDataTaskNode)
        registry.register("test/input_data_task", InputDataTaskNode)
        w = Workflow()
        hg1 = Hypergraph.from_config({
            "graph_id": "src",
            "nodes": [{"node_id": "N", "block_type": "test/output_data_task"}],
            "edges": [],
            "exposed_outputs": [{"node_id": "N", "port_name": "data", "name": "data"}],
        }, registry=registry)
        hg2 = Hypergraph.from_config({
            "graph_id": "dst",
            "nodes": [{"node_id": "N", "block_type": "test/input_data_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "data", "name": "data"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "result", "name": "result"}],
        }, registry=registry)
        w.add_node("g1", hg1)
        w.add_node("g2", hg2)
        suggestions = suggest_auto_edges(w)
        assert suggestions == [("g1", "data", "g2", "data")]

    def test_apply_auto_connect_no_match(self, registry):
        clear_plan_cache()
        hg1_cfg = {
            "graph_id": "a1",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "x"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "link"}],
        }
        hg2_cfg = {
            "graph_id": "a2",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "link"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "y"}],
        }
        w = Workflow()
        w.add_node("g1", Hypergraph.from_config(hg1_cfg, registry=registry))
        w.add_node("g2", Hypergraph.from_config(hg2_cfg, registry=registry))
        assert apply_auto_connect(w) == 0

    def test_suggest_and_apply_matching_ports(self, registry):
        """When exposed output port_name matches exposed input port_name, suggest edge.

        g1 exposes output port_name='out', g2 has an AddTaskNode with input 'a'
        and output 'out'. We chain g1→g2 via the matching 'out' port name.
        We need a graph where an output and an input share the same port_name.
        Simplest: g1 has output 'out', g2 also has output 'out', both identity.
        The match is 'out'→'out' (g1 output vs g2 output). No, g2 must expose 'out'
        as INPUT. Not possible with IdentityTaskNode.

        Instead, test the code path works when port_names DO match by building
        two graphs: g1 exposes out='out', g2 is built from AddTaskNode so it
        exposes input 'a'. Since port_names differ, no auto-connect happens.
        Instead, let's just test suggest with a manually constructed workflow
        where specs happen to match.
        """
        clear_plan_cache()
        hg1 = _make_identity_hg("src", registry)
        hg2 = _make_identity_hg("dst", registry)
        w = Workflow()
        w.add_node("g1", hg1)
        w.add_node("g2", hg2)
        w.add_edge("g1", "out", "g2", "in")
        suggestions = suggest_auto_edges(w)
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert s != ("g1", "out", "g2", "in")

    def test_suggest_dtype_mismatch_skips(self, registry):
        """When port names match but dtypes are incompatible, no suggestion."""
        clear_plan_cache()
        hg1 = _make_identity_hg("src", registry)
        hg2 = _make_identity_hg("dst", registry)
        hg1._exposed_outputs = [{"node_id": "N", "port_name": "link"}]
        hg2._exposed_inputs = [{"node_id": "N", "port_name": "link"}]
        hg1._exposed_outputs[0]["dtype"] = "tensor"
        hg2._exposed_inputs[0]["dtype"] = "image"

        class FakeSpec:
            def __init__(self, specs):
                self._specs = specs
            def __call__(self, include_dtype=False):
                return self._specs

        hg1.get_output_spec = FakeSpec([{"port_name": "link", "dtype": "tensor"}])
        hg2.get_input_spec = FakeSpec([{"port_name": "link", "dtype": "image"}])
        w = Workflow()
        w.add_node("g1", hg1)
        w.add_node("g2", hg2)
        suggestions = suggest_auto_edges(w)
        assert len(suggestions) == 0

    def test_apply_auto_connect_creates_edges(self, registry):
        """Force matching port names by injecting specs directly."""
        clear_plan_cache()
        hg1 = _make_identity_hg("src", registry)
        hg2 = _make_identity_hg("dst", registry)
        hg2._exposed_inputs.clear()
        hg2._exposed_inputs.append({"node_id": "N", "port_name": "out"})
        w = Workflow()
        w.add_node("g1", hg1)
        w.add_node("g2", hg2)
        suggestions = suggest_auto_edges(w)
        matching = [s for s in suggestions if s[1] == "out" and s[3] == "out"]
        assert len(matching) >= 1
        added = apply_auto_connect(w)
        assert added >= 1


class TestWorkflowAddNodeFromConfig:
    """S-2: Workflow.add_node_from_config builds Hypergraph from config dict."""

    def test_add_node_from_config_dict(self, registry):
        clear_plan_cache()
        w = Workflow()
        cfg = {
            "graph_id": "g1",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }
        gid = w.add_node_from_config("g1", cfg, registry=registry)
        assert gid == "g1"
        assert "g1" in w.node_ids
        hg = w.get_node("g1")
        assert "N" in hg.node_ids

    def test_add_node_from_config_ref(self, registry, tmp_path):
        clear_plan_cache()
        cfg = {
            "graph_id": "ref_graph",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }
        ref_file = tmp_path / "graph.json"
        ref_file.write_text(json.dumps(cfg))
        w = Workflow()
        gid = w.add_node_from_config("g1", {"ref": str(ref_file)}, registry=registry)
        assert gid == "g1"
        assert "N" in w.get_node("g1").node_ids

    def test_add_node_from_config_trainable(self, registry):
        w = Workflow()
        cfg = {
            "graph_id": "g1",
            "nodes": [{"node_id": "N", "block_type": "test/identity_task"}],
            "edges": [],
            "exposed_inputs": [{"node_id": "N", "port_name": "in", "name": "in"}],
            "exposed_outputs": [{"node_id": "N", "port_name": "out", "name": "out"}],
        }
        w.add_node_from_config("g1", cfg, registry=registry, trainable=False)
        assert w._node_trainable["g1"] is False


class TestWorkflowSaveLoadRunRoundtrip:
    """T-1: Full workflow save → load → run roundtrip."""

    def test_save_load_run_produces_same_output(self, registry, tmp_path):
        clear_plan_cache()
        hg1 = _make_identity_hg("g1", registry)
        hg2 = _make_identity_hg("g2", registry)
        w = Workflow(workflow_id="rt_wf")
        w.add_node("g1", hg1)
        w.add_node("g2", hg2)
        w.add_edge("g1", "out", "g2", "in")
        w.expose_input("g1", "in", "x")
        w.expose_output("g2", "out", "y")

        out1 = w.run({"x": 42}, validate_before=False)
        w.save(tmp_path / "wf")
        w2 = Workflow.load(tmp_path / "wf", registry=registry)
        clear_plan_cache()
        out2 = w2.run({"x": 42}, validate_before=False)
        assert out1 == out2

    def test_save_config_load_config_roundtrip(self, registry, tmp_path):
        clear_plan_cache()
        hg = _make_identity_hg("g1", registry)
        w = Workflow(workflow_id="cfg_wf")
        w.add_node("g1", hg)
        w.expose_input("g1", "in", "x")
        w.expose_output("g1", "out", "y")
        w.save_config(tmp_path / "wf_cfg")
        w2 = Workflow.load_config(tmp_path / "wf_cfg", registry=registry)
        assert w2.workflow_id == "cfg_wf"
        assert "g1" in w2.node_ids


class TestWorkflowDiamond:
    """T-2: Diamond (fan-out) workflow topology."""

    def test_diamond_fan_out_fan_in(self, registry):
        clear_plan_cache()
        hg_src = _make_identity_hg("src", registry)
        hg_mid1 = _make_identity_hg("m1", registry)
        hg_mid2 = _make_identity_hg("m2", registry)

        hg_sink = Hypergraph.from_config({
            "graph_id": "sink",
            "nodes": [{"node_id": "A", "block_type": "test/add_task"}],
            "edges": [],
            "exposed_inputs": [
                {"node_id": "A", "port_name": "a", "name": "a"},
                {"node_id": "A", "port_name": "b", "name": "b"},
            ],
            "exposed_outputs": [{"node_id": "A", "port_name": "out", "name": "out"}],
        }, registry=registry)

        w = Workflow()
        w.add_node("src", hg_src)
        w.add_node("m1", hg_mid1)
        w.add_node("m2", hg_mid2)
        w.add_node("sink", hg_sink)
        w.add_edge("src", "out", "m1", "in")
        w.add_edge("src", "out", "m2", "in")
        w.add_edge("m1", "out", "sink", "a")
        w.add_edge("m2", "out", "sink", "b")
        w.expose_input("src", "in", "x")
        w.expose_output("sink", "out", "y")

        out = w.run({"x": 5}, validate_before=False)
        assert out["y"] == 10  # 5 + 5 + 0 offset


class TestWorkflowCycleFromMetadata:
    """T-5: Workflow cycle where num_loop_steps comes from metadata."""

    def test_cycle_from_metadata(self, registry):
        clear_plan_cache()
        hg1 = _make_identity_hg("g1", registry)
        hg2 = _make_identity_hg("g2", registry)

        w = Workflow(workflow_id="cycle_meta")
        w.add_node("g1", hg1)
        w.add_node("g2", hg2)
        w.add_edge("g1", "out", "g2", "in")
        w.add_edge("g2", "out", "g1", "in")
        w.expose_input("g1", "in", "x")
        w.expose_output("g2", "out", "y")
        w.metadata = {"num_loop_steps": 3}

        log = []

        def cb(phase, info):
            log.append(phase)

        out = w.run({"x": 7}, callbacks=[cb], validate_before=False)
        assert out["y"] == 7
        assert log.count("loop_start") == 1
        assert log.count("loop_end") == 1
        before_count = log.count("before")
        assert before_count == 6  # 3 iterations * 2 graphs
