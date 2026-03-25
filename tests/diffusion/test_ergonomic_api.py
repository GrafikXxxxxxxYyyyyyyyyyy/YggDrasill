"""Tests for the ergonomic Hypergraph API: kwargs run, add_node with type/pretrained,
replace_node, port-name auto-connect, DiffusionOutput, and multi-adapter wiring.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yggdrasill import Hypergraph
from yggdrasill.integrations.diffusers.output import DiffusionOutput
from yggdrasill.integrations.diffusers.components import (
    COMPONENT_REGISTRY,
    resolve_component_type,
    is_component_type,
    is_block_type,
)
from yggdrasill.engine.edge import Edge
from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect


# ---------------------------------------------------------------------------
# Test: top-level import
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_yggdrasill(self):
        from yggdrasill import Hypergraph as H
        assert H is Hypergraph

    def test_hypergraph_is_engine_structure(self):
        from yggdrasill.engine.structure import Hypergraph as Direct
        assert Hypergraph is Direct


# ---------------------------------------------------------------------------
# Test: Hypergraph(name=...)
# ---------------------------------------------------------------------------


class TestNameAlias:
    def test_name_kwarg(self):
        g = Hypergraph(name="MyGraph")
        assert g.graph_id == "MyGraph"

    def test_positional_graph_id(self):
        g = Hypergraph("LegacyId")
        assert g.graph_id == "LegacyId"

    def test_name_takes_precedence(self):
        g = Hypergraph("LegacyId", name="Preferred")
        assert g.graph_id == "Preferred"

    def test_default_graph_id(self):
        g = Hypergraph()
        assert g.graph_id == "graph"


# ---------------------------------------------------------------------------
# Test: DiffusionOutput
# ---------------------------------------------------------------------------


class TestDiffusionOutput:
    def test_from_executor_output_single_image(self):
        raw = {"decoded_image": "img_obj", "latents": "lat_obj"}
        out = DiffusionOutput.from_executor_output(raw)
        assert out.images == ["img_obj"]
        assert out.latents == "lat_obj"
        assert out.raw is raw

    def test_from_executor_output_list_images(self):
        raw = {"decoded_image": ["img1", "img2"]}
        out = DiffusionOutput.from_executor_output(raw)
        assert out.images == ["img1", "img2"]

    def test_from_executor_output_no_images(self):
        raw = {"some_other_key": 42}
        out = DiffusionOutput.from_executor_output(raw)
        assert out.images == []

    def test_custom_image_key(self):
        raw = {"output_image": "custom_img"}
        out = DiffusionOutput.from_executor_output(raw, image_key="output_image")
        assert out.images == ["custom_img"]

    def test_nsfw_flag(self):
        raw = {"decoded_image": "img", "nsfw_content_detected": [False, True]}
        out = DiffusionOutput.from_executor_output(raw)
        assert out.nsfw_content_detected == [False, True]


# ---------------------------------------------------------------------------
# Test: Component Registry
# ---------------------------------------------------------------------------


class TestComponentRegistry:
    def test_sdxl_prompt_encoder_alias_spec(self):
        spec = resolve_component_type("sdxl.prompt_encoder")
        assert spec.block_types == ["sdxl/prompt_encoder"]
        assert set(spec.load_keys) == {"text_encoder", "text_encoder_2"}

    def test_sdxl_backbone_alias_spec(self):
        spec = resolve_component_type("sdxl.backbone")
        assert spec.block_types == ["sdxl/unet"]

    def test_sdxl_autoencoder_alias_spec(self):
        spec = resolve_component_type("sdxl.autoencoder")
        assert spec.block_types == ["sdxl/vae_decode"]

    def test_sdxl_unet_spec(self):
        spec = resolve_component_type("sdxl.unet")
        assert spec.block_types == ["sdxl/unet"]
        assert spec.load_keys == ["unet"]

    def test_sdxl_scheduler_creates_two_nodes(self):
        spec = resolve_component_type("sdxl.scheduler")
        assert len(spec.block_types) == 2
        assert "sdxl/scheduler_setup" in spec.block_types
        assert "sdxl/scheduler_step" in spec.block_types

    def test_sdxl_tokenizer_is_converter_node(self):
        """Tokenizer is a standalone Converter node (canon: converter/tokenizer)."""
        spec = resolve_component_type("sdxl.tokenizer")
        assert spec.block_types == ["sdxl/tokenizer"]
        assert spec.group is None

    def test_sdxl_text_encoder_in_prompt_encoder_group(self):
        """Text encoder is part of prompt_encoder Conjector group."""
        e_spec = resolve_component_type("sdxl.text_encoder")
        assert e_spec.group == "sdxl.prompt_encoder"

    def test_adapter_controlnet(self):
        spec = resolve_component_type("adapter.controlnet")
        assert spec.block_types == ["adapter/controlnet"]

    def test_sdxl_controlnet(self):
        spec = resolve_component_type("sdxl.controlnet")
        assert spec.block_types == ["adapter/controlnet"]
        assert spec.load_family == "sdxl"

    def test_sdxl_ipadapter(self):
        spec = resolve_component_type("sdxl.ipadapter")
        assert spec.block_types == ["adapter/ip_adapter"]
        assert spec.load_subfolder_map.get("image_encoder") == "sdxl_models/image_encoder"

    def test_adapter_controlnet_flux(self):
        spec = resolve_component_type("adapter.controlnet_flux")
        assert spec.block_types == ["adapter/controlnet_flux"]
        assert spec.load_family == "flux"

    def test_flux_controlnet_backward_compat(self):
        """flux.controlnet resolves to adapter/controlnet_flux for backward compat."""
        spec = resolve_component_type("flux.controlnet")
        assert spec.block_types == ["adapter/controlnet_flux"]
        assert spec.load_family == "flux"

    def test_flux_transformer(self):
        spec = resolve_component_type("flux.transformer")
        assert spec.block_types == ["flux/transformer"]

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown component type"):
            resolve_component_type("unknown.thing")

    def test_is_component_type(self):
        assert is_component_type("sdxl.unet") is True
        assert is_component_type("sdxl/unet") is False
        assert is_component_type("plain") is False

    def test_is_block_type(self):
        assert is_block_type("sdxl/unet") is True
        assert is_block_type("sdxl.unet") is False

    def test_all_families_covered(self):
        families = {k.split(".")[0] for k in COMPONENT_REGISTRY}
        assert families >= {"sd15", "sdxl", "flux", "adapter"}


# ---------------------------------------------------------------------------
# Test: Port-name auto-connect
# ---------------------------------------------------------------------------


class TestPortNameAutoConnect:
    """Test port-name matching on a simple synthetic graph."""

    def _make_simple_node(self, node_id, in_ports, out_ports):
        from yggdrasill.foundation.port import Port, PortDirection, PortType
        from yggdrasill.task_nodes.abstract import AbstractHelper

        class _DummyNode(AbstractHelper):
            _in = in_ports
            _out = out_ports

            @property
            def block_type(self):
                return "helper/test"

            def declare_ports(self):
                ports = []
                for name in self._in:
                    ports.append(Port(name, PortDirection.IN, PortType.ANY))
                for name in self._out:
                    ports.append(Port(name, PortDirection.OUT, PortType.ANY))
                return ports

            def forward(self, inputs):
                return {}

        return _DummyNode(node_id=node_id)

    def test_matching_port_names_create_edges(self):
        g = Hypergraph(name="test")
        node_a = self._make_simple_node("a", [], ["x", "y"])
        node_b = self._make_simple_node("b", ["x", "z"], [])

        g.add_node("a", node_a)
        g.add_node("b", node_b)
        added = apply_port_name_auto_connect(g, "b", node_b)

        assert added == 1
        edges = g.get_edges()
        assert len(edges) == 1
        assert edges[0].source_node == "a"
        assert edges[0].source_port == "x"
        assert edges[0].target_node == "b"
        assert edges[0].target_port == "x"

    def test_no_match_no_edges(self):
        g = Hypergraph(name="test")
        node_a = self._make_simple_node("a", [], ["x"])
        node_b = self._make_simple_node("b", ["y"], [])

        g.add_node("a", node_a)
        g.add_node("b", node_b)
        added = apply_port_name_auto_connect(g, "b", node_b)

        assert added == 0
        assert len(g.get_edges()) == 0

    def test_alias_next_latent_to_latents(self):
        g = Hypergraph(name="test")
        node_a = self._make_simple_node("a", [], ["next_latent"])
        node_b = self._make_simple_node("b", ["latents"], [])

        g.add_node("a", node_a)
        g.add_node("b", node_b)
        added = apply_port_name_auto_connect(g, "a", node_a)

        assert added == 1
        edge = g.get_edges()[0]
        assert edge.source_port == "next_latent"
        assert edge.target_port == "latents"

    def test_reverse_alias_latents_to_next_latent(self):
        g = Hypergraph(name="test")
        node_a = self._make_simple_node("a", ["latents"], [])
        node_b = self._make_simple_node("b", [], ["next_latent"])

        g.add_node("b", node_b)
        g.add_node("a", node_a)
        added = apply_port_name_auto_connect(g, "a", node_a)

        assert added == 1


# ---------------------------------------------------------------------------
# Test: Enhanced add_node (polymorphic)
# ---------------------------------------------------------------------------


class TestAddNodePolymorphic:
    def test_add_raw_node(self):
        """Existing API: add_node(id, object) still works."""
        g = Hypergraph(name="test")
        obj = MagicMock()
        g.add_node("n1", obj)
        assert g.get_node("n1") is obj

    def test_add_node_requires_type_or_object(self):
        g = Hypergraph(name="test")
        with pytest.raises(ValueError, match="node object.*type="):
            g.add_node("n1")

    def test_add_node_rejects_bad_type_format(self):
        g = Hypergraph(name="test")
        with pytest.raises(ValueError, match="family/block"):
            g.add_node("n1", type="noslash_nodot")

    def test_add_node_with_block_type(self):
        """Block-level type should build via BlockRegistry."""
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("lat", type="sdxl/latent_init", auto_connect=False)
        node = g.get_node("lat")
        assert node is not None
        assert node.block_type == "sdxl/latent_init"

    def test_add_node_empty_id_raises(self):
        g = Hypergraph(name="test")
        with pytest.raises(ValueError, match="non-empty"):
            g.add_node("")

    @pytest.mark.skip(reason="Component-type add_node removed; use DiffusionGraphBuilder")
    def test_sdxl_component_api_auto_adds_helper_nodes(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes

        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("unet", type="sdxl.unet", auto_connect=False)

        block_types = {
            getattr(node, "block_type", None)
            for node in (g.get_node(nid) for nid in g.node_ids)
        }
        assert "sdxl/unet" in block_types
        assert "sdxl/latent_init" in block_types
        assert "sdxl/added_conditioning" in block_types

    @pytest.mark.skip(reason="Component-type add_node removed; use DiffusionGraphBuilder")
    def test_sdxl_helper_nodes_are_not_duplicated(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes

        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("unet", type="sdxl.unet", auto_connect=False)
        g.add_node("vae", type="sdxl.vae", auto_connect=False)
        g.add_node("scheduler", type="sdxl.scheduler", auto_connect=False)

        block_types = [
            getattr(g.get_node(nid), "block_type", None)
            for nid in g.node_ids
        ]
        assert block_types.count("sdxl/latent_init") == 1
        assert block_types.count("sdxl/added_conditioning") == 1


# ---------------------------------------------------------------------------
# Test: replace_node
# ---------------------------------------------------------------------------


class TestReplaceNode:
    def _build_graph_with_two_nodes(self):
        from yggdrasill.foundation.port import Port, PortDirection, PortType
        from yggdrasill.task_nodes.abstract import AbstractHelper

        class _Producer(AbstractHelper):
            @property
            def block_type(self):
                return "helper/prod"

            def declare_ports(self):
                return [Port("x", PortDirection.OUT, PortType.ANY)]

            def forward(self, inputs):
                return {"x": 1}

        class _Consumer(AbstractHelper):
            @property
            def block_type(self):
                return "helper/cons"

            def declare_ports(self):
                return [Port("x", PortDirection.IN, PortType.ANY)]

            def forward(self, inputs):
                return {}

        g = Hypergraph(name="test")
        prod = _Producer(node_id="prod")
        cons = _Consumer(node_id="cons")
        g.add_node("prod", prod)
        g.add_node("cons", cons)
        g.add_edge(Edge("prod", "x", "cons", "x"))
        return g, _Producer, _Consumer

    def test_replace_with_compatible_node(self):
        g, Producer, _ = self._build_graph_with_two_nodes()
        new_prod = Producer(node_id="prod")

        g.replace_node("prod", node=new_prod)
        assert g.get_node("prod") is new_prod
        edges = g.get_edges()
        assert len(edges) == 1
        assert edges[0].source_node == "prod"

    def test_replace_nonexistent_raises(self):
        g = Hypergraph(name="test")
        with pytest.raises(ValueError, match="not in graph"):
            g.replace_node("missing", node=MagicMock())

    def test_replace_no_args_raises(self):
        g = Hypergraph(name="test")
        g.add_node("n", MagicMock())
        with pytest.raises(ValueError, match="at least one"):
            g.replace_node("n")

    def test_replace_uses_aliases_from_metadata(self):
        g, Producer, _ = self._build_graph_with_two_nodes()
        g.metadata = {"node_aliases": {"backbone": "prod"}}
        new_prod = Producer(node_id="prod")

        g.replace_node("backbone", node=new_prod)
        assert g.get_node("prod") is new_prod

    @pytest.mark.skip(reason="_replace_pretrained_only removed from engine")
    def test_replace_scheduler_alias_targets_multiple_nodes(self):
        g = Hypergraph(name="test")
        g.add_node("sched_setup", MagicMock(block_type="sdxl/scheduler_setup"))
        g.add_node("sched_step", MagicMock(block_type="sdxl/scheduler_step"))
        g.metadata = {
            "node_aliases": {
                "scheduler": ["sched_setup", "sched_step"],
            },
        }

        with patch.object(g, "_replace_pretrained_only") as replace_mock:
            g.replace_node("scheduler", pretrained="repo")

        assert replace_mock.call_count == 2


# ---------------------------------------------------------------------------
# Test: Kwargs-based run
# ---------------------------------------------------------------------------


class TestKwargsRun:
    def test_run_with_dict_inputs(self):
        """Original dict-based run still works."""
        g = Hypergraph(name="test")
        g._exposed_inputs = []
        g._exposed_outputs = []

        with patch("yggdrasill.engine.executor.run") as mock_run:
            mock_run.return_value = {"result": 42}
            out = g.run({"prompt": "hello"})
            assert mock_run.called
            assert out == {"result": 42}

    def test_run_with_kwargs(self):
        """Kwargs should be routed to inputs."""
        g = Hypergraph(name="test")
        g._exposed_inputs = [{"node_id": "enc", "port_name": "prompt"}]
        g._exposed_outputs = []

        with patch("yggdrasill.engine.executor.run") as mock_run:
            mock_run.return_value = {"out": "val"}
            g.run(prompt="hello world")
            call_args = mock_run.call_args
            inputs_dict = call_args[0][1]
            assert inputs_dict["prompt"] == "hello world"

    def test_num_inference_steps_maps_to_num_loop_steps(self):
        g = Hypergraph(name="test")
        g._exposed_inputs = []
        g._exposed_outputs = []

        with patch("yggdrasill.engine.executor.run") as mock_run:
            mock_run.return_value = {}
            g.run(num_inference_steps=25)
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["num_loop_steps"] == 25

    def test_seed_forwarded(self):
        g = Hypergraph(name="test")
        g._exposed_inputs = []
        g._exposed_outputs = []

        with patch("yggdrasill.engine.executor.run") as mock_run:
            mock_run.return_value = {}
            g.run(seed=42)
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["seed"] == 42

    def test_diffusion_output_wrapping(self):
        """Use run_diffusion from addon to get DiffusionOutput; graph.run() returns raw dict."""
        from yggdrasill.integrations.diffusers.run import run as run_diffusion

        g = Hypergraph(name="test")
        g._exposed_inputs = []
        g._exposed_outputs = [
            {"node_id": "vae", "port_name": "decoded_image"},
        ]

        with patch("yggdrasill.engine.executor.run") as mock_run:
            mock_run.return_value = {"decoded_image": "fake_image"}
            out = run_diffusion(g)
            assert isinstance(out, DiffusionOutput)
            assert out.images == ["fake_image"]

    def test_non_diffusion_returns_dict(self):
        """Non-diffusion graphs return raw dict."""
        g = Hypergraph(name="test")
        g._exposed_inputs = []
        g._exposed_outputs = [
            {"node_id": "n", "port_name": "result"},
        ]

        with patch("yggdrasill.engine.executor.run") as mock_run:
            mock_run.return_value = {"result": 99}
            out = g.run()
            assert isinstance(out, dict)
            assert out["result"] == 99


# ---------------------------------------------------------------------------
# Test: Multi-adapter port aggregation
# ---------------------------------------------------------------------------


class TestMultiAdapterAggregation:
    def test_sdxl_unet_down_block_port_is_concat(self):
        from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
        from yggdrasill.foundation.port import PortAggregation

        node = SDXLUNetNode(node_id="unet")
        port = node.get_port("down_block_additional_residuals")
        assert port is not None
        assert port.aggregation == PortAggregation.CONCAT

    def test_sdxl_unet_mid_block_port_is_concat(self):
        from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
        from yggdrasill.foundation.port import PortAggregation

        node = SDXLUNetNode(node_id="unet")
        port = node.get_port("mid_block_additional_residual")
        assert port is not None
        assert port.aggregation == PortAggregation.CONCAT

    def test_sdxl_unet_image_embeds_is_concat(self):
        from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
        from yggdrasill.foundation.port import PortAggregation

        node = SDXLUNetNode(node_id="unet")
        port = node.get_port("image_embeds")
        assert port is not None
        assert port.aggregation == PortAggregation.CONCAT

    def test_sd15_unet_down_block_port_is_concat(self):
        from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
        from yggdrasill.foundation.port import PortAggregation

        node = SD15UNetNode(node_id="unet")
        port = node.get_port("down_block_additional_residuals")
        assert port is not None
        assert port.aggregation == PortAggregation.CONCAT

    def test_sd15_unet_image_embeds_is_concat(self):
        from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
        from yggdrasill.foundation.port import PortAggregation

        node = SD15UNetNode(node_id="unet")
        port = node.get_port("image_embeds")
        assert port is not None
        assert port.aggregation == PortAggregation.CONCAT
        assert port.optional

    def test_flux_transformer_controlnet_ports_are_concat(self):
        from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
        from yggdrasill.foundation.port import PortAggregation

        node = FluxTransformerNode(node_id="tr")
        p1 = node.get_port("controlnet_block_samples")
        p2 = node.get_port("controlnet_single_block_samples")
        assert p1 is not None and p1.aggregation == PortAggregation.CONCAT
        assert p2 is not None and p2.aggregation == PortAggregation.CONCAT

    def testmerge_residuals_single_value(self):
        from yggdrasill.integrations.diffusers.common.merge import merge_residuals
        val = (1, 2, 3)
        assert merge_residuals(val) == (1, 2, 3)

    def testmerge_residuals_list_of_one(self):
        from yggdrasill.integrations.diffusers.common.merge import merge_residuals
        val = [(1, 2, 3)]
        assert merge_residuals(val) == (1, 2, 3)

    def testmerge_residuals_scalar(self):
        from yggdrasill.integrations.diffusers.common.merge import merge_residuals
        assert merge_residuals(42) == 42


# ---------------------------------------------------------------------------
# Test: Full graph wiring via add_node with auto-connect
# ---------------------------------------------------------------------------


class TestFullGraphAutoConnect:
    """Build a small SDXL graph using add_node with block types and verify wiring."""

    @pytest.mark.skip(reason="Component-type add_node removed; use DiffusionGraphBuilder")
    def test_component_api_full_sdxl_build_without_explicit_helpers(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes

        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("backbone", type="sdxl.unet", auto_connect=False)
        g.add_node("scheduler", type="sdxl.scheduler")
        g.add_node("tokenizer", type="sdxl.tokenizer")
        g.add_node("prompt", type="sdxl.prompt_encoder")
        g.add_node("vae", type="sdxl.vae")

        block_types = {
            getattr(g.get_node(nid), "block_type", None)
            for nid in g.node_ids
        }
        assert "sdxl/added_conditioning" in block_types
        assert "sdxl/latent_init" in block_types

        port_pairs = {
            (e.source_port, e.target_port) for e in g.get_edges()
        }
        assert ("pooled_prompt_embeds", "pooled_prompt_embeds") in port_pairs or (
            "pooled_prompt_embeds",
            "add_text_embeds",
        ) in port_pairs
        assert ("scheduler_state", "scheduler_state") in port_pairs
        assert ("latents", "latents") in port_pairs

    @pytest.mark.skip(reason="Component-type add_node removed; use DiffusionGraphBuilder")
    def test_component_api_supports_large_user_facing_sdxl_types(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes

        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("prompt", type="sdxl.prompt_encoder", auto_connect=False)
        g.add_node("backbone", type="sdxl.backbone")
        g.add_node("scheduler", type="sdxl.scheduler")
        g.add_node("autoencoder", type="sdxl.autoencoder")

        block_types = {
            getattr(g.get_node(nid), "block_type", None)
            for nid in g.node_ids
        }
        assert "sdxl/prompt_encoder" in block_types
        assert "sdxl/unet" in block_types
        assert "sdxl/vae_decode" in block_types

    def test_prompt_encoder_to_unet_auto_wiring(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("enc", type="sdxl/prompt_encoder", auto_connect=False)
        g.add_node("unet", type="sdxl/unet")

        edges = g.get_edges()
        port_pairs = {(e.source_port, e.target_port) for e in edges}

        assert ("prompt_embeds", "prompt_embeds") in port_pairs
        assert ("negative_prompt_embeds", "negative_prompt_embeds") in port_pairs

    def test_added_cond_to_unet(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("enc", type="sdxl/prompt_encoder", auto_connect=False)
        g.add_node("cond", type="sdxl/added_conditioning")
        g.add_node("unet", type="sdxl/unet")

        edges = g.get_edges()
        pairs = {(e.source_node, e.target_node, e.source_port) for e in edges}

        assert ("cond", "unet", "add_text_embeds") in pairs
        assert ("cond", "unet", "add_time_ids") in pairs

    def test_controlnet_auto_wires_to_unet(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("unet", type="sdxl/unet", auto_connect=False)
        g.add_node("cn", type="adapter/controlnet")

        edges = g.get_edges()
        cn_to_unet = [e for e in edges if e.source_node == "cn" and e.target_node == "unet"]
        out_ports = {e.source_port for e in cn_to_unet}
        assert "down_block_additional_residuals" in out_ports
        assert "mid_block_additional_residual" in out_ports

    def test_two_controlnets_both_wire_to_unet(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes()

        g = Hypergraph(name="test")
        g.add_node("unet", type="sdxl/unet", auto_connect=False)
        g.add_node("cn1", type="adapter/controlnet")
        g.add_node("cn2", type="adapter/controlnet")

        edges = g.get_edges()
        down_edges = [
            e for e in edges
            if e.target_port == "down_block_additional_residuals"
            and e.target_node == "unet"
        ]
        assert len(down_edges) == 2

    def test_infer_exposed_ports_after_build(self):
        from yggdrasill.integrations.diffusers import contracts as C
        from yggdrasill.engine.edge import Edge
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        from yggdrasill.integrations.diffusers.sdxl.tokenizer import SDXLTokenizerNode
        from tests.diffusion.conftest import FakeTokenizer

        register_diffusion_nodes()

        g = Hypergraph(name="test")
        tok = SDXLTokenizerNode(
            "tok",
            tokenizer=FakeTokenizer(),
            tokenizer_2=FakeTokenizer(),
        )
        g.add_node("tok", tok)
        g.add_node("enc", type="sdxl/prompt_encoder", auto_connect=False)
        g.add_node("unet", type="sdxl/unet")
        g.add_edge(Edge("tok", C.PORT_INPUT_IDS, "enc", C.PORT_INPUT_IDS))
        g.add_edge(Edge("tok", C.PORT_INPUT_IDS_2, "enc", C.PORT_INPUT_IDS_2))
        g.add_edge(Edge("tok", C.PORT_NEGATIVE_INPUT_IDS, "enc", C.PORT_NEGATIVE_INPUT_IDS))
        g.add_edge(Edge("tok", C.PORT_NEGATIVE_INPUT_IDS_2, "enc", C.PORT_NEGATIVE_INPUT_IDS_2))
        g.infer_exposed_ports()

        input_names = {
            spec["port_name"] for spec in g.get_input_spec()
        }
        assert "prompt" in input_names
        assert "latents" in input_names
