"""Tests for SD1.5 graph builders: text2img, img2img, inpaint."""
from __future__ import annotations

import pytest

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.presets.sd15 import (
    build_sd15_text2img_graph,
    build_sd15_img2img_graph,
    build_sd15_inpaint_graph,
)

from tests.diffusion.conftest import (
    FakeScheduler,
    FakeTextEncoder,
    FakeTokenizer,
    FakeUNet,
    FakeVAE,
)


@pytest.fixture
def sd15_kwargs():
    return {
        "tokenizer": FakeTokenizer(),
        "text_encoder": FakeTextEncoder(),
        "unet": FakeUNet(),
        "vae": FakeVAE(),
        "scheduler": FakeScheduler(),
    }


class TestSD15Text2ImgGraph:

    def test_builds_successfully(self, sd15_kwargs):
        g = build_sd15_text2img_graph(**sd15_kwargs)
        assert g.graph_id == "sd15_text2img"

    def test_has_required_nodes(self, sd15_kwargs):
        g = build_sd15_text2img_graph(**sd15_kwargs)
        assert "tokenizer" in g.node_ids
        assert "prompt_enc" in g.node_ids
        assert "unet" in g.node_ids
        assert "vae_decode" in g.node_ids
        assert "latent_init" in g.node_ids
        assert "sched_setup" in g.node_ids
        assert "sched_step" in g.node_ids

    def test_exposed_inputs(self, sd15_kwargs):
        g = build_sd15_text2img_graph(**sd15_kwargs)
        spec = g.get_input_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_PROMPT in port_names
        assert C.PORT_NEGATIVE_PROMPT in port_names

    def test_exposed_outputs(self, sd15_kwargs):
        g = build_sd15_text2img_graph(**sd15_kwargs)
        spec = g.get_output_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_DECODED_IMAGE in port_names

    def test_edges_count(self, sd15_kwargs):
        g = build_sd15_text2img_graph(**sd15_kwargs)
        edges = g.get_edges()
        assert len(edges) >= 6

    def test_config_roundtrip(self, sd15_kwargs):
        g = build_sd15_text2img_graph(**sd15_kwargs)
        cfg = g.to_config()
        assert cfg["graph_id"] == "sd15_text2img"
        assert len(cfg["nodes"]) >= 6
        assert len(cfg["edges"]) >= 6

    def test_custom_steps(self, sd15_kwargs):
        g = build_sd15_text2img_graph(**sd15_kwargs, config={"num_inference_steps": 20})
        assert g.metadata.get("num_loop_steps") == 20

    def test_with_safety(self, sd15_kwargs):
        from tests.diffusion.conftest import FakeFeatureExtractor

        class FakeSafetyChecker:
            device = "cpu"
            def to(self, *args):
                return self

        sd15_kwargs["safety_checker"] = FakeSafetyChecker()
        sd15_kwargs["feature_extractor"] = FakeFeatureExtractor()
        g = build_sd15_text2img_graph(**sd15_kwargs, config={"enable_safety": True})
        assert "safety" in g.node_ids


class TestSD15Img2ImgGraph:

    def test_builds_successfully(self, sd15_kwargs):
        g = build_sd15_img2img_graph(**sd15_kwargs)
        assert g.graph_id == "sd15_img2img"

    def test_has_image_encode_node(self, sd15_kwargs):
        g = build_sd15_img2img_graph(**sd15_kwargs)
        assert "img_encode" in g.node_ids

    def test_exposed_inputs_include_image(self, sd15_kwargs):
        g = build_sd15_img2img_graph(**sd15_kwargs)
        spec = g.get_input_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_INIT_IMAGE in port_names
        assert C.PORT_PROMPT in port_names

    def test_img_encode_to_latent_init_edge(self, sd15_kwargs):
        g = build_sd15_img2img_graph(**sd15_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_node == "img_encode" and e.target_node == "latent_init"
            and e.source_port == C.PORT_LATENTS and e.target_port == C.PORT_INIT_LATENTS
            for e in edges
        )
        assert found, "img_encode → latent_init edge missing"


class TestSD15InpaintGraph:

    def test_builds_successfully(self, sd15_kwargs):
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        assert g.graph_id == "sd15_inpaint"

    def test_has_mask_prep_node(self, sd15_kwargs):
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        assert "mask_prep" in g.node_ids

    def test_exposed_inputs_include_mask(self, sd15_kwargs):
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        spec = g.get_input_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_MASK_IMAGE in port_names
        assert C.PORT_INIT_IMAGE in port_names
