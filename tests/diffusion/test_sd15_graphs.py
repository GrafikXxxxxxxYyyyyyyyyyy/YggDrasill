"""Tests for SD1.5 graph builders: text2img, img2img, inpaint."""
from __future__ import annotations

import pytest

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.presets.sd15 import (
    build_sd15_text2img_graph,
    build_sd15_img2img_graph,
    build_sd15_inpaint_graph,
    reconfigure_sd15_inpaint_for_unet_in_channels,
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

    def test_has_img_encode_for_full_image_latents(self, sd15_kwargs):
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        assert "img_encode" in g.node_ids

    def test_img_encode_to_latent_init_for_inpaint(self, sd15_kwargs):
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        assert any(
            e.source_node == "img_encode"
            and e.target_node == "latent_init"
            and e.source_port == C.PORT_LATENTS
            and e.target_port == C.PORT_INIT_LATENTS
            for e in g.get_edges()
        )

    def test_exposed_inputs_include_mask(self, sd15_kwargs):
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        spec = g.get_input_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_MASK_IMAGE in port_names
        assert C.PORT_INIT_IMAGE in port_names

    def test_mask_prep_wires_to_unet(self, sd15_kwargs):
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        edges = g.get_edges()
        assert any(
            e.source_node == "mask_prep"
            and e.target_node == "unet"
            and e.source_port == C.PORT_MASK_LATENTS
            for e in edges
        )
        assert any(
            e.source_node == "mask_prep"
            and e.target_node == "unet"
            and e.source_port == C.PORT_MASKED_IMAGE_LATENTS
            for e in edges
        )

    def test_four_channel_unet_adds_inpaint_blend_node(self, sd15_kwargs):
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        assert "inpaint_blend" in g.node_ids

    def test_nine_channel_unet_skips_inpaint_blend(self):
        kw = {
            "tokenizer": FakeTokenizer(),
            "text_encoder": FakeTextEncoder(),
            "unet": FakeUNet(channels=9),
            "vae": FakeVAE(),
            "scheduler": FakeScheduler(),
            "config": {"device": "cpu"},
        }
        g = build_sd15_inpaint_graph(**kw)
        assert "inpaint_blend" not in g.node_ids


class TestSD15InpaintReconfigureChannels:

    def test_nine_ch_graph_gains_blend_when_reconfigured_to_four_ch(self, sd15_kwargs):
        sd15_kwargs["unet"] = FakeUNet(channels=9)
        g = build_sd15_inpaint_graph(**sd15_kwargs)
        assert "inpaint_blend" not in g.node_ids
        reconfigure_sd15_inpaint_for_unet_in_channels(g, in_channels=4)
        assert "inpaint_blend" in g.node_ids
        assert g.get_node("latent_init")._config.get("inpaint_4ch_composite") is True
        edges = g.get_edges()
        assert any(
            e.source_node == "sched_step"
            and e.target_node == "inpaint_blend"
            and e.source_port == "next_latent"
            for e in edges
        )

    def test_four_ch_graph_loses_blend_when_reconfigured_to_nine_ch(self):
        kw = {
            "tokenizer": FakeTokenizer(),
            "text_encoder": FakeTextEncoder(),
            "unet": FakeUNet(channels=4),
            "vae": FakeVAE(),
            "scheduler": FakeScheduler(),
            "config": {"device": "cpu"},
        }
        g = build_sd15_inpaint_graph(**kw)
        assert "inpaint_blend" in g.node_ids
        reconfigure_sd15_inpaint_for_unet_in_channels(g, in_channels=9)
        assert "inpaint_blend" not in g.node_ids
        assert g.get_node("latent_init")._config.get("inpaint_4ch_composite") is False
        edges = g.get_edges()
        assert any(
            e.source_node == "sched_step"
            and e.target_node == "unet"
            and e.source_port == "next_latent"
            for e in edges
        )
