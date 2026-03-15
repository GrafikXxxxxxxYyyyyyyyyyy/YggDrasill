"""Tests for FLUX graph builders: text2img, img2img, inpaint, controlnet_text2img."""
from __future__ import annotations

import pytest

from yggdrasill.diffusion import contracts as C
from yggdrasill.diffusion.presets.flux import (
    build_flux_text2img_graph,
    build_flux_img2img_graph,
    build_flux_inpaint_graph,
    build_flux_controlnet_text2img_graph,
)

from tests.diffusion.conftest import (
    FakeCLIPEncoderWithPooler,
    FakeFlowMatchScheduler,
    FakeFluxControlNet,
    FakeFluxTransformer,
    FakeFluxVAE,
    FakeT5Encoder,
    FakeT5Tokenizer,
    FakeTokenizer,
)


@pytest.fixture
def flux_kwargs():
    return {
        "tokenizer": FakeTokenizer(),
        "tokenizer_2": FakeT5Tokenizer(),
        "text_encoder": FakeCLIPEncoderWithPooler(),
        "text_encoder_2": FakeT5Encoder(),
        "transformer": FakeFluxTransformer(),
        "vae": FakeFluxVAE(),
        "scheduler": FakeFlowMatchScheduler(),
    }


class TestFluxText2ImgGraph:

    def test_builds_successfully(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        assert g.graph_id == "flux_text2img"

    def test_has_required_nodes(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        expected = {"tokenizer", "prompt_enc", "sched_setup", "latent_init",
                    "transformer", "sched_step", "vae_decode"}
        assert expected.issubset(g.node_ids)

    def test_transformer_is_backbone(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        from yggdrasill.task_nodes.abstract import AbstractBackbone
        assert isinstance(g.get_node("transformer"), AbstractBackbone)

    def test_exposed_inputs(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        spec = g.get_input_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_PROMPT in port_names
        assert C.PORT_GUIDANCE in port_names

    def test_exposed_outputs(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        spec = g.get_output_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_DECODED_IMAGE in port_names

    def test_prompt_embeds_edge(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_port == C.PORT_PROMPT_EMBEDS
            and e.target_node == "transformer"
            for e in edges
        )
        assert found, "prompt_embeds -> transformer edge missing"

    def test_pooled_projections_edge(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_port == C.PORT_POOLED_PROMPT_EMBEDS
            and e.target_port == C.PORT_POOLED_PROJECTIONS
            for e in edges
        )
        assert found, "pooled_prompt_embeds -> pooled_projections edge missing"

    def test_txt_ids_edge(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_port == C.PORT_TXT_IDS
            and e.target_node == "transformer"
            for e in edges
        )
        assert found, "txt_ids -> transformer edge missing"

    def test_packed_latents_edge(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_port == C.PORT_PACKED_LATENTS
            and e.target_node == "transformer"
            for e in edges
        )
        assert found, "packed_latents -> transformer edge missing"

    def test_img_ids_edge(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_port == C.PORT_IMG_IDS
            and e.target_node == "transformer"
            for e in edges
        )
        assert found, "img_ids -> transformer edge missing"

    def test_config_roundtrip(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        cfg = g.to_config()
        assert cfg["graph_id"] == "flux_text2img"
        assert len(cfg["nodes"]) >= 6

    def test_no_negative_prompt(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        spec = g.get_input_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_NEGATIVE_PROMPT not in port_names

    def test_default_28_steps(self, flux_kwargs):
        g = build_flux_text2img_graph(**flux_kwargs)
        assert g.metadata.get("num_loop_steps") == 28


class TestFluxImg2ImgGraph:

    def test_builds(self, flux_kwargs):
        g = build_flux_img2img_graph(**flux_kwargs)
        assert "img_encode" in g.node_ids

    def test_latent_init_edge(self, flux_kwargs):
        g = build_flux_img2img_graph(**flux_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_node == "img_encode" and e.target_port == C.PORT_INIT_LATENTS
            for e in edges
        )
        assert found

    def test_exposed_init_image(self, flux_kwargs):
        g = build_flux_img2img_graph(**flux_kwargs)
        port_names = {s["port_name"] for s in g.get_input_spec()}
        assert C.PORT_INIT_IMAGE in port_names


class TestFluxInpaintGraph:

    def test_builds(self, flux_kwargs):
        g = build_flux_inpaint_graph(**flux_kwargs)
        assert "mask_prep" in g.node_ids

    def test_exposed_mask(self, flux_kwargs):
        g = build_flux_inpaint_graph(**flux_kwargs)
        port_names = {s["port_name"] for s in g.get_input_spec()}
        assert C.PORT_MASK_IMAGE in port_names

    def test_exposed_init_image(self, flux_kwargs):
        g = build_flux_inpaint_graph(**flux_kwargs)
        port_names = {s["port_name"] for s in g.get_input_spec()}
        assert C.PORT_INIT_IMAGE in port_names


class TestFluxControlNetText2ImgGraph:

    def test_builds(self, flux_kwargs):
        g = build_flux_controlnet_text2img_graph(
            **flux_kwargs, controlnet=FakeFluxControlNet(),
        )
        assert "controlnet" in g.node_ids

    def test_has_controlnet_edges(self, flux_kwargs):
        g = build_flux_controlnet_text2img_graph(
            **flux_kwargs, controlnet=FakeFluxControlNet(),
        )
        edges = g.get_edges()
        found_block = any(
            e.source_port == C.PORT_CONTROLNET_BLOCK_SAMPLES
            and e.target_node == "transformer"
            for e in edges
        )
        found_single = any(
            e.source_port == C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES
            and e.target_node == "transformer"
            for e in edges
        )
        assert found_block, "controlnet_block_samples -> transformer edge missing"
        assert found_single, "controlnet_single_block_samples -> transformer edge missing"

    def test_exposed_control_image(self, flux_kwargs):
        g = build_flux_controlnet_text2img_graph(
            **flux_kwargs, controlnet=FakeFluxControlNet(),
        )
        port_names = {s["port_name"] for s in g.get_input_spec()}
        assert C.PORT_CONTROL_IMAGE in port_names

    def test_prompt_embeds_to_controlnet(self, flux_kwargs):
        g = build_flux_controlnet_text2img_graph(
            **flux_kwargs, controlnet=FakeFluxControlNet(),
        )
        edges = g.get_edges()
        found = any(
            e.source_port == C.PORT_PROMPT_EMBEDS
            and e.target_node == "controlnet"
            for e in edges
        )
        assert found, "prompt_embeds -> controlnet edge missing"
