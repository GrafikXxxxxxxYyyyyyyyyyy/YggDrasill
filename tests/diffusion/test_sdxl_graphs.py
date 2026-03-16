"""Tests for SDXL graph builders: text2img, img2img, inpaint, base+refiner."""
from __future__ import annotations

import pytest

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.presets.sdxl import (
    build_sdxl_text2img_graph,
    build_sdxl_img2img_graph,
    build_sdxl_inpaint_graph,
    build_sdxl_base_refiner_workflow,
)

from tests.diffusion.conftest import (
    FakeScheduler,
    FakeTextEncoder,
    FakeTextEncoder2,
    FakeTokenizer,
    FakeUNet,
    FakeVAE,
)


@pytest.fixture
def sdxl_kwargs():
    return {
        "tokenizer": FakeTokenizer(),
        "tokenizer_2": FakeTokenizer(),
        "text_encoder": FakeTextEncoder(),
        "text_encoder_2": FakeTextEncoder2(),
        "unet": FakeUNet(),
        "vae": FakeVAE(scaling_factor=0.13025),
        "scheduler": FakeScheduler(),
    }


class TestSDXLText2ImgGraph:

    def test_builds_successfully(self, sdxl_kwargs):
        g = build_sdxl_text2img_graph(**sdxl_kwargs)
        assert g.graph_id == "sdxl_text2img"

    def test_has_required_nodes(self, sdxl_kwargs):
        g = build_sdxl_text2img_graph(**sdxl_kwargs)
        expected = {"tokenizer", "prompt_enc", "added_cond", "sched_setup", "latent_init",
                    "unet", "sched_step", "vae_decode"}
        assert expected.issubset(g.node_ids)

    def test_has_added_conditioning(self, sdxl_kwargs):
        g = build_sdxl_text2img_graph(**sdxl_kwargs)
        node = g.get_node("added_cond")
        assert node is not None
        assert node.block_type == "sdxl/added_conditioning"

    def test_exposed_inputs(self, sdxl_kwargs):
        g = build_sdxl_text2img_graph(**sdxl_kwargs)
        spec = g.get_input_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_PROMPT in port_names
        assert C.PORT_NEGATIVE_PROMPT in port_names
        assert C.PORT_PROMPT_2 in port_names

    def test_exposed_outputs(self, sdxl_kwargs):
        g = build_sdxl_text2img_graph(**sdxl_kwargs)
        spec = g.get_output_spec()
        port_names = {s["port_name"] for s in spec}
        assert C.PORT_DECODED_IMAGE in port_names

    def test_pooled_embeds_edge(self, sdxl_kwargs):
        g = build_sdxl_text2img_graph(**sdxl_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_port == C.PORT_POOLED_PROMPT_EMBEDS
            and e.target_node == "added_cond"
            for e in edges
        )
        assert found, "pooled_prompt_embeds → added_cond edge missing"

    def test_add_time_ids_edge(self, sdxl_kwargs):
        g = build_sdxl_text2img_graph(**sdxl_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_port == C.PORT_ADD_TIME_IDS
            and e.target_node == "unet"
            for e in edges
        )
        assert found, "add_time_ids → unet edge missing"

    def test_config_roundtrip(self, sdxl_kwargs):
        g = build_sdxl_text2img_graph(**sdxl_kwargs)
        cfg = g.to_config()
        assert cfg["graph_id"] == "sdxl_text2img"
        assert len(cfg["nodes"]) >= 7


class TestSDXLImg2ImgGraph:

    def test_builds(self, sdxl_kwargs):
        g = build_sdxl_img2img_graph(**sdxl_kwargs)
        assert "img_encode" in g.node_ids

    def test_latent_init_edge(self, sdxl_kwargs):
        g = build_sdxl_img2img_graph(**sdxl_kwargs)
        edges = g.get_edges()
        found = any(
            e.source_node == "img_encode" and e.target_port == C.PORT_INIT_LATENTS
            for e in edges
        )
        assert found

    def test_exposed_init_image(self, sdxl_kwargs):
        g = build_sdxl_img2img_graph(**sdxl_kwargs)
        port_names = {s["port_name"] for s in g.get_input_spec()}
        assert C.PORT_INIT_IMAGE in port_names


class TestSDXLInpaintGraph:

    def test_builds(self, sdxl_kwargs):
        g = build_sdxl_inpaint_graph(**sdxl_kwargs)
        assert "mask_prep" in g.node_ids

    def test_exposed_mask(self, sdxl_kwargs):
        g = build_sdxl_inpaint_graph(**sdxl_kwargs)
        port_names = {s["port_name"] for s in g.get_input_spec()}
        assert C.PORT_MASK_IMAGE in port_names


class TestSDXLBaseRefinerWorkflow:

    def test_builds(self, sdxl_kwargs):
        base_comp = dict(sdxl_kwargs)
        refiner_comp = dict(sdxl_kwargs)
        w = build_sdxl_base_refiner_workflow(
            base_components=base_comp,
            refiner_components=refiner_comp,
            config={"denoising_end": 0.8},
        )
        assert "base" in w.node_ids
        assert "refiner" in w.node_ids

    def test_base_refiner_edge(self, sdxl_kwargs):
        base_comp = dict(sdxl_kwargs)
        refiner_comp = dict(sdxl_kwargs)
        w = build_sdxl_base_refiner_workflow(
            base_components=base_comp,
            refiner_components=refiner_comp,
        )
        edges = w.get_edges()
        assert len(edges) == 1
        e = edges[0]
        assert e.source_node == "base"
        assert e.target_node == "refiner"

    def test_exposed_prompt(self, sdxl_kwargs):
        base_comp = dict(sdxl_kwargs)
        refiner_comp = dict(sdxl_kwargs)
        w = build_sdxl_base_refiner_workflow(
            base_components=base_comp,
            refiner_components=refiner_comp,
        )
        port_names = {s["port_name"] for s in w.get_input_spec()}
        assert C.PORT_PROMPT in port_names

    def test_config_roundtrip(self, sdxl_kwargs):
        base_comp = dict(sdxl_kwargs)
        refiner_comp = dict(sdxl_kwargs)
        w = build_sdxl_base_refiner_workflow(
            base_components=base_comp,
            refiner_components=refiner_comp,
        )
        cfg = w.to_config()
        assert cfg["workflow_id"] == "sdxl_base_refiner"
        assert len(cfg["graphs"]) == 2
