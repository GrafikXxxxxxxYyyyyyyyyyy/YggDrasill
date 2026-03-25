"""ControlNet / IP-Adapter inputs must be exposed after builder._completed is True."""
from __future__ import annotations

from unittest.mock import MagicMock

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
from yggdrasill.integrations.diffusers.adapters.lora import LoRAInjectorNode
from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder
from yggdrasill.integrations.diffusers.presets.sd15 import build_sd15_text2img_graph

from tests.diffusion.conftest import (
    FakeControlNet,
    FakeScheduler,
    FakeTextEncoder,
    FakeTokenizer,
    FakeUNet,
    FakeVAE,
)


def test_controlnet_port_exposed_after_graph_was_completed():
    """Regression: add_component after first .graph must register control_image in input spec."""
    g = build_sd15_text2img_graph(
        tokenizer=FakeTokenizer(),
        text_encoder=FakeTextEncoder(),
        unet=FakeUNet(),
        vae=FakeVAE(),
        scheduler=FakeScheduler(),
        config={"device": "cpu"},
    )
    b = DiffusionGraphBuilder(g)
    _ = b.graph  # sets _completed; simulates prior cell that already ran the pipeline

    cn = ControlNetNode(
        "MyControlNet",
        controlnet=FakeControlNet(),
        config={"pretrained": "dummy/canny", "device": "cpu"},
    )
    b.add_node("MyControlNet", cn)

    keys = {e.get("name") for e in g.get_input_spec()}
    assert f"MyControlNet:{C.PORT_CONTROL_IMAGE}" in keys


def test_two_lora_loaders_expose_per_node_scale_keys():
    """Several LoRA injectors use node-scoped names (like ControlNet) so scales do not collide."""
    g = Hypergraph()
    pipe = MagicMock()
    g.add_node(
        "LoRA1",
        LoRAInjectorNode("LoRA1", pipe=pipe, config={"lora_weights": []}),
    )
    g.add_node(
        "LoRA2",
        LoRAInjectorNode("LoRA2", pipe=pipe, config={"lora_weights": []}),
    )
    b = DiffusionGraphBuilder(g)
    b.expose_default_io()
    lora_entries = [e for e in g.get_input_spec() if e.get("port_name") == C.PORT_LORA_SCALE]
    assert {e.get("name") for e in lora_entries} == {
        f"LoRA1:{C.PORT_LORA_SCALE}",
        f"LoRA2:{C.PORT_LORA_SCALE}",
    }


def test_single_lora_loader_keeps_lora_conditioning_scale_alias():
    g = Hypergraph()
    pipe = MagicMock()
    g.add_node("LoRA", LoRAInjectorNode("LoRA", pipe=pipe, config={"lora_weights": []}))
    DiffusionGraphBuilder(g).expose_default_io()
    lora_entries = [e for e in g.get_input_spec() if e.get("port_name") == C.PORT_LORA_SCALE]
    assert len(lora_entries) == 1
    assert lora_entries[0].get("name") == "lora_conditioning_scale"
