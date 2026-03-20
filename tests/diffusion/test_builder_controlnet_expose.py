"""ControlNet / IP-Adapter inputs must be exposed after builder._completed is True."""
from __future__ import annotations

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
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
