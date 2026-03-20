"""Universal SD1.5 manual stack (DiffusionGraphBuilder): text2img/img2img/inpaint by run inputs."""
from __future__ import annotations

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers.run import _sd15_universal_skip_nodes
from yggdrasill.integrations.diffusers.sd15.universal import try_complete_sd15_universal_diffusion

from tests.diffusion.conftest import (
    FakeScheduler,
    FakeTextEncoder,
    FakeTokenizer,
    FakeUNet,
    FakeVAE,
)


def _manual_sd15_stack(*, unet_channels: int = 4) -> Hypergraph:
    from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
    from yggdrasill.integrations.diffusers.sd15.scheduler import (
        SD15SchedulerSetupNode,
        SD15SchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sd15.tokenizer import SD15TokenizerNode
    from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
    from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEDecodeNode

    sched = FakeScheduler()
    h = Hypergraph(graph_id="manual_sd15")
    h.add_node("Tok", SD15TokenizerNode("Tok", tokenizer=FakeTokenizer()), auto_connect=False)
    h.add_node(
        "PE",
        SD15PromptEncoderNode("PE", text_encoder=FakeTextEncoder()),
        auto_connect=False,
    )
    h.add_node("Backbone", SD15UNetNode("Backbone", unet=FakeUNet(channels=unet_channels)), auto_connect=False)
    h.add_node(
        "Sched_setup",
        SD15SchedulerSetupNode("Sched_setup", scheduler=sched, config={"device": "cpu"}),
        auto_connect=False,
    )
    h.add_node("Sched_step", SD15SchedulerStepNode("Sched_step", scheduler=sched), auto_connect=False)
    h.add_node("V", SD15VAEDecodeNode("V", vae=FakeVAE()), auto_connect=False)
    return h


def test_try_complete_refuses_nine_channel_unet() -> None:
    g = _manual_sd15_stack(unet_channels=9)
    assert try_complete_sd15_universal_diffusion(g) is False


def test_try_complete_wires_four_channel_universal() -> None:
    g = _manual_sd15_stack(unet_channels=4)
    assert try_complete_sd15_universal_diffusion(g) is True
    assert g.metadata.get("sd15_universal") is True
    assert "img_encode" in g.node_ids
    assert "mask_prep" in g.node_ids
    assert "latent_init" in g.node_ids
    assert "inpaint_blend" in g.node_ids
    assert isinstance(g.metadata.get("sd15_role_ids"), dict)


def test_universal_skip_nodes_without_image() -> None:
    g = Hypergraph()
    g.metadata["sd15_universal"] = True
    assert _sd15_universal_skip_nodes(g, {}, {}) == {"img_encode", "mask_prep"}


def test_universal_skip_nodes_with_image() -> None:
    g = Hypergraph()
    g.metadata["sd15_universal"] = True
    assert _sd15_universal_skip_nodes(g, {"image": "x"}, {}) == set()
