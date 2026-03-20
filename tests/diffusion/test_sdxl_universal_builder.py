"""Universal SDXL manual stack (DiffusionGraphBuilder): 4-ch UNet + added conditioning."""
from __future__ import annotations

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.run import _diffusion_universal_skip_nodes
from yggdrasill.integrations.diffusers.sdxl.universal import (
    discover_sdxl_manual_stack_roles,
    try_complete_sdxl_universal_diffusion,
)

from tests.diffusion.conftest import (
    FakeScheduler,
    FakeTextEncoder,
    FakeTextEncoder2,
    FakeTokenizer,
    FakeUNet,
    FakeVAE,
)


def _manual_sdxl_stack(*, unet_channels: int = 4) -> Hypergraph:
    from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
    from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
    from yggdrasill.integrations.diffusers.sdxl.scheduler import (
        SDXLSchedulerSetupNode,
        SDXLSchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sdxl.tokenizer import SDXLTokenizerNode
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
    from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEDecodeNode

    sched = FakeScheduler()
    h = Hypergraph(graph_id="manual_sdxl")
    h.add_node("Tok", SDXLTokenizerNode("Tok", tokenizer=FakeTokenizer(), tokenizer_2=FakeTokenizer()), auto_connect=False)
    h.add_node(
        "PE",
        SDXLPromptEncoderNode(
            "PE", text_encoder=FakeTextEncoder(), text_encoder_2=FakeTextEncoder2(),
        ),
        auto_connect=False,
    )
    h.add_node("AC", SDXLAddedConditioningNode("AC", config={}), auto_connect=False)
    h.add_node("Backbone", SDXLUNetNode("Backbone", unet=FakeUNet(channels=unet_channels)), auto_connect=False)
    h.add_node(
        "Sched_setup",
        SDXLSchedulerSetupNode("Sched_setup", scheduler=sched, config={"device": "cpu"}),
        auto_connect=False,
    )
    h.add_node("Sched_step", SDXLSchedulerStepNode("Sched_step", scheduler=sched), auto_connect=False)
    h.add_node("V", SDXLVAEDecodeNode("V", vae=FakeVAE()), auto_connect=False)
    return h


def _manual_sdxl_stack_no_added_cond(*, unet_channels: int = 4) -> Hypergraph:
    """Same as ``_manual_sdxl_stack`` but without an explicit added-conditioning node."""
    from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
    from yggdrasill.integrations.diffusers.sdxl.scheduler import (
        SDXLSchedulerSetupNode,
        SDXLSchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sdxl.tokenizer import SDXLTokenizerNode
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
    from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEDecodeNode

    sched = FakeScheduler()
    h = Hypergraph(graph_id="manual_sdxl_no_ac")
    h.add_node("Tok", SDXLTokenizerNode("Tok", tokenizer=FakeTokenizer(), tokenizer_2=FakeTokenizer()), auto_connect=False)
    h.add_node(
        "PE",
        SDXLPromptEncoderNode(
            "PE", text_encoder=FakeTextEncoder(), text_encoder_2=FakeTextEncoder2(),
        ),
        auto_connect=False,
    )
    h.add_node("Backbone", SDXLUNetNode("Backbone", unet=FakeUNet(channels=unet_channels)), auto_connect=False)
    h.add_node(
        "Sched_setup",
        SDXLSchedulerSetupNode("Sched_setup", scheduler=sched, config={"device": "cpu", "height": 512, "width": 768}),
        auto_connect=False,
    )
    h.add_node("Sched_step", SDXLSchedulerStepNode("Sched_step", scheduler=sched), auto_connect=False)
    h.add_node("V", SDXLVAEDecodeNode("V", vae=FakeVAE()), auto_connect=False)
    return h


def test_discover_roles_without_explicit_added_conditioning() -> None:
    g = _manual_sdxl_stack_no_added_cond()
    roles = discover_sdxl_manual_stack_roles(g)
    assert roles is not None
    assert "added_conditioning" not in roles
    assert roles["tokenizer"] == "Tok"


def test_sdxl_tokenizer_and_prompt_encoder_merge_from_components() -> None:
    from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
    from yggdrasill.integrations.diffusers.sdxl.tokenizer import SDXLTokenizerNode

    tok = SDXLTokenizerNode("T", tokenizer=FakeTokenizer(), tokenizer_2=None)
    tok.update_from_components({"tokenizer_2": FakeTokenizer()})
    assert tok._tokenizer_2 is not None

    pe = SDXLPromptEncoderNode("P", text_encoder=FakeTextEncoder(), text_encoder_2=None)
    pe.update_from_components({"text_encoder_2": FakeTextEncoder2()})
    assert pe._text_encoder_2 is not None


def test_try_complete_injects_added_conditioning_when_missing() -> None:
    g = _manual_sdxl_stack_no_added_cond()
    assert try_complete_sdxl_universal_diffusion(g) is True
    ac_id = g.metadata.get("sdxl_role_ids", {}).get("added_conditioning")
    assert ac_id in ("added_cond", "_sdxl_universal_added_cond")
    node = g.get_node(ac_id)
    assert node is not None
    assert getattr(node, "block_type", "") == "sdxl/added_conditioning"
    cfg = getattr(node, "_config", None) or {}
    assert cfg.get("original_size") == (512, 768)
    assert cfg.get("target_size") == (512, 768)


def test_try_complete_refuses_nine_channel_unet() -> None:
    g = _manual_sdxl_stack(unet_channels=9)
    assert try_complete_sdxl_universal_diffusion(g) is False


def test_try_complete_wires_four_channel_universal() -> None:
    g = _manual_sdxl_stack(unet_channels=4)
    assert try_complete_sdxl_universal_diffusion(g) is True
    assert g.metadata.get("sdxl_universal") is True
    assert "img_encode" in g.node_ids
    assert "mask_prep" in g.node_ids
    assert "latent_init" in g.node_ids
    assert "inpaint_blend" in g.node_ids
    assert isinstance(g.metadata.get("sdxl_role_ids"), dict)
    roles = g.metadata["sdxl_role_ids"]
    pe, u = roles["prompt_encoder"], roles["unet"]
    assert any(
        e.source_node == pe
        and e.target_node == u
        and e.source_port == C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS
        and e.target_port == C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS
        for e in g.get_edges()
    )


def test_sdxl_universal_skip_nodes_without_image() -> None:
    g = Hypergraph()
    g.metadata["sdxl_universal"] = True
    assert _diffusion_universal_skip_nodes(g, {}, {}) == {"img_encode", "mask_prep"}
