import pytest

pytest.importorskip("torch")
import torch

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode


class _FixedMaskNode(AbstractBaseBlock, AbstractGraphNode):
    """Outputs a fixed tensor on ip_adapter_masks (no diffusers dependency)."""

    def __init__(self, node_id: str, mask: torch.Tensor) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self._mask = mask

    @property
    def block_type(self) -> str:
        return "test/fixed_ip_mask"

    def declare_ports(self):
        return [Port(C.PORT_IP_ADAPTER_MASKS, PortDirection.OUT, PortType.TENSOR)]

    def forward(self, inputs):
        return {C.PORT_IP_ADAPTER_MASKS: self._mask}


def _dummy_latents() -> torch.Tensor:
    return torch.zeros(1, 4, 8, 8)


def _dummy_prompt_embeds_sdxl() -> torch.Tensor:
    return torch.zeros(1, 77, 2048)


def _dummy_add_text_embeds() -> torch.Tensor:
    return torch.zeros(1, 1280)


def _dummy_add_time_ids() -> torch.Tensor:
    return torch.zeros(1, 6)


def _dummy_timestep() -> torch.Tensor:
    return torch.tensor([999], dtype=torch.long)


def test_sdxl_unet_multi_mask_sources_are_concat_ordered() -> None:
    captured: dict = {}

    class DummyUNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.config = type("C", (), {"in_channels": 4, "time_cond_proj_dim": None})()

        def forward(self, sample, timestep, **kwargs):
            captured.update(kwargs)
            return type("O", (), {"sample": torch.zeros_like(sample)})()

    # Distinguish sources to verify ordering.
    m1 = torch.ones(1, 1, 8, 8)
    m2 = torch.full((1, 1, 8, 8), 2.0)

    h = Hypergraph(name="sdxl_ip_mask_concat")
    # Ordering should be driven by graph.metadata when provided (mirrors image_embeds ordering).
    h.metadata["ip_adapter_weight_node_ids"] = ["m2", "m1"]
    h.add_node("m1", _FixedMaskNode("m1", m1))
    h.add_node("m2", _FixedMaskNode("m2", m2))
    h.add_node(
        "unet",
        SDXLUNetNode("unet", unet=DummyUNet(), config={"guidance_scale": 1.0}),
    )

    # Multi-edge wiring: m1 -> unet.ip_adapter_masks, m2 -> unet.ip_adapter_masks
    h.add_edge(Edge("m1", C.PORT_IP_ADAPTER_MASKS, "unet", C.PORT_IP_ADAPTER_MASKS))
    h.add_edge(Edge("m2", C.PORT_IP_ADAPTER_MASKS, "unet", C.PORT_IP_ADAPTER_MASKS))

    h.expose_input("unet", C.PORT_LATENTS, C.PORT_LATENTS)
    h.expose_input("unet", C.PORT_TIMESTEP, C.PORT_TIMESTEP)
    h.expose_input("unet", C.PORT_PROMPT_EMBEDS, C.PORT_PROMPT_EMBEDS)
    h.expose_input("unet", C.PORT_ADD_TEXT_EMBEDS, C.PORT_ADD_TEXT_EMBEDS)
    h.expose_input("unet", C.PORT_ADD_TIME_IDS, C.PORT_ADD_TIME_IDS)
    h.expose_output("unet", C.PORT_NOISE_PRED, "noise")

    out = h.run(
        {
            C.PORT_LATENTS: _dummy_latents(),
            C.PORT_TIMESTEP: _dummy_timestep(),
            C.PORT_PROMPT_EMBEDS: _dummy_prompt_embeds_sdxl(),
            C.PORT_ADD_TEXT_EMBEDS: _dummy_add_text_embeds(),
            C.PORT_ADD_TIME_IDS: _dummy_add_time_ids(),
        },
        validate_before=False,
    )
    assert "noise" in out

    assert "cross_attention_kwargs" in captured
    ip_masks = captured["cross_attention_kwargs"]["ip_adapter_masks"]
    assert isinstance(ip_masks, list)
    assert len(ip_masks) == 2
    assert torch.equal(ip_masks[0], m2)
    assert torch.equal(ip_masks[1], m1)


def test_sd15_unet_multi_mask_sources_are_concat_ordered() -> None:
    captured: dict = {}

    class DummyUNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.config = type("C", (), {"in_channels": 4})()

        def forward(self, sample, timestep, **kwargs):
            captured.update(kwargs)
            return type("O", (), {"sample": torch.zeros_like(sample)})()

    m1 = torch.ones(1, 1, 8, 8)
    m2 = torch.full((1, 1, 8, 8), 2.0)

    h = Hypergraph(name="sd15_ip_mask_concat")
    h.metadata["ip_adapter_weight_node_ids"] = ["m2", "m1"]
    h.add_node("m1", _FixedMaskNode("m1", m1))
    h.add_node("m2", _FixedMaskNode("m2", m2))
    h.add_node("unet", SD15UNetNode("unet", unet=DummyUNet(), config={"guidance_scale": 1.0}))

    h.add_edge(Edge("m1", C.PORT_IP_ADAPTER_MASKS, "unet", C.PORT_IP_ADAPTER_MASKS))
    h.add_edge(Edge("m2", C.PORT_IP_ADAPTER_MASKS, "unet", C.PORT_IP_ADAPTER_MASKS))

    # Minimal SD1.5 UNet inputs for forward without CFG.
    h.expose_input("unet", C.PORT_LATENTS, C.PORT_LATENTS)
    h.expose_input("unet", C.PORT_TIMESTEP, C.PORT_TIMESTEP)
    h.expose_input("unet", C.PORT_PROMPT_EMBEDS, C.PORT_PROMPT_EMBEDS)
    h.expose_output("unet", C.PORT_NOISE_PRED, "noise")

    h.run(
        {
            C.PORT_LATENTS: _dummy_latents(),
            C.PORT_TIMESTEP: _dummy_timestep(),
            C.PORT_PROMPT_EMBEDS: torch.zeros(1, 77, 768),
        },
        validate_before=False,
    )

    assert "cross_attention_kwargs" in captured
    ip_masks = captured["cross_attention_kwargs"]["ip_adapter_masks"]
    assert isinstance(ip_masks, list)
    assert len(ip_masks) == 2
    assert torch.equal(ip_masks[0], m2)
    assert torch.equal(ip_masks[1], m1)

