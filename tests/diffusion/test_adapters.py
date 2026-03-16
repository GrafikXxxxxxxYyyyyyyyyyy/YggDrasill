"""Tests for adapter nodes: ControlNet, IP-Adapter, LoRA, Textual Inversion."""
from __future__ import annotations


from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import PortDirection

from tests.diffusion.conftest import (
    FakeControlNet,
    FakeFeatureExtractor,
    FakeImageEncoder,
    FakeTensor,
    requires_torch,
)


class TestControlNetNode:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
        node = ControlNetNode("cn", controlnet=FakeControlNet())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_LATENTS in in_names
        assert C.PORT_CONTROL_IMAGE in in_names
        assert C.PORT_DOWN_BLOCK_RESIDUALS in out_names
        assert C.PORT_MID_BLOCK_RESIDUAL in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
        assert ControlNetNode("cn", controlnet=FakeControlNet()).block_type == "adapter/controlnet"

    @requires_torch
    def test_forward(self):
        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
        node = ControlNetNode("cn", controlnet=FakeControlNet())
        out = node.forward({
            C.PORT_LATENTS: FakeTensor((1, 4, 64, 64)),
            C.PORT_TIMESTEP: FakeTensor((1,)),
            C.PORT_PROMPT_EMBEDS: FakeTensor((1, 77, 768)),
            C.PORT_CONTROL_IMAGE: FakeTensor((1, 3, 512, 512)),
        })
        assert C.PORT_DOWN_BLOCK_RESIDUALS in out
        assert C.PORT_MID_BLOCK_RESIDUAL in out
        assert isinstance(out[C.PORT_DOWN_BLOCK_RESIDUALS], list)


class TestIPAdapterNode:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
        node = IPAdapterNode("ip")
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_IP_ADAPTER_IMAGE in in_names
        assert C.PORT_IMAGE_EMBEDS in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
        assert IPAdapterNode("ip").block_type == "adapter/ip_adapter"

    @requires_torch
    def test_forward_with_encoder(self):
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
        node = IPAdapterNode(
            "ip",
            image_encoder=FakeImageEncoder(),
            feature_extractor=FakeFeatureExtractor(),
        )
        out = node.forward({C.PORT_IP_ADAPTER_IMAGE: FakeTensor((1, 3, 224, 224))})
        assert C.PORT_IMAGE_EMBEDS in out

    @requires_torch
    def test_forward_tensor_passthrough(self):
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
        node = IPAdapterNode("ip")
        embeds = FakeTensor((1, 1024))
        out = node.forward({C.PORT_IP_ADAPTER_IMAGE: embeds})
        assert out[C.PORT_IMAGE_EMBEDS] is embeds


class TestLoRALoaderNode:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.adapters.lora import LoRALoaderNode
        node = LoRALoaderNode("lora")
        ports = node.declare_ports()
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert "result" in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.adapters.lora import LoRALoaderNode
        assert LoRALoaderNode("lora").block_type == "adapter/lora_loader"

    def test_forward_no_weights(self):
        from yggdrasill.integrations.diffusers.adapters.lora import LoRALoaderNode
        from unittest.mock import MagicMock
        pipe = MagicMock()
        node = LoRALoaderNode("lora", pipe=pipe, config={"lora_weights": []})
        out = node.forward({})
        assert out["result"]["loaded_loras"] == []


class TestTextualInversionNode:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.adapters.textual_inversion import TextualInversionNode
        node = TextualInversionNode("ti")
        ports = node.declare_ports()
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert "result" in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.adapters.textual_inversion import TextualInversionNode
        assert TextualInversionNode("ti").block_type == "adapter/textual_inversion"

    def test_forward_no_embeddings(self):
        from yggdrasill.integrations.diffusers.adapters.textual_inversion import TextualInversionNode
        from unittest.mock import MagicMock
        pipe = MagicMock()
        node = TextualInversionNode("ti", pipe=pipe, config={"embeddings": []})
        out = node.forward({})
        assert out["result"]["loaded_tokens"] == []
