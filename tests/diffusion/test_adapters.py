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
        assert C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS in in_names
        assert C.PORT_NEGATIVE_ADD_TIME_IDS in in_names
        assert C.PORT_DOWN_BLOCK_RESIDUALS in out_names
        assert C.PORT_MID_BLOCK_RESIDUAL in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
        assert ControlNetNode("cn", controlnet=FakeControlNet()).block_type == "adapter/controlnet"

    def test_role_inner_module(self):
        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
        from yggdrasill.task_nodes.roles import Role
        node = ControlNetNode("cn", controlnet=FakeControlNet())
        assert node.role == Role.INNER_MODULE

    @requires_torch
    def test_forward(self):
        import torch
        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
        node = ControlNetNode("cn", controlnet=FakeControlNet())
        out = node.forward({
            C.PORT_LATENTS: FakeTensor((1, 4, 64, 64)),
            C.PORT_TIMESTEP: FakeTensor((1,)),
            C.PORT_PROMPT_EMBEDS: FakeTensor((1, 77, 768)),
            C.PORT_CONTROL_IMAGE: torch.zeros(1, 3, 512, 512),
        })
        assert C.PORT_DOWN_BLOCK_RESIDUALS in out
        assert C.PORT_MID_BLOCK_RESIDUAL in out
        assert isinstance(out[C.PORT_DOWN_BLOCK_RESIDUALS], tuple)

    @requires_torch
    def test_forward_skips_without_control_image(self):
        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
        node = ControlNetNode("cn", controlnet=FakeControlNet())
        out = node.forward({
            C.PORT_LATENTS: FakeTensor((1, 4, 64, 64)),
            C.PORT_TIMESTEP: FakeTensor((1,)),
            C.PORT_PROMPT_EMBEDS: FakeTensor((1, 77, 768)),
        })
        assert out == {}

    @requires_torch
    def test_sdxl_cfg_concatenates_added_cond_like_diffusers_pipeline(self):
        """Under CFG, SDXL ControlNet must get [neg, pos] stacked text_embeds / time_ids."""
        import torch
        from unittest.mock import MagicMock

        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode

        captured: dict = {}

        def fake_cn(latents, timestep, **kwargs):
            captured["added_cond_kwargs"] = kwargs.get("added_cond_kwargs")
            down = [torch.zeros(latents.shape[0], 1, 4, 4) for _ in range(2)]
            mid = torch.zeros(latents.shape[0], 1, 2, 2)
            return down, mid

        mock_cn = MagicMock(side_effect=fake_cn)
        mock_cn.config = type("C", (), {"global_pool_conditions": False})()
        mock_cn.parameters = lambda: iter([torch.zeros(1)])

        node = ControlNetNode(
            "cn",
            controlnet=mock_cn,
            config={"guidance_scale": 7.5, "height": 64, "width": 64},
        )
        node.forward({
            C.PORT_LATENTS: torch.zeros(1, 4, 8, 8),
            C.PORT_TIMESTEP: torch.tensor(999, dtype=torch.long),
            C.PORT_PROMPT_EMBEDS: torch.zeros(1, 77, 2048),
            C.PORT_NEGATIVE_PROMPT_EMBEDS: torch.zeros(1, 77, 2048),
            C.PORT_CONTROL_IMAGE: torch.zeros(1, 3, 64, 64),
            C.PORT_ADD_TEXT_EMBEDS: torch.zeros(1, 1280),
            C.PORT_ADD_TIME_IDS: torch.zeros(1, 6),
            C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS: torch.ones(1, 1280),
            C.PORT_NEGATIVE_ADD_TIME_IDS: torch.ones(1, 6),
        })
        ac = captured.get("added_cond_kwargs") or {}
        assert "text_embeds" in ac and "time_ids" in ac
        assert ac["text_embeds"].shape[0] == 2
        assert ac["time_ids"].shape[0] == 2
        assert torch.allclose(ac["text_embeds"][0], torch.ones(1, 1280))
        assert torch.allclose(ac["text_embeds"][1], torch.zeros(1, 1280))


class TestIPAdapterNode:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
        node = IPAdapterNode("ip")
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_IP_ADAPTER_IMAGE in in_names
        assert C.PORT_IP_ADAPTER_IMAGE_EMBEDS in in_names
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
        import torch
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
        node = IPAdapterNode("ip")
        embeds = torch.zeros(1, 1024)
        out = node.forward({C.PORT_IP_ADAPTER_IMAGE: embeds})
        assert C.PORT_IMAGE_EMBEDS in out
        assert out[C.PORT_IMAGE_EMBEDS] is embeds

    @requires_torch
    def test_forward_precomputed_embeds_port_skips_image_encoder(self):
        import torch
        from unittest.mock import MagicMock

        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode

        bad_enc = MagicMock()
        bad_enc.side_effect = AssertionError("encoder should not run when embeds port is set")

        node = IPAdapterNode(
            "ip",
            image_encoder=bad_enc,
            feature_extractor=MagicMock(),
        )
        emb = torch.ones(1, 512)
        out = node.forward({C.PORT_IP_ADAPTER_IMAGE_EMBEDS: emb})
        assert torch.equal(out[C.PORT_IMAGE_EMBEDS], emb)

    @requires_torch
    def test_forward_precomputed_list_multi_slot(self):
        import torch
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode

        node = IPAdapterNode("ip")
        a, b = torch.zeros(1, 128), torch.ones(1, 128)
        out = node.forward({C.PORT_IP_ADAPTER_IMAGE_EMBEDS: [a, b]})
        assert out[C.PORT_IMAGE_EMBEDS] == [a, b]

    @requires_torch
    def test_forward_inactive_returns_zero_embeds(self):
        import torch
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
        node = IPAdapterNode("ip")
        out = node.forward({})
        assert C.PORT_IMAGE_EMBEDS in out
        z = out[C.PORT_IMAGE_EMBEDS]
        assert isinstance(z, torch.Tensor)
        assert z.shape == (1, 1024)


class TestIPAdapterLoader:

    @requires_torch
    def test_load_ip_adapter_into_unet_calls_unet_method(self):
        from unittest.mock import MagicMock, patch
        from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import (
            load_ip_adapter_into_unet,
        )
        unet = MagicMock()
        unet._load_ip_adapter_weights = MagicMock()
        with patch("yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._load_ip_adapter_state_dict") as load_sd:
            load_sd.return_value = {"image_proj": {}, "ip_adapter": {}}
            load_ip_adapter_into_unet(unet, "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            unet._load_ip_adapter_weights.assert_called_once()
            call_args = unet._load_ip_adapter_weights.call_args
            assert len(call_args[0][0]) == 1
            assert call_args[0][0][0] == {"image_proj": {}, "ip_adapter": {}}


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
