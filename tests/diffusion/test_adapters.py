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
    def test_guess_mode_scales_residuals_by_depth(self):
        import torch
        from unittest.mock import MagicMock

        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode

        def fake_cn(latents, timestep, **kwargs):
            # Emulate diffusers ControlNetModel scaling behavior:
            # if guess_mode and not global_pool_conditions:
            #   scales = logspace(-1, 0, len(down)+1) * conditioning_scale
            #   down[i] *= scales[i]; mid *= scales[-1]
            b = latents.shape[0]
            down = [torch.ones(b, 1, 2, 2), torch.ones(b, 1, 2, 2)]
            mid = torch.ones(b, 1, 1, 1)
            cs = float(kwargs.get("conditioning_scale", 1.0))
            if kwargs.get("guess_mode", False):
                scales = torch.logspace(-1, 0, len(down) + 1, device=latents.device) * cs
                down = [d * float(s) for d, s in zip(down, scales)]
                mid = mid * float(scales[-1])
            else:
                down = [d * cs for d in down]
                mid = mid * cs
            return down, mid

        mock_cn = MagicMock(side_effect=fake_cn)
        mock_cn.config = type("C", (), {"global_pool_conditions": False})()
        mock_cn.parameters = lambda: iter([torch.zeros(1)])

        node = ControlNetNode(
            "cn",
            controlnet=mock_cn,
            config={
                "guidance_scale": 1.0,  # no CFG
                "height": 64,
                "width": 64,
                "guess_mode": True,
                "conditioning_scale": 2.0,
            },
        )

        out = node.forward({
            C.PORT_LATENTS: torch.zeros(1, 4, 8, 8),
            C.PORT_TIMESTEP: torch.tensor(999, dtype=torch.long),
            C.PORT_PROMPT_EMBEDS: torch.zeros(1, 77, 768),
            C.PORT_CONTROL_IMAGE: torch.zeros(1, 3, 64, 64),
        })

        down = out[C.PORT_DOWN_BLOCK_RESIDUALS]
        mid = out[C.PORT_MID_BLOCK_RESIDUAL]
        assert isinstance(down, tuple) and len(down) == 2
        # Diffusers profile uses logspace over (down blocks + mid):
        # scales = [0.1, sqrt(0.1), 1.0] for two down blocks; mid uses last=1.0.
        assert torch.allclose(down[0], torch.ones_like(down[0]) * 0.2)
        assert torch.allclose(down[1], torch.ones_like(down[1]) * (2.0 * (10 ** (-0.5))), rtol=1e-5, atol=1e-6)
        assert torch.allclose(mid, torch.ones_like(mid) * 2.0)

    @requires_torch
    def test_guess_mode_can_be_set_via_run_kwargs(self):
        import torch
        from unittest.mock import MagicMock

        from yggdrasill.engine.structure import Hypergraph
        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
        from yggdrasill.integrations.diffusers.run import run as run_diffusion

        captured: dict = {}

        def fake_cn(latents, timestep, **kwargs):
            captured["guess_mode"] = kwargs.get("guess_mode")
            down = [torch.zeros(latents.shape[0], 1, 2, 2)]
            mid = torch.zeros(latents.shape[0], 1, 1, 1)
            return down, mid

        mock_cn = MagicMock(side_effect=fake_cn)
        mock_cn.config = type("C", (), {"global_pool_conditions": False})()
        mock_cn.parameters = lambda: iter([torch.zeros(1)])

        g = Hypergraph()
        g.add_node("cn", ControlNetNode("cn", controlnet=mock_cn, config={"guidance_scale": 1.0, "height": 64, "width": 64}))
        g.expose_input("cn", C.PORT_CONTROL_IMAGE, f"cn:{C.PORT_CONTROL_IMAGE}")
        g.expose_input("cn", C.PORT_LATENTS, f"cn:{C.PORT_LATENTS}")
        g.expose_input("cn", C.PORT_TIMESTEP, f"cn:{C.PORT_TIMESTEP}")
        g.expose_input("cn", C.PORT_PROMPT_EMBEDS, f"cn:{C.PORT_PROMPT_EMBEDS}")
        g.expose_output("cn", C.PORT_MID_BLOCK_RESIDUAL, "mid")

        run_diffusion(
            g,
            inputs={
                f"cn:{C.PORT_LATENTS}": torch.zeros(1, 4, 8, 8),
                f"cn:{C.PORT_TIMESTEP}": torch.tensor(999, dtype=torch.long),
                f"cn:{C.PORT_PROMPT_EMBEDS}": torch.zeros(1, 77, 768),
                f"cn:{C.PORT_CONTROL_IMAGE}": torch.zeros(1, 3, 64, 64),
            },
            guess_mode=True,
            wrap_output=False,
        )
        assert captured.get("guess_mode") is True

    @requires_torch
    def test_control_guidance_window_matches_diffusers_keep(self):
        import torch
        from unittest.mock import MagicMock

        from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode

        captured: dict = {}

        def fake_cn(latents, timestep, **kwargs):
            captured["conditioning_scale"] = float(kwargs.get("conditioning_scale", -1))
            down = [torch.zeros(latents.shape[0], 1, 2, 2)]
            mid = torch.zeros(latents.shape[0], 1, 1, 1)
            return down, mid

        mock_cn = MagicMock(side_effect=fake_cn)
        mock_cn.config = type("C", (), {"global_pool_conditions": False})()
        mock_cn.parameters = lambda: iter([torch.zeros(1)])

        class _Sched:
            def __init__(self):
                self.timesteps = list(range(10))  # L=10
                self._yggdrasill_step_idx = 0

            def scale_model_input(self, x, t):
                return x

        sched = _Sched()

        node = ControlNetNode(
            "cn",
            controlnet=mock_cn,
            config={
                "guidance_scale": 1.0,
                "height": 64,
                "width": 64,
                "conditioning_scale": 2.0,
                "control_guidance_start": 0.2,
                "control_guidance_end": 0.6,
            },
        )

        def _call_at(i: int) -> float:
            sched._yggdrasill_step_idx = i
            node.forward({
                C.PORT_LATENTS: torch.zeros(1, 4, 8, 8),
                C.PORT_TIMESTEP: torch.tensor(999, dtype=torch.long),
                C.PORT_PROMPT_EMBEDS: torch.zeros(1, 77, 768),
                C.PORT_CONTROL_IMAGE: torch.zeros(1, 3, 64, 64),
                C.PORT_SCHEDULER_STATE: {"scheduler": sched},
            })
            return float(captured["conditioning_scale"])

        # Diffusers keep: keep=1 unless i/L < start OR (i+1)/L > end.
        # i=0 => 0/10 < 0.2 => keep=0
        assert _call_at(0) == 0.0
        # i=2 => 0.2 < 0.2 is False, (3/10)>0.6 False => keep=1
        assert _call_at(2) == 2.0
        # i=6 => (7/10)>0.6 True => keep=0
        assert _call_at(6) == 0.0

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

    def test_role_is_conjector(self):
        from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
        from yggdrasill.task_nodes.roles import Role

        assert IPAdapterNode("ip").role == Role.CONJECTOR

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


def test_unwrap_ip_adapter_loaded_image_nested_singleton() -> None:
    from yggdrasill.integrations.diffusers.adapters.ip_adapter import _unwrap_ip_adapter_loaded_image

    assert _unwrap_ip_adapter_loaded_image([["a"]]) == ["a"]
    assert _unwrap_ip_adapter_loaded_image([[["b"]]]) == ["b"]
    assert _unwrap_ip_adapter_loaded_image(["x", "y"]) == ["x", "y"]
    assert _unwrap_ip_adapter_loaded_image("z") == "z"
