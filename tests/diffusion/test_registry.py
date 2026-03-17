"""Tests for diffusion node registry and factory API."""
from __future__ import annotations


from yggdrasill.foundation.registry import BlockRegistry


class TestDiffusionRegistry:

    def setup_method(self):
        self.registry = BlockRegistry()

    def test_register_all_nodes(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes(self.registry)

        expected_types = [
            "sd15/prompt_encoder",
            "sd15/unet",
            "sd15/scheduler_setup",
            "sd15/scheduler_step",
            "sd15/latent_init",
            "sd15/vae_encode",
            "sd15/vae_decode",
            "common/mask_prep",
            "sd15/safety",
            "sdxl/prompt_encoder",
            "sdxl/added_conditioning",
            "sdxl/unet",
            "sdxl/scheduler_setup",
            "sdxl/scheduler_step",
            "sdxl/latent_init",
            "sdxl/vae_encode",
            "sdxl/vae_decode",
            "adapter/lora_loader",
            "adapter/controlnet",
            "adapter/ip_adapter",
            "adapter/textual_inversion",
            "flux/prompt_encoder",
            "flux/transformer",
            "flux/scheduler_setup",
            "flux/scheduler_step",
            "flux/latent_init",
            "flux/vae_encode",
            "flux/vae_decode",
            "adapter/controlnet_flux",
        ]

        for bt in expected_types:
            assert bt in self.registry, f"Missing: {bt}"

    def test_build_sd15_latent_init(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes(self.registry)

        node = self.registry.build({
            "block_type": "sd15/latent_init",
            "node_id": "test_li",
        })
        assert node.node_id == "test_li"
        assert node.block_type == "sd15/latent_init"

    def test_build_sdxl_added_conditioning(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes(self.registry)

        node = self.registry.build({
            "block_type": "sdxl/added_conditioning",
            "node_id": "test_ac",
            "config": {"original_size": [1024, 1024]},
        })
        assert node.node_id == "test_ac"
        assert node.block_type == "sdxl/added_conditioning"

    def test_registered_types_count(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes(self.registry)
        assert len(self.registry.registered_types) >= 29

    def test_build_all_types_with_node_id(self):
        from yggdrasill.integrations.diffusers.registry import register_diffusion_nodes
        register_diffusion_nodes(self.registry)

        for bt in self.registry.registered_types:
            node = self.registry.build({"block_type": bt, "node_id": f"test_{bt.replace('/', '_')}"})
            assert hasattr(node, "node_id")
            assert hasattr(node, "block_type")
            assert node.block_type == bt


class TestDiffusionParity:
    """Verify structural parity between SD1.5/SDXL graphs and Diffusers semantics."""

    def test_sd15_text2img_node_roles(self):
        from yggdrasill.integrations.diffusers.presets.sd15 import build_sd15_text2img_graph
        from tests.diffusion.conftest import (
            FakeTokenizer, FakeTextEncoder, FakeUNet, FakeVAE, FakeScheduler,
        )
        g = build_sd15_text2img_graph(
            tokenizer=FakeTokenizer(), text_encoder=FakeTextEncoder(),
            unet=FakeUNet(), vae=FakeVAE(), scheduler=FakeScheduler(),
        )
        unet = g.get_node("unet")
        enc = g.get_node("prompt_enc")
        vae = g.get_node("vae_decode")

        from yggdrasill.task_nodes.abstract import AbstractBackbone, AbstractConjector, AbstractConverter
        assert isinstance(unet, AbstractBackbone)
        assert isinstance(enc, AbstractConjector)
        assert isinstance(vae, AbstractConverter)

    def test_sdxl_has_dual_encoder_path(self):
        from yggdrasill.integrations.diffusers.presets.sdxl import build_sdxl_text2img_graph
        from tests.diffusion.conftest import (
            FakeTokenizer, FakeTextEncoder, FakeTextEncoder2,
            FakeUNet, FakeVAE, FakeScheduler,
        )
        g = build_sdxl_text2img_graph(
            tokenizer=FakeTokenizer(), tokenizer_2=FakeTokenizer(),
            text_encoder=FakeTextEncoder(), text_encoder_2=FakeTextEncoder2(),
            unet=FakeUNet(), vae=FakeVAE(0.13025), scheduler=FakeScheduler(),
        )
        input_spec = g.get_input_spec()
        port_names = {s["port_name"] for s in input_spec}
        assert "prompt_2" in port_names
