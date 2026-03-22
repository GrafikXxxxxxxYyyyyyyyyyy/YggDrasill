"""Tests for SDXL core nodes: dual prompt encoder, added conditioning, UNet."""
from __future__ import annotations

import pytest

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import PortDirection

from tests.diffusion.conftest import (
    FakeScheduler,
    FakeTensor,
    FakeTextEncoder,
    FakeTextEncoder2,
    FakeTokenizer,
    FakeUNet,
    FakeVAE,
    requires_torch,
)


class TestSDXLPromptEncoder:

    def test_ports_no_duplicates(self):
        from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
        node = SDXLPromptEncoderNode(
            "enc",
            text_encoder=FakeTextEncoder(), text_encoder_2=FakeTextEncoder2(),
        )
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert in_names & out_names == set(), "No port name should appear as both IN and OUT"

    def test_output_ports(self):
        from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
        node = SDXLPromptEncoderNode(
            "enc",
            text_encoder=FakeTextEncoder(), text_encoder_2=FakeTextEncoder2(),
        )
        ports = node.declare_ports()
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_PROMPT_EMBEDS in out_names
        assert C.PORT_NEGATIVE_PROMPT_EMBEDS in out_names
        assert C.PORT_POOLED_PROMPT_EMBEDS in out_names
        assert C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
        node = SDXLPromptEncoderNode(
            "enc",
            text_encoder=FakeTextEncoder(), text_encoder_2=FakeTextEncoder2(),
        )
        assert node.block_type == "sdxl/prompt_encoder"

    def test_dual_encoder_input_ports(self):
        from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
        node = SDXLPromptEncoderNode(
            "enc",
            text_encoder=FakeTextEncoder(), text_encoder_2=FakeTextEncoder2(),
        )
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        assert C.PORT_INPUT_IDS in in_names
        assert C.PORT_INPUT_IDS_2 in in_names


class TestSDXLAddedConditioning:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
        node = SDXLAddedConditioningNode("ac")
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_POOLED_PROMPT_EMBEDS in in_names
        assert C.PORT_ADD_TEXT_EMBEDS in out_names
        assert C.PORT_ADD_TIME_IDS in out_names
        assert C.PORT_NEGATIVE_ADD_TIME_IDS in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
        assert SDXLAddedConditioningNode("ac").block_type == "sdxl/added_conditioning"

    @requires_torch
    def test_forward_base_mode(self):
        from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
        node = SDXLAddedConditioningNode("ac", config={
            "original_size": (1024, 1024),
            "target_size": (1024, 1024),
        })
        out = node.forward({
            C.PORT_POOLED_PROMPT_EMBEDS: FakeTensor((1, 1280)),
        })
        assert C.PORT_ADD_TEXT_EMBEDS in out
        assert C.PORT_ADD_TIME_IDS in out
        assert C.PORT_NEGATIVE_ADD_TIME_IDS in out

    @requires_torch
    def test_forward_refiner_mode(self):
        from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
        node = SDXLAddedConditioningNode("ac", config={
            "requires_aesthetics_score": True,
            "aesthetic_score": 6.5,
            "negative_aesthetic_score": 2.0,
        })
        out = node.forward({
            C.PORT_POOLED_PROMPT_EMBEDS: FakeTensor((1, 1280)),
        })
        assert C.PORT_ADD_TIME_IDS in out


class TestSDXLUNet:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
        node = SDXLUNetNode("unet", unet=FakeUNet())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_LATENTS in in_names
        assert C.PORT_ADD_TEXT_EMBEDS in in_names
        assert C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS in in_names
        assert C.PORT_ADD_TIME_IDS in in_names
        assert C.PORT_MASK_LATENTS in in_names
        assert C.PORT_MASKED_IMAGE_LATENTS in in_names
        assert C.PORT_SCHEDULER_STATE in in_names
        assert C.PORT_NOISE_PRED in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
        assert SDXLUNetNode("u", unet=FakeUNet()).block_type == "sdxl/unet"

    @requires_torch
    def test_forward_no_cfg(self):
        from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
        node = SDXLUNetNode("unet", unet=FakeUNet(), config={"guidance_scale": 1.0})
        out = node.forward({
            C.PORT_LATENTS: FakeTensor((1, 4, 128, 128)),
            C.PORT_TIMESTEP: FakeTensor((1,)),
            C.PORT_PROMPT_EMBEDS: FakeTensor((1, 77, 2048)),
            C.PORT_ADD_TEXT_EMBEDS: FakeTensor((1, 1280)),
            C.PORT_ADD_TIME_IDS: FakeTensor((1, 6)),
        })
        assert C.PORT_NOISE_PRED in out


class TestSDXLScheduler:

    def test_setup_denoising_end(self):
        from yggdrasill.integrations.diffusers.sdxl.scheduler import SDXLSchedulerSetupNode
        sched = FakeScheduler()
        node = SDXLSchedulerSetupNode("ss", scheduler=sched, config={
            "num_inference_steps": 50,
            "denoising_end": 0.8,
        })
        out = node.forward({"input": None})
        assert "timesteps" in out

    def test_step_block_type(self):
        from yggdrasill.integrations.diffusers.sdxl.scheduler import SDXLSchedulerStepNode
        assert SDXLSchedulerStepNode("s", scheduler=FakeScheduler()).block_type == "sdxl/scheduler_step"


class TestSDXLVAE:

    def test_encode_block_type(self):
        from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEEncodeNode
        assert SDXLVAEEncodeNode("e", vae=FakeVAE()).block_type == "sdxl/vae_encode"

    def test_decode_block_type(self):
        from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEDecodeNode
        assert SDXLVAEDecodeNode("d", vae=FakeVAE()).block_type == "sdxl/vae_decode"

    @requires_torch
    def test_decode_latent_passthrough(self):
        from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEDecodeNode
        node = SDXLVAEDecodeNode("d", vae=FakeVAE(), config={"output_type": "latent"})
        latents = FakeTensor((1, 4, 128, 128))
        out = node.forward({C.PORT_LATENTS: latents})
        assert out[C.PORT_DECODED_IMAGE] is latents


class TestSDXLLatentInit:

    def test_default_1024(self):
        from yggdrasill.integrations.diffusers.sdxl.latent_init import SDXLLatentInitNode
        node = SDXLLatentInitNode("li")
        assert node.config.get("height", 1024) == 1024
        assert node.config.get("width", 1024) == 1024

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.sdxl.latent_init import SDXLLatentInitNode
        assert SDXLLatentInitNode("li").block_type == "sdxl/latent_init"

    @requires_torch
    def test_img2img_euler_add_noise_uses_batch_timesteps(self):
        """EulerDiscreteScheduler.add_noise requires 1-D timesteps (0-d raises IndexError)."""
        import torch
        pytest.importorskip("diffusers")
        from diffusers import EulerDiscreteScheduler

        from yggdrasill.integrations.diffusers.sdxl.latent_init import SDXLLatentInitNode

        sched = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type="epsilon",
            timestep_spacing="leading",
            steps_offset=1,
        )
        sched.set_timesteps(50, device="cpu")
        state = {
            "scheduler": sched,
            "init_noise_sigma": float(getattr(sched, "init_noise_sigma", 1.0)),
            "num_loop_steps": 50,
        }
        node = SDXLLatentInitNode(
            "li",
            config={"strength": 0.9, "device": "cpu", "seed": 42},
        )
        enc = torch.randn(1, 4, 128, 128)
        out = node.forward({
            C.PORT_INIT_LATENTS: enc,
            C.PORT_SCHEDULER_STATE: state,
        })
        assert C.PORT_LATENTS in out
        assert not torch.allclose(out[C.PORT_LATENTS], enc)
