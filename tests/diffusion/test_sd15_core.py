"""Tests for SD1.5 core nodes: prompt encoder, UNet, scheduler, VAE, safety."""
from __future__ import annotations

import pytest

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import PortDirection

from tests.diffusion.conftest import (
    FakeScheduler,
    FakeTensor,
    FakeTextEncoder,
    FakeUNet,
    FakeVAE,
    requires_torch,
)


class TestSD15PromptEncoder:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
        node = SD15PromptEncoderNode("enc", text_encoder=FakeTextEncoder())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_INPUT_IDS in in_names
        assert "prompt_embeds" in out_names
        assert "negative_prompt_embeds" in out_names
        assert in_names & out_names == set(), "No port name should appear as both IN and OUT"

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
        node = SD15PromptEncoderNode("enc", text_encoder=FakeTextEncoder())
        assert node.block_type == "sd15/prompt_encoder"

    @requires_torch
    def test_forward(self):
        from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
        node = SD15PromptEncoderNode("enc", text_encoder=FakeTextEncoder())
        out = node.forward({
            C.PORT_INPUT_IDS: FakeTensor((1, 77)),
            C.PORT_NEGATIVE_INPUT_IDS: FakeTensor((1, 77)),
        })
        assert "prompt_embeds" in out
        assert "negative_prompt_embeds" in out

    def test_to_device(self):
        from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
        enc = FakeTextEncoder()
        node = SD15PromptEncoderNode("enc", text_encoder=enc)
        result = node.to("cpu")
        assert result is node


class TestSD15UNet:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
        node = SD15UNetNode("unet", unet=FakeUNet())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_LATENTS in in_names
        assert C.PORT_TIMESTEP in in_names
        assert C.PORT_PROMPT_EMBEDS in in_names
        assert C.PORT_MASK_LATENTS in in_names
        assert C.PORT_MASKED_IMAGE_LATENTS in in_names
        assert C.PORT_NOISE_PRED in out_names

    @requires_torch
    def test_forward_no_cfg(self):
        from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
        node = SD15UNetNode("unet", unet=FakeUNet(), config={"guidance_scale": 1.0})
        out = node.forward({
            C.PORT_LATENTS: FakeTensor((1, 4, 64, 64)),
            C.PORT_TIMESTEP: FakeTensor((1,)),
            C.PORT_PROMPT_EMBEDS: FakeTensor((1, 77, 768)),
        })
        assert C.PORT_NOISE_PRED in out

    @requires_torch
    def test_forward_with_cfg(self):
        import torch
        from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
        node = SD15UNetNode("unet", unet=FakeUNet(), config={"guidance_scale": 7.5})
        # CFG path uses torch.cat on latents/embeds; FakeTensor is not a torch.Tensor.
        out = node.forward({
            C.PORT_LATENTS: torch.zeros(1, 4, 64, 64),
            C.PORT_TIMESTEP: torch.tensor(999, dtype=torch.long),
            C.PORT_PROMPT_EMBEDS: torch.zeros(1, 77, 768),
            C.PORT_NEGATIVE_PROMPT_EMBEDS: torch.zeros(1, 77, 768),
        })
        assert C.PORT_NOISE_PRED in out

    @requires_torch
    def test_forward_inpaint_nine_channel(self):
        import torch
        from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
        unet = FakeUNet(channels=9)
        node = SD15UNetNode("unet", unet=unet, config={"guidance_scale": 1.0})
        out = node.forward({
            C.PORT_LATENTS: torch.zeros(1, 4, 64, 64),
            C.PORT_TIMESTEP: torch.tensor(999, dtype=torch.long),
            C.PORT_PROMPT_EMBEDS: torch.zeros(1, 77, 768),
            C.PORT_MASK_LATENTS: torch.zeros(1, 1, 64, 64),
            C.PORT_MASKED_IMAGE_LATENTS: torch.zeros(1, 4, 64, 64),
        })
        assert C.PORT_NOISE_PRED in out

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
        assert SD15UNetNode("u", unet=FakeUNet()).block_type == "sd15/unet"


class TestSD15Scheduler:

    def test_setup_node_ports(self):
        from yggdrasill.integrations.diffusers.sd15.scheduler import SD15SchedulerSetupNode
        node = SD15SchedulerSetupNode("ss", scheduler=FakeScheduler())
        ports = node.declare_ports()
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_TIMESTEPS in out_names
        assert C.PORT_SCHEDULER_STATE in out_names

    def test_setup_forward(self):
        from yggdrasill.integrations.diffusers.sd15.scheduler import SD15SchedulerSetupNode
        sched = FakeScheduler()
        node = SD15SchedulerSetupNode("ss", scheduler=sched, config={"num_inference_steps": 20})
        out = node.forward({"input": None})
        assert "timesteps" in out
        assert "scheduler_state" in out
        state = out["scheduler_state"]
        assert "scheduler" in state
        assert "init_noise_sigma" in state

    def test_step_node_forward(self):
        pytest.importorskip("torch")
        from yggdrasill.integrations.diffusers.sd15.scheduler import SD15SchedulerStepNode
        sched = FakeScheduler()
        node = SD15SchedulerStepNode("step", scheduler=sched)
        out = node.forward({
            C.PORT_LATENTS: FakeTensor((1, 4, 64, 64)),
            C.PORT_TIMESTEP: FakeTensor((1,)),
            C.PORT_NOISE_PRED: FakeTensor((1, 4, 64, 64)),
        })
        assert "next_latent" in out
        assert "next_timestep" in out


class TestSD15VAE:

    def test_decode_ports(self):
        from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEDecodeNode
        node = SD15VAEDecodeNode("dec", vae=FakeVAE())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_LATENTS in in_names
        assert C.PORT_DECODED_IMAGE in out_names

    def test_encode_ports(self):
        from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEEncodeNode
        node = SD15VAEEncodeNode("enc", vae=FakeVAE())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_INIT_IMAGE in in_names
        assert C.PORT_LATENTS in out_names

    @requires_torch
    def test_decode_latent_passthrough(self):
        from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEDecodeNode
        node = SD15VAEDecodeNode("dec", vae=FakeVAE(), config={"output_type": "latent"})
        latents = FakeTensor((1, 4, 64, 64))
        out = node.forward({C.PORT_LATENTS: latents})
        assert out[C.PORT_DECODED_IMAGE] is latents


class TestSD15LatentInit:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode
        node = SD15LatentInitNode("li")
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_INIT_LATENTS in in_names
        assert C.PORT_SCHEDULER_STATE in in_names
        assert C.PORT_LATENTS in out_names
        assert in_names & out_names == set()

    @requires_torch
    def test_passthrough_existing(self):
        from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode
        node = SD15LatentInitNode("li")
        init = FakeTensor((1, 4, 64, 64))
        out = node.forward({C.PORT_INIT_LATENTS: init})
        assert C.PORT_LATENTS in out

    @requires_torch
    def test_img2img_strength_truncates_scheduler(self):
        import torch
        pytest.importorskip("diffusers")
        from diffusers import DDIMScheduler

        from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode

        sched = DDIMScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            clip_sample=False,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            set_alpha_to_one=False,
            steps_offset=1,
        )
        sched.set_timesteps(50, device="cpu")
        state = {
            "scheduler": sched,
            "init_noise_sigma": float(getattr(sched, "init_noise_sigma", 1.0)),
            "num_loop_steps": 50,
        }
        node = SD15LatentInitNode(
            "li",
            config={"strength": 0.8, "device": "cpu", "seed": 42},
        )
        enc = torch.randn(1, 4, 64, 64)
        out = node.forward({
            C.PORT_INIT_LATENTS: enc,
            C.PORT_SCHEDULER_STATE: state,
        })
        assert C.PORT_LATENTS in out
        assert len(sched.timesteps) == 40
        assert state["num_loop_steps"] == 40
        assert not torch.allclose(out[C.PORT_LATENTS], enc)


class TestSD15Safety:

    def test_disabled(self):
        from yggdrasill.integrations.diffusers.sd15.safety import SD15SafetyNode
        node = SD15SafetyNode("safety", config={"enabled": False})
        out = node.forward({C.PORT_DECODED_IMAGE: ["img1"]})
        assert out[C.PORT_OUTPUT_IMAGE] == ["img1"]
        assert out[C.PORT_NSFW_DETECTED] == [False]

    def test_no_checker(self):
        from yggdrasill.integrations.diffusers.sd15.safety import SD15SafetyNode
        node = SD15SafetyNode("safety")
        out = node.forward({C.PORT_DECODED_IMAGE: ["img1", "img2"]})
        assert len(out[C.PORT_NSFW_DETECTED]) == 2

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.sd15.safety import SD15SafetyNode
        assert SD15SafetyNode("s").block_type == "sd15/safety"


class TestSD15MaskPrep:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.common.mask_prep import InpaintMaskPrepNode
        node = InpaintMaskPrepNode("mp", vae=FakeVAE())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_INIT_IMAGE in in_names
        assert C.PORT_MASK_IMAGE in in_names
        assert C.PORT_MASK_LATENTS in out_names
        assert C.PORT_MASKED_IMAGE_LATENTS in out_names
