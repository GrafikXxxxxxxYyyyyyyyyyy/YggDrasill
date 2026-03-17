"""Tests for FLUX core nodes: prompt encoder, transformer, scheduler, latent init, VAE, controlnet."""
from __future__ import annotations

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import PortDirection

from tests.diffusion.conftest import (
    FakeCLIPEncoderWithPooler,
    FakeFlowMatchScheduler,
    FakeFluxControlNet,
    FakeFluxTransformer,
    FakeFluxVAE,
    FakeT5Encoder,
    FakeT5Tokenizer,
    FakeTensor,
    FakeTokenizer,
    requires_torch,
)


class TestFluxPromptEncoder:

    def test_ports_no_duplicates(self):
        from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
        node = FluxPromptEncoderNode(
            "enc",
            text_encoder=FakeCLIPEncoderWithPooler(), text_encoder_2=FakeT5Encoder(),
        )
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert in_names & out_names == set()

    def test_output_ports(self):
        from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
        node = FluxPromptEncoderNode(
            "enc",
            text_encoder=FakeCLIPEncoderWithPooler(), text_encoder_2=FakeT5Encoder(),
        )
        ports = node.declare_ports()
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_PROMPT_EMBEDS in out_names
        assert C.PORT_POOLED_PROMPT_EMBEDS in out_names
        assert C.PORT_TXT_IDS in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
        node = FluxPromptEncoderNode(
            "enc",
            text_encoder=FakeCLIPEncoderWithPooler(), text_encoder_2=FakeT5Encoder(),
        )
        assert node.block_type == "flux/prompt_encoder"

    def test_no_negative_prompt_port(self):
        from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
        node = FluxPromptEncoderNode(
            "enc",
            text_encoder=FakeCLIPEncoderWithPooler(), text_encoder_2=FakeT5Encoder(),
        )
        ports = node.declare_ports()
        port_names = {p.name for p in ports}
        assert C.PORT_NEGATIVE_PROMPT not in port_names
        assert C.PORT_NEGATIVE_PROMPT_EMBEDS not in port_names


class TestFluxTransformer:

    def test_ports(self):
        from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
        node = FluxTransformerNode("tf", transformer=FakeFluxTransformer())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_PACKED_LATENTS in in_names
        assert C.PORT_TIMESTEP in in_names
        assert C.PORT_PROMPT_EMBEDS in in_names
        assert C.PORT_POOLED_PROJECTIONS in in_names
        assert C.PORT_IMG_IDS in in_names
        assert C.PORT_TXT_IDS in in_names
        assert C.PORT_GUIDANCE in in_names
        assert C.PORT_NOISE_PRED in out_names

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
        assert FluxTransformerNode("tf", transformer=FakeFluxTransformer()).block_type == "flux/transformer"

    def test_no_cfg_ports(self):
        from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
        node = FluxTransformerNode("tf", transformer=FakeFluxTransformer())
        ports = node.declare_ports()
        port_names = {p.name for p in ports}
        assert C.PORT_NEGATIVE_PROMPT_EMBEDS not in port_names
        assert C.PORT_ADD_TEXT_EMBEDS not in port_names
        assert C.PORT_ADD_TIME_IDS not in port_names

    @requires_torch
    def test_forward(self):
        from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
        node = FluxTransformerNode("tf", transformer=FakeFluxTransformer())
        out = node.forward({
            C.PORT_PACKED_LATENTS: FakeTensor((1, 4096, 64)),
            C.PORT_TIMESTEP: FakeTensor((1,), 500.0),
            C.PORT_PROMPT_EMBEDS: FakeTensor((1, 512, 4096)),
            C.PORT_POOLED_PROJECTIONS: FakeTensor((1, 768)),
            C.PORT_IMG_IDS: FakeTensor((4096, 3)),
            C.PORT_TXT_IDS: FakeTensor((512, 3)),
        })
        assert C.PORT_NOISE_PRED in out


class TestFluxScheduler:

    def test_setup_block_type(self):
        from yggdrasill.integrations.diffusers.flux.scheduler import FluxSchedulerSetupNode
        node = FluxSchedulerSetupNode("ss", scheduler=FakeFlowMatchScheduler())
        assert node.block_type == "flux/scheduler_setup"

    def test_setup_ports(self):
        from yggdrasill.integrations.diffusers.flux.scheduler import FluxSchedulerSetupNode
        node = FluxSchedulerSetupNode("ss", scheduler=FakeFlowMatchScheduler())
        ports = node.declare_ports()
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_TIMESTEPS in out_names
        assert C.PORT_SCHEDULER_STATE in out_names

    def test_setup_forward(self):
        from yggdrasill.integrations.diffusers.flux.scheduler import FluxSchedulerSetupNode
        sched = FakeFlowMatchScheduler()
        node = FluxSchedulerSetupNode("ss", scheduler=sched, config={
            "num_inference_steps": 28,
        })
        out = node.forward({"input": None})
        assert C.PORT_TIMESTEPS in out
        assert C.PORT_SCHEDULER_STATE in out
        assert len(out[C.PORT_TIMESTEPS]) == 28

    def test_step_block_type(self):
        from yggdrasill.integrations.diffusers.flux.scheduler import FluxSchedulerStepNode
        assert FluxSchedulerStepNode("s", scheduler=FakeFlowMatchScheduler()).block_type == "flux/scheduler_step"

    def test_step_ports(self):
        from yggdrasill.integrations.diffusers.flux.scheduler import FluxSchedulerStepNode
        node = FluxSchedulerStepNode("s", scheduler=FakeFlowMatchScheduler())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_PACKED_LATENTS in in_names
        assert C.PORT_NOISE_PRED in in_names
        assert "next_latent" in out_names

    def test_step_forward(self):
        from yggdrasill.integrations.diffusers.flux.scheduler import FluxSchedulerStepNode
        node = FluxSchedulerStepNode("s", scheduler=FakeFlowMatchScheduler())
        out = node.forward({
            C.PORT_PACKED_LATENTS: FakeTensor((1, 4096, 64)),
            C.PORT_TIMESTEP: FakeTensor((1,)),
            C.PORT_NOISE_PRED: FakeTensor((1, 4096, 64)),
        })
        assert "next_latent" in out


class TestFluxLatentInit:

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
        assert FluxLatentInitNode("li").block_type == "flux/latent_init"

    def test_default_1024(self):
        from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
        node = FluxLatentInitNode("li")
        assert node.config.get("height") == 1024
        assert node.config.get("width") == 1024

    def test_16_latent_channels(self):
        from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
        node = FluxLatentInitNode("li")
        assert node.config.get("num_latent_channels") == 16

    def test_ports(self):
        from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
        node = FluxLatentInitNode("li")
        ports = node.declare_ports()
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_PACKED_LATENTS in out_names
        assert C.PORT_IMG_IDS in out_names

    @requires_torch
    def test_forward_text2img(self):
        from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
        node = FluxLatentInitNode("li", config={
            "height": 1024, "width": 1024,
            "batch_size": 1, "device": "cpu", "dtype": "float32",
        })
        out = node.forward({})
        assert C.PORT_PACKED_LATENTS in out
        assert C.PORT_IMG_IDS in out
        packed = out[C.PORT_PACKED_LATENTS]
        assert packed.shape[0] == 1
        assert packed.shape[1] == (128 // 2) * (128 // 2)
        assert packed.shape[2] == 16 * 4


class TestFluxVAE:

    def test_encode_block_type(self):
        from yggdrasill.integrations.diffusers.flux.vae import FluxVAEEncodeNode
        assert FluxVAEEncodeNode("e", vae=FakeFluxVAE()).block_type == "flux/vae_encode"

    def test_decode_block_type(self):
        from yggdrasill.integrations.diffusers.flux.vae import FluxVAEDecodeNode
        assert FluxVAEDecodeNode("d", vae=FakeFluxVAE()).block_type == "flux/vae_decode"

    def test_decode_ports(self):
        from yggdrasill.integrations.diffusers.flux.vae import FluxVAEDecodeNode
        node = FluxVAEDecodeNode("d", vae=FakeFluxVAE())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_PACKED_LATENTS in in_names
        assert C.PORT_DECODED_IMAGE in out_names

    @requires_torch
    def test_decode_latent_passthrough(self):
        from yggdrasill.integrations.diffusers.flux.vae import FluxVAEDecodeNode
        node = FluxVAEDecodeNode("d", vae=FakeFluxVAE(), config={"output_type": "latent"})
        packed = FakeTensor((1, 4096, 64))
        out = node.forward({C.PORT_PACKED_LATENTS: packed})
        assert out[C.PORT_DECODED_IMAGE] is packed


class TestFluxControlNet:

    def test_block_type(self):
        from yggdrasill.integrations.diffusers.flux.controlnet import FluxControlNetNode
        assert FluxControlNetNode("cn", controlnet=FakeFluxControlNet()).block_type == "adapter/controlnet_flux"

    def test_role_inner_module(self):
        from yggdrasill.integrations.diffusers.flux.controlnet import FluxControlNetNode
        from yggdrasill.task_nodes.roles import Role
        node = FluxControlNetNode("cn", controlnet=FakeFluxControlNet())
        assert node.role == Role.INNER_MODULE

    def test_ports(self):
        from yggdrasill.integrations.diffusers.flux.controlnet import FluxControlNetNode
        node = FluxControlNetNode("cn", controlnet=FakeFluxControlNet())
        ports = node.declare_ports()
        in_names = {p.name for p in ports if p.direction == PortDirection.IN}
        out_names = {p.name for p in ports if p.direction == PortDirection.OUT}
        assert C.PORT_PACKED_LATENTS in in_names
        assert C.PORT_CONTROL_IMAGE in in_names
        assert C.PORT_CONTROLNET_BLOCK_SAMPLES in out_names
        assert C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES in out_names

    @requires_torch
    def test_forward(self):
        from yggdrasill.integrations.diffusers.flux.controlnet import FluxControlNetNode
        node = FluxControlNetNode("cn", controlnet=FakeFluxControlNet())
        out = node.forward({
            C.PORT_PACKED_LATENTS: FakeTensor((1, 4096, 64)),
            C.PORT_TIMESTEP: FakeTensor((1,), 500.0),
            C.PORT_PROMPT_EMBEDS: FakeTensor((1, 512, 4096)),
            C.PORT_POOLED_PROJECTIONS: FakeTensor((1, 768)),
            C.PORT_IMG_IDS: FakeTensor((4096, 3)),
            C.PORT_TXT_IDS: FakeTensor((512, 3)),
            C.PORT_CONTROL_IMAGE: FakeTensor((1, 3, 1024, 1024)),
        })
        assert C.PORT_CONTROLNET_BLOCK_SAMPLES in out
        assert C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES in out
