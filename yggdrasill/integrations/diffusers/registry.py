"""Register all Diffusers integration nodes in the global BlockRegistry."""
from __future__ import annotations

from typing import Optional

from yggdrasill.foundation.registry import BlockRegistry


def register_diffusion_nodes(registry: Optional[BlockRegistry] = None) -> None:
    """Register all SD1.5, SDXL, and adapter node types."""
    reg = registry or BlockRegistry.global_registry()

    from yggdrasill.integrations.diffusers.sd15.tokenizer import SD15TokenizerNode
    from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
    from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
    from yggdrasill.integrations.diffusers.sd15.scheduler import (
        SD15SchedulerSetupNode, SD15SchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode
    from yggdrasill.integrations.diffusers.sd15.inpaint_blend import (
        SD15InpaintFourChannelBlendNode,
    )
    from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEEncodeNode, SD15VAEDecodeNode
    from yggdrasill.integrations.diffusers.common.mask_prep import InpaintMaskPrepNode
    from yggdrasill.integrations.diffusers.sd15.safety import SD15SafetyNode

    from yggdrasill.integrations.diffusers.sdxl.tokenizer import SDXLTokenizerNode
    from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
    from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
    from yggdrasill.integrations.diffusers.sdxl.scheduler import (
        SDXLSchedulerSetupNode, SDXLSchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sdxl.latent_init import SDXLLatentInitNode
    from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEEncodeNode, SDXLVAEDecodeNode

    from yggdrasill.integrations.diffusers.adapters.lora import LoRALoaderNode
    from yggdrasill.integrations.diffusers.adapters.controlnet import ControlNetNode
    from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
    from yggdrasill.integrations.diffusers.adapters.textual_inversion import TextualInversionNode

    from yggdrasill.integrations.diffusers.flux.tokenizer import FluxTokenizerNode
    from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
    from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
    from yggdrasill.integrations.diffusers.flux.scheduler import (
        FluxSchedulerSetupNode, FluxSchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
    from yggdrasill.integrations.diffusers.flux.vae import FluxVAEEncodeNode, FluxVAEDecodeNode
    from yggdrasill.integrations.diffusers.flux.controlnet import FluxControlNetNode

    nodes = {
        "sd15/tokenizer": SD15TokenizerNode,
        "sd15/prompt_encoder": SD15PromptEncoderNode,
        "sd15/unet": SD15UNetNode,
        "sd15/scheduler_setup": SD15SchedulerSetupNode,
        "sd15/scheduler_step": SD15SchedulerStepNode,
        "sd15/latent_init": SD15LatentInitNode,
        "sd15/inpaint_four_channel_blend": SD15InpaintFourChannelBlendNode,
        "sd15/vae_encode": SD15VAEEncodeNode,
        "sd15/vae_decode": SD15VAEDecodeNode,
        "common/mask_prep": InpaintMaskPrepNode,
        "sd15/safety": SD15SafetyNode,
        "sdxl/tokenizer": SDXLTokenizerNode,
        "sdxl/prompt_encoder": SDXLPromptEncoderNode,
        "sdxl/added_conditioning": SDXLAddedConditioningNode,
        "sdxl/unet": SDXLUNetNode,
        "sdxl/scheduler_setup": SDXLSchedulerSetupNode,
        "sdxl/scheduler_step": SDXLSchedulerStepNode,
        "sdxl/latent_init": SDXLLatentInitNode,
        "sdxl/vae_encode": SDXLVAEEncodeNode,
        "sdxl/vae_decode": SDXLVAEDecodeNode,
        "adapter/lora_loader": LoRALoaderNode,
        "adapter/controlnet": ControlNetNode,
        "adapter/ip_adapter": IPAdapterNode,
        "adapter/textual_inversion": TextualInversionNode,
        "flux/tokenizer": FluxTokenizerNode,
        "flux/prompt_encoder": FluxPromptEncoderNode,
        "flux/transformer": FluxTransformerNode,
        "flux/scheduler_setup": FluxSchedulerSetupNode,
        "flux/scheduler_step": FluxSchedulerStepNode,
        "flux/latent_init": FluxLatentInitNode,
        "flux/vae_encode": FluxVAEEncodeNode,
        "flux/vae_decode": FluxVAEDecodeNode,
        "adapter/controlnet_flux": FluxControlNetNode,
    }

    for block_type, cls in nodes.items():
        reg.register(block_type, cls)
