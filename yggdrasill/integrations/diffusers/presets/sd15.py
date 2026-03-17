"""SD1.5 graph builders: text2img, img2img, inpaint.

Each builder returns a fully-wired Hypergraph or Workflow that can be
executed with ``run()``. Components are injected from a loaded model store
or directly as constructor arguments.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph


def build_sd15_text2img_graph(
    *,
    tokenizer: Any = None,
    text_encoder: Any = None,
    unet: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    safety_checker: Any = None,
    feature_extractor: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical SD1.5 text-to-image hypergraph.

    Graph structure::

        prompt ──► Converter (tokenizer) ──► Conjector (text_encoder) ──► UNet ◄── LatentInit
                                      │              ▲
                                      ▼              │
                                SchedulerStep ──────┘
                                      │
                                      ▼
                                  VAEDecode ──► [Safety] ──► output
    """
    from yggdrasill.integrations.diffusers.sd15.tokenizer import SD15TokenizerNode
    from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
    from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
    from yggdrasill.integrations.diffusers.sd15.scheduler import (
        SD15SchedulerSetupNode,
        SD15SchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode
    from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEDecodeNode
    from yggdrasill.integrations.diffusers.sd15.safety import SD15SafetyNode

    cfg = config or {}

    h = Hypergraph(graph_id="sd15_text2img")

    tok_node = SD15TokenizerNode("tokenizer", tokenizer=tokenizer)
    enc = SD15PromptEncoderNode("prompt_enc", text_encoder=text_encoder,
                                 config={"clip_skip": cfg.get("clip_skip")})
    sched_setup = SD15SchedulerSetupNode("sched_setup", scheduler=scheduler,
                                          config={"num_inference_steps": cfg.get("num_inference_steps", 50),
                                                   "device": cfg.get("device", "cpu")})
    lat_init = SD15LatentInitNode("latent_init", config={
        "height": cfg.get("height", 512),
        "width": cfg.get("width", 512),
        "batch_size": cfg.get("batch_size", 1),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "float16"),
        "seed": cfg.get("seed"),
    })
    unet_node = SD15UNetNode("unet", unet=unet, config={
        "guidance_scale": cfg.get("guidance_scale", 7.5),
    })
    sched_step = SD15SchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = SD15VAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
    })

    h.add_node("tokenizer", tok_node)
    h.add_node("prompt_enc", enc)
    h.add_node("sched_setup", sched_setup)
    h.add_node("latent_init", lat_init)
    h.add_node("unet", unet_node)
    h.add_node("sched_step", sched_step)
    h.add_node("vae_decode", vae_dec)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "unet", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "unet", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_PROMPT_EMBEDS, "unet", C.PORT_NEGATIVE_PROMPT_EMBEDS))
    h.add_edge(Edge("latent_init", C.PORT_LATENTS, "unet", C.PORT_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_LATENTS, "sched_step", C.PORT_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_TIMESTEP, "unet", C.PORT_TIMESTEP))
    h.add_edge(Edge("latent_init", C.PORT_TIMESTEP, "sched_step", C.PORT_TIMESTEP))
    h.add_edge(Edge("unet", C.PORT_NOISE_PRED, "sched_step", C.PORT_NOISE_PRED))
    h.add_edge(Edge("sched_step", "next_latent", "unet", C.PORT_LATENTS))
    h.add_edge(Edge("sched_step", "next_latent", "sched_step", C.PORT_LATENTS))
    h.add_edge(Edge("sched_step", "next_timestep", "unet", C.PORT_TIMESTEP))
    h.add_edge(Edge("sched_step", "next_timestep", "sched_step", C.PORT_TIMESTEP))
    h.add_edge(Edge("sched_step", "next_latent", "vae_decode", C.PORT_LATENTS))

    h.expose_input("tokenizer", C.PORT_PROMPT, C.PORT_PROMPT)
    h.expose_input("tokenizer", C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {"num_loop_steps": cfg.get("num_inference_steps", 50)}

    if safety_checker is not None and cfg.get("enable_safety", True):
        safety = SD15SafetyNode("safety", safety_checker=safety_checker,
                                 feature_extractor=feature_extractor, config={"enabled": True})
        h.add_node("safety", safety)
        h.add_edge(Edge("vae_decode", C.PORT_DECODED_IMAGE, "safety", C.PORT_DECODED_IMAGE))
        h._exposed_outputs = [
            e for e in h._exposed_outputs if e.get("name") != C.PORT_OUTPUT_IMAGE
        ]
        h.expose_output("safety", C.PORT_OUTPUT_IMAGE, C.PORT_OUTPUT_IMAGE)

    return h


def build_sd15_img2img_graph(
    *,
    tokenizer: Any = None,
    text_encoder: Any = None,
    unet: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical SD1.5 image-to-image hypergraph."""
    from yggdrasill.integrations.diffusers.sd15.tokenizer import SD15TokenizerNode
    from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
    from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
    from yggdrasill.integrations.diffusers.sd15.scheduler import (
        SD15SchedulerSetupNode,
        SD15SchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEEncodeNode, SD15VAEDecodeNode
    from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode

    cfg = config or {}

    h = Hypergraph(graph_id="sd15_img2img")

    tok_node = SD15TokenizerNode("tokenizer", tokenizer=tokenizer)
    enc = SD15PromptEncoderNode("prompt_enc", text_encoder=text_encoder)
    img_enc = SD15VAEEncodeNode("img_encode", vae=vae, config={
        "height": cfg.get("height", 512),
        "width": cfg.get("width", 512),
        "device": cfg.get("device", "cpu"),
    })
    sched_setup = SD15SchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 50),
        "device": cfg.get("device", "cpu"),
    })
    lat_init = SD15LatentInitNode("latent_init", config={
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "float16"),
    })
    unet_node = SD15UNetNode("unet", unet=unet, config={
        "guidance_scale": cfg.get("guidance_scale", 7.5),
    })
    sched_step = SD15SchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = SD15VAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
    })

    h.add_node("tokenizer", tok_node)
    h.add_node("prompt_enc", enc)
    h.add_node("img_encode", img_enc)
    h.add_node("sched_setup", sched_setup)
    h.add_node("latent_init", lat_init)
    h.add_node("unet", unet_node)
    h.add_node("sched_step", sched_step)
    h.add_node("vae_decode", vae_dec)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS))
    h.add_edge(Edge("img_encode", C.PORT_LATENTS, "latent_init", C.PORT_INIT_LATENTS))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "unet", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "unet", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_PROMPT_EMBEDS, "unet", C.PORT_NEGATIVE_PROMPT_EMBEDS))
    h.add_edge(Edge("latent_init", C.PORT_LATENTS, "unet", C.PORT_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_LATENTS, "sched_step", C.PORT_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_TIMESTEP, "unet", C.PORT_TIMESTEP))
    h.add_edge(Edge("latent_init", C.PORT_TIMESTEP, "sched_step", C.PORT_TIMESTEP))
    h.add_edge(Edge("unet", C.PORT_NOISE_PRED, "sched_step", C.PORT_NOISE_PRED))
    h.add_edge(Edge("sched_step", "next_latent", "unet", C.PORT_LATENTS))
    h.add_edge(Edge("sched_step", "next_latent", "sched_step", C.PORT_LATENTS))
    h.add_edge(Edge("sched_step", "next_timestep", "unet", C.PORT_TIMESTEP))
    h.add_edge(Edge("sched_step", "next_timestep", "sched_step", C.PORT_TIMESTEP))
    h.add_edge(Edge("sched_step", "next_latent", "vae_decode", C.PORT_LATENTS))

    h.expose_input("tokenizer", C.PORT_PROMPT, C.PORT_PROMPT)
    h.expose_input("tokenizer", C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT)
    h.expose_input("img_encode", C.PORT_INIT_IMAGE, C.PORT_INIT_IMAGE)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {"num_loop_steps": cfg.get("num_inference_steps", 50)}

    return h


def build_sd15_inpaint_graph(
    *,
    tokenizer: Any = None,
    text_encoder: Any = None,
    unet: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical SD1.5 inpainting hypergraph."""
    from yggdrasill.integrations.diffusers.sd15.tokenizer import SD15TokenizerNode
    from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
    from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
    from yggdrasill.integrations.diffusers.sd15.scheduler import (
        SD15SchedulerSetupNode,
        SD15SchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEDecodeNode
    from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode
    from yggdrasill.integrations.diffusers.common.mask_prep import InpaintMaskPrepNode

    cfg = config or {}

    h = Hypergraph(graph_id="sd15_inpaint")

    tok_node = SD15TokenizerNode("tokenizer", tokenizer=tokenizer)
    enc = SD15PromptEncoderNode("prompt_enc", text_encoder=text_encoder)
    mask_prep = InpaintMaskPrepNode("mask_prep", vae=vae, config={
        "height": cfg.get("height", 512),
        "width": cfg.get("width", 512),
        "device": cfg.get("device", "cpu"),
    })
    sched_setup = SD15SchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 50),
        "device": cfg.get("device", "cpu"),
    })
    lat_init = SD15LatentInitNode("latent_init", config={
        "height": cfg.get("height", 512),
        "width": cfg.get("width", 512),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "float16"),
        "seed": cfg.get("seed"),
    })
    unet_node = SD15UNetNode("unet", unet=unet, config={
        "guidance_scale": cfg.get("guidance_scale", 7.5),
    })
    sched_step = SD15SchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = SD15VAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
    })

    h.add_node("tokenizer", tok_node)
    h.add_node("prompt_enc", enc)
    h.add_node("mask_prep", mask_prep)
    h.add_node("sched_setup", sched_setup)
    h.add_node("latent_init", lat_init)
    h.add_node("unet", unet_node)
    h.add_node("sched_step", sched_step)
    h.add_node("vae_decode", vae_dec)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "unet", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "unet", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_PROMPT_EMBEDS, "unet", C.PORT_NEGATIVE_PROMPT_EMBEDS))
    h.add_edge(Edge("latent_init", C.PORT_LATENTS, "unet", C.PORT_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_LATENTS, "sched_step", C.PORT_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_TIMESTEP, "unet", C.PORT_TIMESTEP))
    h.add_edge(Edge("latent_init", C.PORT_TIMESTEP, "sched_step", C.PORT_TIMESTEP))
    h.add_edge(Edge("unet", C.PORT_NOISE_PRED, "sched_step", C.PORT_NOISE_PRED))
    h.add_edge(Edge("sched_step", "next_latent", "unet", C.PORT_LATENTS))
    h.add_edge(Edge("sched_step", "next_latent", "sched_step", C.PORT_LATENTS))
    h.add_edge(Edge("sched_step", "next_timestep", "unet", C.PORT_TIMESTEP))
    h.add_edge(Edge("sched_step", "next_timestep", "sched_step", C.PORT_TIMESTEP))
    h.add_edge(Edge("sched_step", "next_latent", "vae_decode", C.PORT_LATENTS))

    h.expose_input("tokenizer", C.PORT_PROMPT, C.PORT_PROMPT)
    h.expose_input("tokenizer", C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT)
    h.expose_input("mask_prep", C.PORT_INIT_IMAGE, C.PORT_INIT_IMAGE)
    h.expose_input("mask_prep", C.PORT_MASK_IMAGE, C.PORT_MASK_IMAGE)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {"num_loop_steps": cfg.get("num_inference_steps", 50)}

    return h
