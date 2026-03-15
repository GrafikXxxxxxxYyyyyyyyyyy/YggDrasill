"""FLUX graph builders: text2img, img2img, inpaint, controlnet_text2img."""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph


def build_flux_text2img_graph(
    *,
    tokenizer: Any = None,
    tokenizer_2: Any = None,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    transformer: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical FLUX text-to-image hypergraph.

    Graph structure::

        prompt ──► Converter (tokenizer) ──► Conjector (text_encoder) ──► FluxTransformer ◄── FluxLatentInit
                                             │                     ▲
                                             ▼                     │
                                       FluxSchedulerStep ──────────┘
                                             │
                                             ▼
                                        FluxVAEDecode ──► output
    """
    from yggdrasill.integrations.diffusers.flux.tokenizer import FluxTokenizerNode
    from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
    from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
    from yggdrasill.integrations.diffusers.flux.scheduler import (
        FluxSchedulerSetupNode, FluxSchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
    from yggdrasill.integrations.diffusers.flux.vae import FluxVAEDecodeNode

    cfg = config or {}

    h = Hypergraph(graph_id="flux_text2img")

    tok_node = FluxTokenizerNode(
        "tokenizer",
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        config={"max_sequence_length": cfg.get("max_sequence_length", 512)},
    )
    prompt_enc = FluxPromptEncoderNode(
        "prompt_enc",
        text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        config={"max_sequence_length": cfg.get("max_sequence_length", 512)},
    )
    sched_setup = FluxSchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 28),
        "device": cfg.get("device", "cpu"),
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "use_dynamic_shifting": cfg.get("use_dynamic_shifting", False),
        "mu": cfg.get("mu"),
    })
    lat_init = FluxLatentInitNode("latent_init", config={
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "batch_size": cfg.get("batch_size", 1),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "bfloat16"),
        "seed": cfg.get("seed"),
    })
    transformer_node = FluxTransformerNode("transformer", transformer=transformer, config={
        "guidance_scale": cfg.get("guidance_scale", 3.5),
        "joint_attention_kwargs": cfg.get("joint_attention_kwargs"),
    })
    sched_step = FluxSchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = FluxVAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
    })

    for nid, node in [
        ("tokenizer", tok_node), ("prompt_enc", prompt_enc), ("sched_setup", sched_setup),
        ("latent_init", lat_init), ("transformer", transformer_node),
        ("sched_step", sched_step), ("vae_decode", vae_dec),
    ]:
        h.add_node(nid, node)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS_2, "prompt_enc", C.PORT_INPUT_IDS_2))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "transformer", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_POOLED_PROMPT_EMBEDS, "transformer", C.PORT_POOLED_PROJECTIONS))
    h.add_edge(Edge("prompt_enc", C.PORT_TXT_IDS, "transformer", C.PORT_TXT_IDS))
    h.add_edge(Edge("latent_init", C.PORT_PACKED_LATENTS, "transformer", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_IMG_IDS, "transformer", C.PORT_IMG_IDS))
    h.add_edge(Edge("transformer", C.PORT_NOISE_PRED, "sched_step", C.PORT_NOISE_PRED))
    h.add_edge(Edge("sched_step", "next_latent", "transformer", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("sched_step", "next_latent", "vae_decode", C.PORT_PACKED_LATENTS))

    h.expose_input("tokenizer", C.PORT_PROMPT, C.PORT_PROMPT)
    h.expose_input("tokenizer", C.PORT_PROMPT_2, C.PORT_PROMPT_2)
    h.expose_input("transformer", C.PORT_GUIDANCE, C.PORT_GUIDANCE)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {"num_loop_steps": cfg.get("num_inference_steps", 28)}

    return h


def build_flux_img2img_graph(
    *,
    tokenizer: Any = None,
    tokenizer_2: Any = None,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    transformer: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical FLUX image-to-image hypergraph."""
    from yggdrasill.integrations.diffusers.flux.tokenizer import FluxTokenizerNode
    from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
    from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
    from yggdrasill.integrations.diffusers.flux.scheduler import (
        FluxSchedulerSetupNode, FluxSchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
    from yggdrasill.integrations.diffusers.flux.vae import FluxVAEEncodeNode, FluxVAEDecodeNode

    cfg = config or {}
    h = Hypergraph(graph_id="flux_img2img")

    tok_node = FluxTokenizerNode(
        "tokenizer", tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        config={"max_sequence_length": cfg.get("max_sequence_length", 512)},
    )
    prompt_enc = FluxPromptEncoderNode(
        "prompt_enc", text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        config={"max_sequence_length": cfg.get("max_sequence_length", 512)},
    )
    img_enc = FluxVAEEncodeNode("img_encode", vae=vae, config={
        "height": cfg.get("height", 1024), "width": cfg.get("width", 1024),
        "device": cfg.get("device", "cpu"),
    })
    sched_setup = FluxSchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 28),
        "device": cfg.get("device", "cpu"),
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
    })
    lat_init = FluxLatentInitNode("latent_init", config={
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "bfloat16"),
        "strength": cfg.get("strength", 0.6),
    })
    transformer_node = FluxTransformerNode("transformer", transformer=transformer, config={
        "guidance_scale": cfg.get("guidance_scale", 7.0),
    })
    sched_step = FluxSchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = FluxVAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
    })

    for nid, node in [
        ("tokenizer", tok_node), ("prompt_enc", prompt_enc), ("img_encode", img_enc),
        ("sched_setup", sched_setup), ("latent_init", lat_init),
        ("transformer", transformer_node), ("sched_step", sched_step),
        ("vae_decode", vae_dec),
    ]:
        h.add_node(nid, node)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS_2, "prompt_enc", C.PORT_INPUT_IDS_2))
    h.add_edge(Edge("img_encode", C.PORT_LATENTS, "latent_init", C.PORT_INIT_LATENTS))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "transformer", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_POOLED_PROMPT_EMBEDS, "transformer", C.PORT_POOLED_PROJECTIONS))
    h.add_edge(Edge("prompt_enc", C.PORT_TXT_IDS, "transformer", C.PORT_TXT_IDS))
    h.add_edge(Edge("latent_init", C.PORT_PACKED_LATENTS, "transformer", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_IMG_IDS, "transformer", C.PORT_IMG_IDS))
    h.add_edge(Edge("transformer", C.PORT_NOISE_PRED, "sched_step", C.PORT_NOISE_PRED))
    h.add_edge(Edge("sched_step", "next_latent", "transformer", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("sched_step", "next_latent", "vae_decode", C.PORT_PACKED_LATENTS))

    h.expose_input("tokenizer", C.PORT_PROMPT, C.PORT_PROMPT)
    h.expose_input("img_encode", C.PORT_INIT_IMAGE, C.PORT_INIT_IMAGE)
    h.expose_input("transformer", C.PORT_GUIDANCE, C.PORT_GUIDANCE)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {"num_loop_steps": cfg.get("num_inference_steps", 28)}
    return h


def build_flux_inpaint_graph(
    *,
    tokenizer: Any = None,
    tokenizer_2: Any = None,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    transformer: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical FLUX inpainting hypergraph."""
    from yggdrasill.integrations.diffusers.flux.tokenizer import FluxTokenizerNode
    from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
    from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
    from yggdrasill.integrations.diffusers.flux.scheduler import (
        FluxSchedulerSetupNode, FluxSchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
    from yggdrasill.integrations.diffusers.flux.vae import FluxVAEDecodeNode
    from yggdrasill.integrations.diffusers.sd15.mask_prep import SD15MaskPrepNode

    cfg = config or {}
    h = Hypergraph(graph_id="flux_inpaint")

    tok_node = FluxTokenizerNode(
        "tokenizer", tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        config={"max_sequence_length": cfg.get("max_sequence_length", 512)},
    )
    prompt_enc = FluxPromptEncoderNode(
        "prompt_enc", text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        config={"max_sequence_length": cfg.get("max_sequence_length", 512)},
    )
    mask_prep = SD15MaskPrepNode("mask_prep", vae=vae, config={
        "height": cfg.get("height", 1024), "width": cfg.get("width", 1024),
        "device": cfg.get("device", "cpu"),
    })
    sched_setup = FluxSchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 28),
        "device": cfg.get("device", "cpu"),
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
    })
    lat_init = FluxLatentInitNode("latent_init", config={
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "bfloat16"),
        "seed": cfg.get("seed"),
        "strength": cfg.get("strength", 0.6),
    })
    transformer_node = FluxTransformerNode("transformer", transformer=transformer, config={
        "guidance_scale": cfg.get("guidance_scale", 3.5),
    })
    sched_step = FluxSchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = FluxVAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
    })

    for nid, node in [
        ("tokenizer", tok_node), ("prompt_enc", prompt_enc), ("mask_prep", mask_prep),
        ("sched_setup", sched_setup), ("latent_init", lat_init),
        ("transformer", transformer_node), ("sched_step", sched_step),
        ("vae_decode", vae_dec),
    ]:
        h.add_node(nid, node)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS_2, "prompt_enc", C.PORT_INPUT_IDS_2))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "transformer", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_POOLED_PROMPT_EMBEDS, "transformer", C.PORT_POOLED_PROJECTIONS))
    h.add_edge(Edge("prompt_enc", C.PORT_TXT_IDS, "transformer", C.PORT_TXT_IDS))
    h.add_edge(Edge("latent_init", C.PORT_PACKED_LATENTS, "transformer", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_IMG_IDS, "transformer", C.PORT_IMG_IDS))
    h.add_edge(Edge("transformer", C.PORT_NOISE_PRED, "sched_step", C.PORT_NOISE_PRED))
    h.add_edge(Edge("sched_step", "next_latent", "transformer", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("sched_step", "next_latent", "vae_decode", C.PORT_PACKED_LATENTS))

    h.expose_input("tokenizer", C.PORT_PROMPT, C.PORT_PROMPT)
    h.expose_input("mask_prep", C.PORT_INIT_IMAGE, C.PORT_INIT_IMAGE)
    h.expose_input("mask_prep", C.PORT_MASK_IMAGE, C.PORT_MASK_IMAGE)
    h.expose_input("transformer", C.PORT_GUIDANCE, C.PORT_GUIDANCE)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {"num_loop_steps": cfg.get("num_inference_steps", 28)}
    return h


def build_flux_controlnet_text2img_graph(
    *,
    tokenizer: Any = None,
    tokenizer_2: Any = None,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    transformer: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    controlnet: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a FLUX text-to-image hypergraph with ControlNet conditioning."""
    from yggdrasill.integrations.diffusers.flux.tokenizer import FluxTokenizerNode
    from yggdrasill.integrations.diffusers.flux.prompt_encoder import FluxPromptEncoderNode
    from yggdrasill.integrations.diffusers.flux.transformer import FluxTransformerNode
    from yggdrasill.integrations.diffusers.flux.scheduler import (
        FluxSchedulerSetupNode, FluxSchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.flux.latent_init import FluxLatentInitNode
    from yggdrasill.integrations.diffusers.flux.vae import FluxVAEDecodeNode
    from yggdrasill.integrations.diffusers.flux.controlnet import FluxControlNetNode

    cfg = config or {}

    h = Hypergraph(graph_id="flux_controlnet_text2img")

    tok_node = FluxTokenizerNode(
        "tokenizer", tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        config={"max_sequence_length": cfg.get("max_sequence_length", 512)},
    )
    prompt_enc = FluxPromptEncoderNode(
        "prompt_enc", text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        config={"max_sequence_length": cfg.get("max_sequence_length", 512)},
    )
    sched_setup = FluxSchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 28),
        "device": cfg.get("device", "cpu"),
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
    })
    lat_init = FluxLatentInitNode("latent_init", config={
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "batch_size": cfg.get("batch_size", 1),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "bfloat16"),
        "seed": cfg.get("seed"),
    })
    cn_node = FluxControlNetNode("controlnet", controlnet=controlnet, config={
        "controlnet_conditioning_scale": cfg.get("controlnet_conditioning_scale", 1.0),
    })
    transformer_node = FluxTransformerNode("transformer", transformer=transformer, config={
        "guidance_scale": cfg.get("guidance_scale", 3.5),
    })
    sched_step = FluxSchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = FluxVAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
    })

    for nid, node in [
        ("tokenizer", tok_node), ("prompt_enc", prompt_enc), ("sched_setup", sched_setup),
        ("latent_init", lat_init), ("controlnet", cn_node),
        ("transformer", transformer_node), ("sched_step", sched_step),
        ("vae_decode", vae_dec),
    ]:
        h.add_node(nid, node)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS_2, "prompt_enc", C.PORT_INPUT_IDS_2))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "transformer", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_POOLED_PROMPT_EMBEDS, "transformer", C.PORT_POOLED_PROJECTIONS))
    h.add_edge(Edge("prompt_enc", C.PORT_TXT_IDS, "transformer", C.PORT_TXT_IDS))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "controlnet", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_POOLED_PROMPT_EMBEDS, "controlnet", C.PORT_POOLED_PROJECTIONS))
    h.add_edge(Edge("prompt_enc", C.PORT_TXT_IDS, "controlnet", C.PORT_TXT_IDS))

    h.add_edge(Edge("latent_init", C.PORT_PACKED_LATENTS, "transformer", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_IMG_IDS, "transformer", C.PORT_IMG_IDS))
    h.add_edge(Edge("latent_init", C.PORT_PACKED_LATENTS, "controlnet", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_IMG_IDS, "controlnet", C.PORT_IMG_IDS))

    h.add_edge(Edge("controlnet", C.PORT_CONTROLNET_BLOCK_SAMPLES, "transformer", C.PORT_CONTROLNET_BLOCK_SAMPLES))
    h.add_edge(Edge("controlnet", C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES, "transformer", C.PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES))

    h.add_edge(Edge("transformer", C.PORT_NOISE_PRED, "sched_step", C.PORT_NOISE_PRED))
    h.add_edge(Edge("sched_step", "next_latent", "transformer", C.PORT_PACKED_LATENTS))
    h.add_edge(Edge("sched_step", "next_latent", "vae_decode", C.PORT_PACKED_LATENTS))

    h.expose_input("tokenizer", C.PORT_PROMPT, C.PORT_PROMPT)
    h.expose_input("controlnet", C.PORT_CONTROL_IMAGE, C.PORT_CONTROL_IMAGE)
    h.expose_input("transformer", C.PORT_GUIDANCE, C.PORT_GUIDANCE)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {"num_loop_steps": cfg.get("num_inference_steps", 28)}

    return h
