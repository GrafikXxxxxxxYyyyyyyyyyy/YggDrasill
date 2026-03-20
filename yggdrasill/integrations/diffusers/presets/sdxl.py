"""SDXL graph builders: text2img, img2img, inpaint, base+refiner workflow."""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.workflow.workflow import Workflow


def build_sdxl_text2img_graph(
    *,
    tokenizer: Any = None,
    tokenizer_2: Any = None,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    unet: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical SDXL text-to-image hypergraph."""
    from yggdrasill.integrations.diffusers.sdxl.tokenizer import SDXLTokenizerNode
    from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
    from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
    from yggdrasill.integrations.diffusers.sdxl.scheduler import SDXLSchedulerSetupNode, SDXLSchedulerStepNode
    from yggdrasill.integrations.diffusers.sdxl.latent_init import SDXLLatentInitNode
    from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEDecodeNode

    cfg = config or {}

    h = Hypergraph(graph_id="sdxl_text2img")

    tok_node = SDXLTokenizerNode(
        "tokenizer",
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
    )
    prompt_enc = SDXLPromptEncoderNode(
        "prompt_enc",
        text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        config={"clip_skip": cfg.get("clip_skip")},
    )
    added_cond = SDXLAddedConditioningNode("added_cond", config={
        "original_size": cfg.get("original_size", (1024, 1024)),
        "target_size": cfg.get("target_size", (1024, 1024)),
        "crops_coords_top_left": cfg.get("crops_coords_top_left", (0, 0)),
        "requires_aesthetics_score": cfg.get("requires_aesthetics_score", False),
    })
    sched_setup = SDXLSchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 50),
        "device": cfg.get("device", "cpu"),
        "denoising_end": cfg.get("denoising_end"),
    })
    lat_init = SDXLLatentInitNode("latent_init", config={
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "batch_size": cfg.get("batch_size", 1),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "float16"),
        "seed": cfg.get("seed"),
    })
    unet_node = SDXLUNetNode("unet", unet=unet, config={
        "guidance_scale": cfg.get("guidance_scale", 7.5),
        "guidance_rescale": cfg.get("guidance_rescale", 0.0),
    })
    sched_step = SDXLSchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = SDXLVAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
    })

    for nid, node in [
        ("tokenizer", tok_node), ("prompt_enc", prompt_enc), ("added_cond", added_cond),
        ("sched_setup", sched_setup), ("latent_init", lat_init),
        ("unet", unet_node), ("sched_step", sched_step), ("vae_decode", vae_dec),
    ]:
        h.add_node(nid, node)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS_2, "prompt_enc", C.PORT_INPUT_IDS_2))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS_2, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS_2))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "unet", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "unet", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_PROMPT_EMBEDS, "unet", C.PORT_NEGATIVE_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_POOLED_PROMPT_EMBEDS, "added_cond", C.PORT_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS,
                     "added_cond", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS,
                     "unet", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("added_cond", C.PORT_ADD_TEXT_EMBEDS, "unet", C.PORT_ADD_TEXT_EMBEDS))
    h.add_edge(Edge("added_cond", C.PORT_ADD_TIME_IDS, "unet", C.PORT_ADD_TIME_IDS))
    h.add_edge(Edge("added_cond", C.PORT_NEGATIVE_ADD_TIME_IDS, "unet", C.PORT_NEGATIVE_ADD_TIME_IDS))
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
    h.expose_input("tokenizer", C.PORT_PROMPT_2, C.PORT_PROMPT_2)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {
        "num_loop_steps": cfg.get("num_inference_steps", 50),
        "node_aliases": {
            "tokenizer": "tokenizer",
            "prompt_encoder": "prompt_enc",
            "conditioning": "added_cond",
            "backbone": "unet",
            "scheduler": ["sched_setup", "sched_step"],
            "latent_initializer": "latent_init",
            "autoencoder": "vae_decode",
        },
    }

    return h


def build_sdxl_img2img_graph(
    *,
    tokenizer: Any = None,
    tokenizer_2: Any = None,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    unet: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical SDXL image-to-image hypergraph."""
    from yggdrasill.integrations.diffusers.sdxl.tokenizer import SDXLTokenizerNode
    from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
    from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
    from yggdrasill.integrations.diffusers.sdxl.scheduler import SDXLSchedulerSetupNode, SDXLSchedulerStepNode
    from yggdrasill.integrations.diffusers.sdxl.latent_init import SDXLLatentInitNode
    from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEEncodeNode, SDXLVAEDecodeNode

    cfg = config or {}
    h = Hypergraph(graph_id="sdxl_img2img")

    tok_node = SDXLTokenizerNode("tokenizer", tokenizer=tokenizer, tokenizer_2=tokenizer_2)
    prompt_enc = SDXLPromptEncoderNode(
        "prompt_enc", text_encoder=text_encoder, text_encoder_2=text_encoder_2,
    )
    added_cond = SDXLAddedConditioningNode("added_cond", config={
        "original_size": cfg.get("original_size", (1024, 1024)),
        "target_size": cfg.get("target_size", (1024, 1024)),
        "requires_aesthetics_score": cfg.get("requires_aesthetics_score", False),
    })
    img_enc = SDXLVAEEncodeNode("img_encode", vae=vae, config={
        "height": cfg.get("height", 1024), "width": cfg.get("width", 1024),
        "device": cfg.get("device", "cpu"),
    })
    sched_setup = SDXLSchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 50),
        "device": cfg.get("device", "cpu"),
        "denoising_start": cfg.get("denoising_start"),
    })
    lat_init = SDXLLatentInitNode("latent_init", config={
        "height": cfg.get("height", 1024),
        "width": cfg.get("width", 1024),
        "batch_size": cfg.get("batch_size", 1),
        "device": cfg.get("device", "cpu"),
        "dtype": cfg.get("dtype", "float16"),
        "seed": cfg.get("seed"),
        "strength": cfg.get("strength", 0.8),
    })
    unet_node = SDXLUNetNode("unet", unet=unet, config={
        "guidance_scale": cfg.get("guidance_scale", 7.5),
    })
    sched_step = SDXLSchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = SDXLVAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
    })

    for nid, node in [
        ("tokenizer", tok_node), ("prompt_enc", prompt_enc), ("added_cond", added_cond),
        ("img_encode", img_enc), ("sched_setup", sched_setup),
        ("latent_init", lat_init), ("unet", unet_node),
        ("sched_step", sched_step), ("vae_decode", vae_dec),
    ]:
        h.add_node(nid, node)

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS_2, "prompt_enc", C.PORT_INPUT_IDS_2))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS_2, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS_2))
    h.add_edge(Edge("img_encode", C.PORT_LATENTS, "latent_init", C.PORT_INIT_LATENTS))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "unet", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "unet", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_PROMPT_EMBEDS, "unet", C.PORT_NEGATIVE_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_POOLED_PROMPT_EMBEDS, "added_cond", C.PORT_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS,
                     "added_cond", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS,
                     "unet", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("added_cond", C.PORT_ADD_TEXT_EMBEDS, "unet", C.PORT_ADD_TEXT_EMBEDS))
    h.add_edge(Edge("added_cond", C.PORT_ADD_TIME_IDS, "unet", C.PORT_ADD_TIME_IDS))
    h.add_edge(Edge("added_cond", C.PORT_NEGATIVE_ADD_TIME_IDS, "unet", C.PORT_NEGATIVE_ADD_TIME_IDS))
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

    h.metadata = {
        "num_loop_steps": cfg.get("num_inference_steps", 50),
        "node_aliases": {
            "tokenizer": "tokenizer",
            "prompt_encoder": "prompt_enc",
            "conditioning": "added_cond",
            "image_encoder": "img_encode",
            "backbone": "unet",
            "scheduler": ["sched_setup", "sched_step"],
            "latent_initializer": "latent_init",
            "autoencoder": "vae_decode",
        },
    }
    return h


def build_sdxl_inpaint_graph(
    *,
    tokenizer: Any = None,
    tokenizer_2: Any = None,
    text_encoder: Any = None,
    text_encoder_2: Any = None,
    unet: Any = None,
    vae: Any = None,
    scheduler: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Hypergraph:
    """Build a canonical SDXL inpainting hypergraph.

    9-channel UNet (official inpaint checkpoint): mask + masked-image latents are
    concatenated at the UNet input. 4-channel UNet uses per-step latent compositing
    via ``common/inpaint_latent_blend`` (same semantics as SD1.5 inpaint).
    """
    from yggdrasill.integrations.diffusers.common.inpaint_latent_blend import InpaintLatentBlendNode
    from yggdrasill.integrations.diffusers.common.mask_prep import InpaintMaskPrepNode
    from yggdrasill.integrations.diffusers.sdxl.added_conditioning import SDXLAddedConditioningNode
    from yggdrasill.integrations.diffusers.sdxl.latent_init import SDXLLatentInitNode
    from yggdrasill.integrations.diffusers.sdxl.prompt_encoder import SDXLPromptEncoderNode
    from yggdrasill.integrations.diffusers.sdxl.scheduler import SDXLSchedulerSetupNode, SDXLSchedulerStepNode
    from yggdrasill.integrations.diffusers.sdxl.tokenizer import SDXLTokenizerNode
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode
    from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEEncodeNode, SDXLVAEDecodeNode

    cfg = config or {}
    unet_cfg = getattr(unet, "config", None)
    unet_in_ch = int(getattr(unet_cfg, "in_channels", 9)) if unet_cfg is not None else 9
    use_4ch_inpaint_blend = unet_in_ch == 4

    h = Hypergraph(graph_id="sdxl_inpaint")

    tok_node = SDXLTokenizerNode("tokenizer", tokenizer=tokenizer, tokenizer_2=tokenizer_2)
    prompt_enc = SDXLPromptEncoderNode(
        "prompt_enc", text_encoder=text_encoder, text_encoder_2=text_encoder_2,
    )
    added_cond = SDXLAddedConditioningNode("added_cond", config={
        "original_size": cfg.get("original_size", (1024, 1024)),
        "target_size": cfg.get("target_size", (1024, 1024)),
        "crops_coords_top_left": cfg.get("crops_coords_top_left", (0, 0)),
        "requires_aesthetics_score": cfg.get("requires_aesthetics_score", False),
    })
    img_encode = SDXLVAEEncodeNode("img_encode", vae=vae, config={
        "height": cfg.get("height", 1024), "width": cfg.get("width", 1024),
        "device": cfg.get("device", "cpu"),
    })
    mask_prep = InpaintMaskPrepNode("mask_prep", vae=vae, config={
        "height": cfg.get("height", 1024), "width": cfg.get("width", 1024),
        "device": cfg.get("device", "cpu"),
    })
    sched_setup = SDXLSchedulerSetupNode("sched_setup", scheduler=scheduler, config={
        "num_inference_steps": cfg.get("num_inference_steps", 50),
        "device": cfg.get("device", "cpu"),
    })
    lat_init = SDXLLatentInitNode("latent_init", config={
        "height": cfg.get("height", 1024), "width": cfg.get("width", 1024),
        "device": cfg.get("device", "cpu"), "dtype": cfg.get("dtype", "float16"),
        "seed": cfg.get("seed"),
        "strength": cfg.get("strength", 1.0),
        "inpaint_4ch_composite": use_4ch_inpaint_blend,
    })
    unet_node = SDXLUNetNode("unet", unet=unet, config={
        "guidance_scale": cfg.get("guidance_scale", 7.5),
        "guidance_rescale": cfg.get("guidance_rescale", 0.0),
    })
    sched_step = SDXLSchedulerStepNode("sched_step", scheduler=scheduler)
    vae_dec = SDXLVAEDecodeNode("vae_decode", vae=vae, config={
        "output_type": cfg.get("output_type", "pil"),
    })

    nodes_list = [
        ("tokenizer", tok_node), ("prompt_enc", prompt_enc), ("added_cond", added_cond),
        ("img_encode", img_encode), ("mask_prep", mask_prep), ("sched_setup", sched_setup),
        ("latent_init", lat_init), ("unet", unet_node),
        ("sched_step", sched_step), ("vae_decode", vae_dec),
    ]
    for nid, node in nodes_list:
        h.add_node(nid, node)
    if use_4ch_inpaint_blend:
        h.add_node("inpaint_blend", InpaintLatentBlendNode("inpaint_blend", config={}))

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS_2, "prompt_enc", C.PORT_INPUT_IDS_2))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS_2, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS_2))
    h.add_edge(Edge("img_encode", C.PORT_LATENTS, "latent_init", C.PORT_INIT_LATENTS))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "unet", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "unet", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_PROMPT_EMBEDS, "unet", C.PORT_NEGATIVE_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_POOLED_PROMPT_EMBEDS, "added_cond", C.PORT_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS,
                     "added_cond", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS,
                     "unet", C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS))
    h.add_edge(Edge("added_cond", C.PORT_ADD_TEXT_EMBEDS, "unet", C.PORT_ADD_TEXT_EMBEDS))
    h.add_edge(Edge("added_cond", C.PORT_ADD_TIME_IDS, "unet", C.PORT_ADD_TIME_IDS))
    h.add_edge(Edge("added_cond", C.PORT_NEGATIVE_ADD_TIME_IDS, "unet", C.PORT_NEGATIVE_ADD_TIME_IDS))
    h.add_edge(Edge("mask_prep", C.PORT_MASK_LATENTS, "unet", C.PORT_MASK_LATENTS))
    h.add_edge(Edge("mask_prep", C.PORT_MASKED_IMAGE_LATENTS, "unet", C.PORT_MASKED_IMAGE_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_LATENTS, "unet", C.PORT_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_LATENTS, "sched_step", C.PORT_LATENTS))
    h.add_edge(Edge("latent_init", C.PORT_TIMESTEP, "unet", C.PORT_TIMESTEP))
    h.add_edge(Edge("latent_init", C.PORT_TIMESTEP, "sched_step", C.PORT_TIMESTEP))
    h.add_edge(Edge("unet", C.PORT_NOISE_PRED, "sched_step", C.PORT_NOISE_PRED))
    h.add_edge(Edge("sched_step", "next_timestep", "unet", C.PORT_TIMESTEP))
    h.add_edge(Edge("sched_step", "next_timestep", "sched_step", C.PORT_TIMESTEP))
    if use_4ch_inpaint_blend:
        h.add_edge(Edge("sched_step", "next_latent", "inpaint_blend", "latents_post_step"))
        h.add_edge(Edge("inpaint_blend", "next_latent", "unet", C.PORT_LATENTS))
        h.add_edge(Edge("inpaint_blend", "next_latent", "sched_step", C.PORT_LATENTS))
        h.add_edge(Edge("inpaint_blend", "next_latent", "vae_decode", C.PORT_LATENTS))
        h.add_edge(Edge("img_encode", C.PORT_LATENTS, "inpaint_blend", C.PORT_CLEAN_IMAGE_LATENTS))
        h.add_edge(Edge("mask_prep", C.PORT_MASK_LATENTS, "inpaint_blend", C.PORT_MASK_LATENTS))
        h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "inpaint_blend", C.PORT_SCHEDULER_STATE))
    else:
        h.add_edge(Edge("sched_step", "next_latent", "unet", C.PORT_LATENTS))
        h.add_edge(Edge("sched_step", "next_latent", "sched_step", C.PORT_LATENTS))
        h.add_edge(Edge("sched_step", "next_latent", "vae_decode", C.PORT_LATENTS))

    h.expose_input("tokenizer", C.PORT_PROMPT, C.PORT_PROMPT)
    h.expose_input("tokenizer", C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT)
    h.expose_input("tokenizer", C.PORT_PROMPT_2, C.PORT_PROMPT_2)
    h.expose_input("img_encode", C.PORT_INIT_IMAGE, C.PORT_INIT_IMAGE)
    h.expose_input("mask_prep", C.PORT_INIT_IMAGE, C.PORT_INIT_IMAGE)
    h.expose_input("mask_prep", C.PORT_MASK_IMAGE, C.PORT_MASK_IMAGE)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {
        "num_loop_steps": cfg.get("num_inference_steps", 50),
        "node_aliases": {
            "tokenizer": "tokenizer",
            "prompt_encoder": "prompt_enc",
            "conditioning": "added_cond",
            "image_encoder": "img_encode",
            "mask_processor": "mask_prep",
            "backbone": "unet",
            "scheduler": ["sched_setup", "sched_step"],
            "latent_initializer": "latent_init",
            "autoencoder": "vae_decode",
        },
    }
    return h


def _sdxl_inpaint_topology_node_ids(graph: Hypergraph) -> Dict[str, str]:
    """Canonical node ids for SDXL inpaint / universal rewiring."""
    meta = getattr(graph, "metadata", None) or {}
    custom = meta.get("sdxl_role_ids")
    if isinstance(custom, dict) and custom.get("scheduler_step") and custom.get("unet"):
        return {
            "scheduler_step": str(custom["scheduler_step"]),
            "unet": str(custom["unet"]),
            "vae_decode": str(custom.get("vae_decode", "vae_decode")),
            "scheduler_setup": str(custom.get("scheduler_setup", "sched_setup")),
            "img_encode": str(custom.get("img_encode", "img_encode")),
            "mask_prep": str(custom.get("mask_prep", "mask_prep")),
            "latent_init": str(custom.get("latent_init", "latent_init")),
        }
    return {
        "scheduler_step": "sched_step",
        "unet": "unet",
        "vae_decode": "vae_decode",
        "scheduler_setup": "sched_setup",
        "img_encode": "img_encode",
        "mask_prep": "mask_prep",
        "latent_init": "latent_init",
    }


def _remove_sdxl_inpaint_direct_sched_latent_edges(graph: Hypergraph, ids: Dict[str, str]) -> None:
    ss, u, v = ids["scheduler_step"], ids["unet"], ids["vae_decode"]
    targets = {(u, C.PORT_LATENTS), (ss, C.PORT_LATENTS), (v, C.PORT_LATENTS)}
    for e in list(graph.get_edges()):
        if e.source_node == ss and e.source_port == "next_latent":
            if (e.target_node, e.target_port) in targets:
                graph.remove_edge(e)


def reconfigure_sdxl_inpaint_for_unet_in_channels(graph: Hypergraph, *, in_channels: int) -> None:
    """Rewire SDXL inpaint / universal graph when UNet ``in_channels`` changes (4 ↔ 9)."""
    meta = getattr(graph, "metadata", None) or {}
    if graph.graph_id != "sdxl_inpaint" and not meta.get("sdxl_universal"):
        return

    ids = _sdxl_inpaint_topology_node_ids(graph)
    ss, u, v, su, im, mp = (
        ids["scheduler_step"],
        ids["unet"],
        ids["vae_decode"],
        ids["scheduler_setup"],
        ids["img_encode"],
        ids["mask_prep"],
    )
    li_nid = ids["latent_init"]

    need_blend = in_channels == 4
    has_blend = "inpaint_blend" in graph.node_ids

    if need_blend:
        if not has_blend:
            from yggdrasill.integrations.diffusers.common.inpaint_latent_blend import (
                InpaintLatentBlendNode,
            )

            graph.add_node(
                "inpaint_blend",
                InpaintLatentBlendNode("inpaint_blend", config={}),
            )
            _remove_sdxl_inpaint_direct_sched_latent_edges(graph, ids)
            graph.add_edge(Edge(ss, "next_latent", "inpaint_blend", "latents_post_step"))
            graph.add_edge(Edge("inpaint_blend", "next_latent", u, C.PORT_LATENTS))
            graph.add_edge(Edge("inpaint_blend", "next_latent", ss, C.PORT_LATENTS))
            graph.add_edge(Edge("inpaint_blend", "next_latent", v, C.PORT_LATENTS))
            graph.add_edge(Edge(im, C.PORT_LATENTS, "inpaint_blend", C.PORT_CLEAN_IMAGE_LATENTS))
            graph.add_edge(Edge(mp, C.PORT_MASK_LATENTS, "inpaint_blend", C.PORT_MASK_LATENTS))
            graph.add_edge(Edge(su, C.PORT_SCHEDULER_STATE, "inpaint_blend", C.PORT_SCHEDULER_STATE))
    else:
        if has_blend:
            graph.remove_node("inpaint_blend")
            graph.add_edge(Edge(ss, "next_latent", u, C.PORT_LATENTS))
            graph.add_edge(Edge(ss, "next_latent", ss, C.PORT_LATENTS))
            graph.add_edge(Edge(ss, "next_latent", v, C.PORT_LATENTS))

    li = graph.get_node(li_nid)
    if li is not None and hasattr(li, "_config"):
        li._config = dict(li._config or {})
        li._config["inpaint_4ch_composite"] = need_blend


def build_sdxl_base_refiner_workflow(
    *,
    base_components: Dict[str, Any],
    refiner_components: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Workflow:
    """Build SDXL base+refiner as a two-stage Workflow."""
    cfg = config or {}
    denoising_end = cfg.get("denoising_end", 0.8)

    base_cfg = dict(cfg)
    base_cfg["output_type"] = "latent"
    base_cfg["denoising_end"] = denoising_end

    base_graph = build_sdxl_text2img_graph(**base_components, config=base_cfg)

    refiner_cfg = dict(cfg)
    refiner_cfg["denoising_start"] = denoising_end
    refiner_cfg.setdefault("requires_aesthetics_score", True)

    refiner_graph = build_sdxl_img2img_graph(**refiner_components, config=refiner_cfg)

    w = Workflow(workflow_id="sdxl_base_refiner")
    w.add_node("base", base_graph)
    w.add_node("refiner", refiner_graph)
    w.add_edge("base", C.PORT_DECODED_IMAGE, "refiner", C.PORT_INIT_IMAGE)
    w.expose_input("base", C.PORT_PROMPT, C.PORT_PROMPT)
    w.expose_input("base", C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT)
    w.expose_output("refiner", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    return w
