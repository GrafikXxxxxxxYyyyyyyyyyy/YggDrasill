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
    enc = SD15PromptEncoderNode("prompt_enc", text_encoder=text_encoder,
                                 config={"clip_skip": cfg.get("clip_skip")})
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
        "strength": cfg.get("strength", 0.8),
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
    """Build a canonical SD1.5 inpainting hypergraph.

    If ``unet.config.in_channels == 9`` (e.g. ``runwayml/stable-diffusion-inpainting``), mask and
    masked-image latents are concatenated at the UNet input (Diffusers default).

    If ``in_channels == 4`` (e.g. base SD1.5 UNet), uses the same per-step latent
    compositing as ``StableDiffusionInpaintPipeline`` for 4-channel UNet: after each
    ``scheduler.step``, unmasked regions are mixed back from noised clean latents.
    """
    from yggdrasill.integrations.diffusers.sd15.tokenizer import SD15TokenizerNode
    from yggdrasill.integrations.diffusers.sd15.prompt_encoder import SD15PromptEncoderNode
    from yggdrasill.integrations.diffusers.sd15.unet import SD15UNetNode
    from yggdrasill.integrations.diffusers.sd15.scheduler import (
        SD15SchedulerSetupNode,
        SD15SchedulerStepNode,
    )
    from yggdrasill.integrations.diffusers.sd15.vae import (
        SD15VAEEncodeNode,
        SD15VAEDecodeNode,
    )
    from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode
    from yggdrasill.integrations.diffusers.sd15.inpaint_blend import (
        SD15InpaintFourChannelBlendNode,
    )
    from yggdrasill.integrations.diffusers.common.mask_prep import InpaintMaskPrepNode

    cfg = config or {}

    unet_cfg = getattr(unet, "config", None)
    unet_in_ch = int(getattr(unet_cfg, "in_channels", 9)) if unet_cfg is not None else 9
    use_4ch_inpaint_blend = unet_in_ch == 4

    h = Hypergraph(graph_id="sd15_inpaint")

    tok_node = SD15TokenizerNode("tokenizer", tokenizer=tokenizer)
    enc = SD15PromptEncoderNode("prompt_enc", text_encoder=text_encoder,
                                 config={"clip_skip": cfg.get("clip_skip")})
    img_encode = SD15VAEEncodeNode("img_encode", vae=vae, config={
        "height": cfg.get("height", 512),
        "width": cfg.get("width", 512),
        "device": cfg.get("device", "cpu"),
    })
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
        "strength": cfg.get("strength", 1.0),
        "inpaint_4ch_composite": use_4ch_inpaint_blend,
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
    h.add_node("img_encode", img_encode)
    h.add_node("mask_prep", mask_prep)
    h.add_node("sched_setup", sched_setup)
    h.add_node("latent_init", lat_init)
    h.add_node("unet", unet_node)
    h.add_node("sched_step", sched_step)
    h.add_node("vae_decode", vae_dec)
    if use_4ch_inpaint_blend:
        h.add_node(
            "inpaint_blend",
            SD15InpaintFourChannelBlendNode("inpaint_blend", config={}),
        )

    h.add_edge(Edge("tokenizer", C.PORT_INPUT_IDS, "prompt_enc", C.PORT_INPUT_IDS))
    h.add_edge(Edge("tokenizer", C.PORT_NEGATIVE_INPUT_IDS, "prompt_enc", C.PORT_NEGATIVE_INPUT_IDS))
    h.add_edge(Edge("img_encode", C.PORT_LATENTS, "latent_init", C.PORT_INIT_LATENTS))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("sched_setup", C.PORT_SCHEDULER_STATE, "unet", C.PORT_SCHEDULER_STATE))
    h.add_edge(Edge("prompt_enc", C.PORT_PROMPT_EMBEDS, "unet", C.PORT_PROMPT_EMBEDS))
    h.add_edge(Edge("prompt_enc", C.PORT_NEGATIVE_PROMPT_EMBEDS, "unet", C.PORT_NEGATIVE_PROMPT_EMBEDS))
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
    # Same external key feeds full-image encode (latent init) and mask prep (masked region).
    h.expose_input("img_encode", C.PORT_INIT_IMAGE, C.PORT_INIT_IMAGE)
    h.expose_input("mask_prep", C.PORT_INIT_IMAGE, C.PORT_INIT_IMAGE)
    h.expose_input("mask_prep", C.PORT_MASK_IMAGE, C.PORT_MASK_IMAGE)
    h.expose_output("vae_decode", C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)

    h.metadata = {"num_loop_steps": cfg.get("num_inference_steps", 50)}

    return h


def _sd15_inpaint_topology_node_ids(graph: Hypergraph) -> Dict[str, str]:
    """Canonical node ids for inpaint / universal SD1.5 rewiring (preset or builder role map)."""
    meta = getattr(graph, "metadata", None) or {}
    custom = meta.get("sd15_role_ids")
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


def _remove_sd15_inpaint_direct_sched_latent_edges(graph: Hypergraph, ids: Dict[str, str]) -> None:
    """Drop sched.next_latent → (unet|sched|vae).latents."""
    ss, u, v = ids["scheduler_step"], ids["unet"], ids["vae_decode"]
    targets = {(u, C.PORT_LATENTS), (ss, C.PORT_LATENTS), (v, C.PORT_LATENTS)}
    for e in list(graph.get_edges()):
        if e.source_node == ss and e.source_port == "next_latent":
            if (e.target_node, e.target_port) in targets:
                graph.remove_edge(e)


def reconfigure_sd15_inpaint_for_unet_in_channels(graph: Hypergraph, *, in_channels: int) -> None:
    """Rewire SD1.5 inpaint / universal graph when UNet ``in_channels`` changes.

    The template is built for either 9-ch (mask concat) or 4-ch (per-step ``inpaint_blend``).
    Loading ``sd15_inpaint`` with the inpainting checkpoint yields 9-ch wiring **without**
    ``inpaint_blend``. Swapping in a 4-ch finetune (e.g. DreamShaper) without this call leaves
    the mask unused — behaviour collapses to image-to-image.
    """
    meta = getattr(graph, "metadata", None) or {}
    if graph.graph_id != "sd15_inpaint" and not meta.get("sd15_universal"):
        return

    ids = _sd15_inpaint_topology_node_ids(graph)
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
            from yggdrasill.integrations.diffusers.sd15.inpaint_blend import (
                SD15InpaintFourChannelBlendNode,
            )

            graph.add_node(
                "inpaint_blend",
                SD15InpaintFourChannelBlendNode("inpaint_blend", config={}),
            )
            _remove_sd15_inpaint_direct_sched_latent_edges(graph, ids)
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
