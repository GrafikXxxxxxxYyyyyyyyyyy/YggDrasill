"""Manual SD1.5 DiffusionGraphBuilder: one graph for text2img / img2img / inpaint.

Completion wires img_encode, mask_prep, latent_init (+ 4-ch blend when needed).
At run time, ``image`` / ``mask_image`` are optional exposed inputs; if ``image`` is
absent, ``img_encode`` and ``mask_prep`` are skipped (text2img). If ``image`` is
present but ``mask_image`` is absent, ``mask_prep`` defaults to a full repaint mask
(img2img). With both, inpaint semantics apply.
"""
from __future__ import annotations

from typing import Dict, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph

from yggdrasill.integrations.diffusers.adapter_rewire import (
    rewire_adapters_after_latent_stack,
)
from yggdrasill.integrations.diffusers.presets.sd15 import (
    reconfigure_sd15_inpaint_for_unet_in_channels,
)


def discover_sd15_manual_stack_roles(graph: Hypergraph) -> Optional[Dict[str, str]]:
    """Map role name → node_id for a manually added SD1.5 stack (any node ids).

    Returns None if required roles are missing or graph is not SD1.5-only for these nodes.
    """
    roles: Dict[str, str] = {}
    for nid in sorted(graph.node_ids):
        node = graph.get_node(nid)
        bt = getattr(node, "block_type", "") or ""
        if not bt.startswith("sd15/"):
            continue
        if "tokenizer" in bt and "tokenizer" not in roles:
            roles["tokenizer"] = nid
        elif "prompt_encoder" in bt and "prompt_encoder" not in roles:
            roles["prompt_encoder"] = nid
        elif bt.endswith("/unet") and "unet" not in roles:
            roles["unet"] = nid
        elif "scheduler_setup" in bt and "scheduler_setup" not in roles:
            roles["scheduler_setup"] = nid
        elif "scheduler_step" in bt and "scheduler_step" not in roles:
            roles["scheduler_step"] = nid
        elif "vae_decode" in bt and "vae_decode" not in roles:
            roles["vae_decode"] = nid

    required = (
        "tokenizer",
        "prompt_encoder",
        "unet",
        "scheduler_setup",
        "scheduler_step",
        "vae_decode",
    )
    if not all(k in roles for k in required):
        return None
    return roles


def try_complete_sd15_universal_diffusion(graph: Hypergraph) -> bool:
    """If graph is an incomplete manual SD1.5 stack, add universal I2I/inpaint nodes and edges.

    Sets ``metadata['sd15_universal']`` and ``metadata['sd15_role_ids']`` for rewiring
    when the backbone is replaced. Returns False if this graph is not a candidate.
    """
    meta = getattr(graph, "metadata", None) or {}
    if meta.get("sd15_universal"):
        return True

    roles = discover_sd15_manual_stack_roles(graph)
    if roles is None:
        return False

    for nid in graph.node_ids:
        node = graph.get_node(nid)
        bt = getattr(node, "block_type", "") or ""
        if "latent_init" in bt or "vae_encode" in bt or "mask_prep" in bt:
            return False

    unet_node = graph.get_node(roles["unet"])
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    unet_mod = resolve_if_lazy(getattr(unet_node, "_unet", None))
    uc = getattr(unet_mod, "config", None)
    # 9-ch inpainting UNet needs mask concat every run; use template ``sd15_inpaint`` instead.
    if uc is None or int(getattr(uc, "in_channels", 4)) != 4:
        return False

    vae_node = graph.get_node(roles["vae_decode"])
    vae = getattr(vae_node, "_vae", None)
    if vae is None:
        return False

    device = "cpu"
    for node in (graph.get_node(roles["scheduler_setup"]), vae_node):
        if node is not None and getattr(node, "_config", None):
            device = node._config.get("device", device)

    h, w = 512, 512
    for node in (graph.get_node(roles["scheduler_setup"]),):
        if node is not None and getattr(node, "_config", None):
            h = int(node._config.get("height", h))
            w = int(node._config.get("width", w))

    from yggdrasill.integrations.diffusers.sd15.latent_init import SD15LatentInitNode
    from yggdrasill.integrations.diffusers.sd15.vae import SD15VAEEncodeNode
    from yggdrasill.integrations.diffusers.common.mask_prep import InpaintMaskPrepNode

    enc_cfg = {"height": h, "width": w, "device": device}
    graph.add_node(
        "img_encode",
        SD15VAEEncodeNode("img_encode", vae=vae, config=enc_cfg),
        auto_connect=False,
    )
    graph.add_node(
        "mask_prep",
        InpaintMaskPrepNode("mask_prep", vae=vae, config=dict(enc_cfg)),
        auto_connect=False,
    )
    graph.add_node(
        "latent_init",
        SD15LatentInitNode(
            "latent_init",
            config={
                "height": h,
                "width": w,
                "device": device,
                "dtype": "float16",
                "strength": 1.0,
                "inpaint_4ch_composite": False,
            },
        ),
        auto_connect=False,
    )

    tok = roles["tokenizer"]
    pe = roles["prompt_encoder"]
    u = roles["unet"]
    su = roles["scheduler_setup"]
    ss = roles["scheduler_step"]
    vd = roles["vae_decode"]

    graph.add_edge(Edge(tok, C.PORT_INPUT_IDS, pe, C.PORT_INPUT_IDS))
    graph.add_edge(Edge(tok, C.PORT_NEGATIVE_INPUT_IDS, pe, C.PORT_NEGATIVE_INPUT_IDS))
    graph.add_edge(Edge("img_encode", C.PORT_LATENTS, "latent_init", C.PORT_INIT_LATENTS))
    graph.add_edge(Edge(su, C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    graph.add_edge(Edge(su, C.PORT_SCHEDULER_STATE, u, C.PORT_SCHEDULER_STATE))
    graph.add_edge(Edge(pe, C.PORT_PROMPT_EMBEDS, u, C.PORT_PROMPT_EMBEDS))
    graph.add_edge(Edge(pe, C.PORT_NEGATIVE_PROMPT_EMBEDS, u, C.PORT_NEGATIVE_PROMPT_EMBEDS))
    graph.add_edge(Edge("latent_init", C.PORT_LATENTS, u, C.PORT_LATENTS))
    graph.add_edge(Edge("latent_init", C.PORT_LATENTS, ss, C.PORT_LATENTS))
    graph.add_edge(Edge("latent_init", C.PORT_TIMESTEP, u, C.PORT_TIMESTEP))
    graph.add_edge(Edge("latent_init", C.PORT_TIMESTEP, ss, C.PORT_TIMESTEP))
    graph.add_edge(Edge(u, C.PORT_NOISE_PRED, ss, C.PORT_NOISE_PRED))
    graph.add_edge(Edge(ss, "next_timestep", u, C.PORT_TIMESTEP))
    graph.add_edge(Edge(ss, "next_timestep", ss, C.PORT_TIMESTEP))
    graph.add_edge(Edge(ss, "next_latent", u, C.PORT_LATENTS))
    graph.add_edge(Edge(ss, "next_latent", ss, C.PORT_LATENTS))
    graph.add_edge(Edge(ss, "next_latent", vd, C.PORT_LATENTS))

    graph.metadata["sd15_universal"] = True
    graph.metadata["sd15_role_ids"] = {
        "tokenizer": tok,
        "prompt_encoder": pe,
        "unet": u,
        "scheduler_setup": su,
        "scheduler_step": ss,
        "vae_decode": vd,
        "img_encode": "img_encode",
        "mask_prep": "mask_prep",
        "latent_init": "latent_init",
    }
    graph.metadata.setdefault("num_loop_steps", 50)

    reconfigure_sd15_inpaint_for_unet_in_channels(graph, in_channels=4)

    rewire_adapters_after_latent_stack(graph)

    graph.expose_input(tok, C.PORT_PROMPT, C.PORT_PROMPT)
    graph.expose_input(tok, C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT)
    graph.expose_input("img_encode", C.PORT_INIT_IMAGE, "image")
    graph.expose_input("mask_prep", C.PORT_INIT_IMAGE, "image")
    graph.expose_input("mask_prep", C.PORT_MASK_IMAGE, "mask_image")
    graph.expose_output(vd, C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)
    return True
