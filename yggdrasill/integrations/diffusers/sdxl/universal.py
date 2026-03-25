"""Manual SDXL DiffusionGraphBuilder: one graph for text2img / img2img / inpaint (4-ch UNet).

Completion wires ``img_encode``, ``mask_prep``, ``latent_init``, and ``inpaint_blend``
when the backbone has 4 input channels — same run-time semantics as SD1.5 universal.
9-channel inpaint checkpoints should use the ``sdxl_inpaint`` preset instead.
"""
from __future__ import annotations

from typing import Dict, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph

from yggdrasill.integrations.diffusers.adapter_rewire import (
    rewire_adapters_after_latent_stack,
)
from yggdrasill.integrations.diffusers.presets.sdxl import (
    reconfigure_sdxl_inpaint_for_unet_in_channels,
)


def discover_sdxl_manual_stack_roles(graph: Hypergraph) -> Optional[Dict[str, str]]:
    """Map role name → node_id for a manually added SDXL stack (any node ids).

    ``added_conditioning`` is optional: :func:`try_complete_sdxl_universal_diffusion`
    can insert a default node when it is missing (same ergonomics as SD1.5 stacks).
    """
    roles: Dict[str, str] = {}
    for nid in sorted(graph.node_ids):
        node = graph.get_node(nid)
        bt = getattr(node, "block_type", "") or ""
        if not bt.startswith("sdxl/"):
            continue
        if "tokenizer" in bt and "tokenizer" not in roles:
            roles["tokenizer"] = nid
        elif "prompt_encoder" in bt and "prompt_encoder" not in roles:
            roles["prompt_encoder"] = nid
        elif "added_conditioning" in bt and "added_conditioning" not in roles:
            roles["added_conditioning"] = nid
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


def try_complete_sdxl_universal_diffusion(graph: Hypergraph) -> bool:
    """If graph is an incomplete manual SDXL stack, add universal I2I/inpaint nodes and edges.

    Sets ``metadata['sdxl_universal']`` and ``metadata['sdxl_role_ids']``. Returns False
    if not a candidate (e.g. 9-ch UNet). When ``added_conditioning`` is absent, inserts
    ``added_cond`` (or ``_sdxl_universal_added_cond`` if that id is taken) with size
    taken from the scheduler setup node.
    """
    meta = getattr(graph, "metadata", None) or {}
    if meta.get("sdxl_universal"):
        return True

    roles = discover_sdxl_manual_stack_roles(graph)
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

    h, w = 1024, 1024
    for node in (graph.get_node(roles["scheduler_setup"]),):
        if node is not None and getattr(node, "_config", None):
            h = int(node._config.get("height", h))
            w = int(node._config.get("width", w))

    if "added_conditioning" not in roles:
        from yggdrasill.integrations.diffusers.sdxl.added_conditioning import (
            SDXLAddedConditioningNode,
        )

        ac_nid = "added_cond"
        if ac_nid in graph.node_ids:
            ac_nid = "_sdxl_universal_added_cond"
        graph.add_node(
            ac_nid,
            SDXLAddedConditioningNode(
                ac_nid,
                config={
                    "original_size": (h, w),
                    "target_size": (h, w),
                    "crops_coords_top_left": (0, 0),
                },
            ),
            auto_connect=False,
        )
        roles["added_conditioning"] = ac_nid

    from yggdrasill.integrations.diffusers.common.mask_prep import InpaintMaskPrepNode
    from yggdrasill.integrations.diffusers.sdxl.latent_init import SDXLLatentInitNode
    from yggdrasill.integrations.diffusers.sdxl.vae import SDXLVAEEncodeNode

    enc_cfg = {"height": h, "width": w, "device": device}
    graph.add_node(
        "img_encode",
        SDXLVAEEncodeNode("img_encode", vae=vae, config=enc_cfg),
        auto_connect=False,
    )
    graph.add_node(
        "mask_prep",
        InpaintMaskPrepNode("mask_prep", vae=vae, config=dict(enc_cfg)),
        auto_connect=False,
    )
    graph.add_node(
        "latent_init",
        SDXLLatentInitNode(
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
    ac = roles["added_conditioning"]
    u = roles["unet"]
    su = roles["scheduler_setup"]
    ss = roles["scheduler_step"]
    vd = roles["vae_decode"]

    graph.add_edge(Edge(tok, C.PORT_INPUT_IDS, pe, C.PORT_INPUT_IDS))
    graph.add_edge(Edge(tok, C.PORT_INPUT_IDS_2, pe, C.PORT_INPUT_IDS_2))
    graph.add_edge(Edge(tok, C.PORT_NEGATIVE_INPUT_IDS, pe, C.PORT_NEGATIVE_INPUT_IDS))
    graph.add_edge(Edge(tok, C.PORT_NEGATIVE_INPUT_IDS_2, pe, C.PORT_NEGATIVE_INPUT_IDS_2))
    graph.add_edge(Edge(pe, C.PORT_POOLED_PROMPT_EMBEDS, ac, C.PORT_POOLED_PROMPT_EMBEDS))
    graph.add_edge(Edge(pe, C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS, ac, C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS))
    graph.add_edge(Edge(ac, C.PORT_ADD_TEXT_EMBEDS, u, C.PORT_ADD_TEXT_EMBEDS))
    graph.add_edge(Edge(ac, C.PORT_ADD_TIME_IDS, u, C.PORT_ADD_TIME_IDS))
    graph.add_edge(Edge(ac, C.PORT_NEGATIVE_ADD_TIME_IDS, u, C.PORT_NEGATIVE_ADD_TIME_IDS))
    graph.add_edge(Edge("img_encode", C.PORT_LATENTS, "latent_init", C.PORT_INIT_LATENTS))
    graph.add_edge(Edge(su, C.PORT_SCHEDULER_STATE, "latent_init", C.PORT_SCHEDULER_STATE))
    graph.add_edge(Edge(su, C.PORT_SCHEDULER_STATE, u, C.PORT_SCHEDULER_STATE))
    graph.add_edge(Edge(pe, C.PORT_PROMPT_EMBEDS, u, C.PORT_PROMPT_EMBEDS))
    graph.add_edge(Edge(pe, C.PORT_NEGATIVE_PROMPT_EMBEDS, u, C.PORT_NEGATIVE_PROMPT_EMBEDS))
    graph.add_edge(Edge(pe, C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS, u, C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS))
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

    graph.metadata["sdxl_universal"] = True
    graph.metadata["sdxl_role_ids"] = {
        "tokenizer": tok,
        "prompt_encoder": pe,
        "added_conditioning": ac,
        "unet": u,
        "scheduler_setup": su,
        "scheduler_step": ss,
        "vae_decode": vd,
        "img_encode": "img_encode",
        "mask_prep": "mask_prep",
        "latent_init": "latent_init",
    }
    graph.metadata.setdefault("num_loop_steps", 50)

    reconfigure_sdxl_inpaint_for_unet_in_channels(graph, in_channels=4)

    rewire_adapters_after_latent_stack(graph)

    graph.expose_input(tok, C.PORT_PROMPT, C.PORT_PROMPT)
    graph.expose_input(tok, C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT)
    graph.expose_input(tok, C.PORT_PROMPT_2, C.PORT_PROMPT_2)
    graph.expose_input("img_encode", C.PORT_INIT_IMAGE, "image")
    graph.expose_input("mask_prep", C.PORT_INIT_IMAGE, "image")
    graph.expose_input("mask_prep", C.PORT_MASK_IMAGE, "mask_image")
    graph.expose_output(vd, C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)
    return True
