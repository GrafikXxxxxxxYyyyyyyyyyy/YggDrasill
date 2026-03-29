"""DiffusionGraphBuilder: high-level API for adding components to diffusion graphs.

Resolves component types via FamilyRegistry, loads from ModelStore,
adds implicit nodes, and delegates to graph.add_node with ready nodes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - exercised in no-diffusion environments
    torch = None  # type: ignore[assignment]

from yggdrasill.engine.edge import Edge
from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import (
    _load_ip_adapter_state_dict,
    reload_ip_adapter_weights_on_unet,
)
from yggdrasill.engine.structure import Hypergraph

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.components import (
    is_component_type,
    load_components_from_pretrained,
    resolve_component_type,
)
from yggdrasill.integrations.diffusers.family_registry import get_family_spec


_FALLBACK_SCHEDULER_REPO: Dict[str, str] = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-dev",
}

# After replace_component swaps UNet/transformer, encourage immediate CUDA reclaim
# (old weights are dropped in Hypergraph.remove_node via _release_node_gpu_backing).
_BACKBONE_REPLACE_COMPONENT_TYPES = frozenset({
    "sd15.unet",
    "sd15.backbone",
    "sdxl.unet",
    "sdxl.backbone",
    "flux.transformer",
    "flux.backbone",
})


def _reclaim_cuda_after_backbone_replace() -> None:
    import gc

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _instantiate_scheduler_by_type(
    scheduler_type: str,
    template: Any,
    family: str,
) -> Any:
    """Build a Diffusers scheduler from *scheduler_type*, optionally matching *template* config."""
    key = scheduler_type.strip().lower().replace("-", "_")
    import diffusers.schedulers as sched_mod

    ldm_classes: Dict[str, Any] = {
        "euler": sched_mod.EulerDiscreteScheduler,
        "euler_ancestral": sched_mod.EulerAncestralDiscreteScheduler,
        "ddim": sched_mod.DDIMScheduler,
        "ddpm": sched_mod.DDPMScheduler,
        "pndm": sched_mod.PNDMScheduler,
        "lms": sched_mod.LMSDiscreteScheduler,
        "dpm": sched_mod.DPMSolverMultistepScheduler,
        "dpm_solver": sched_mod.DPMSolverMultistepScheduler,
        "dpmsolver": sched_mod.DPMSolverMultistepScheduler,
        "dpmsolver_multistep": sched_mod.DPMSolverMultistepScheduler,
        "heun": sched_mod.HeunDiscreteScheduler,
        "unipc": sched_mod.UniPCMultistepScheduler,
    }
    flux_classes: Dict[str, Any] = {
        "flow_match": sched_mod.FlowMatchEulerDiscreteScheduler,
        "flux": sched_mod.FlowMatchEulerDiscreteScheduler,
        "euler": sched_mod.FlowMatchEulerDiscreteScheduler,
    }
    mapping = flux_classes if family == "flux" else ldm_classes
    cls = mapping.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown scheduler_type '{scheduler_type}' for family '{family}'. "
            f"Try one of: {', '.join(sorted(mapping.keys()))}"
        )

    if template is not None and hasattr(template, "config"):
        try:
            return cls.from_config(template.config)
        except Exception:
            pass

    repo = _FALLBACK_SCHEDULER_REPO.get(family, _FALLBACK_SCHEDULER_REPO["sd15"])
    return cls.from_pretrained(repo, subfolder="scheduler")


def _infer_hypergraph_device(graph: Any) -> Any:
    """Resolve target device for a diffusion graph.

    Order: ``metadata['device']`` (set by :meth:`Hypergraph.to`), then any
    non-cpu ``node._config['device']``, then ``cpu`` from config, then first
    module parameter device among common diffusion attributes.
    """
    meta = getattr(graph, "metadata", None) or {}
    d = meta.get("device")
    if d is not None:
        return d
    nodes = getattr(graph, "_nodes", None) or {}
    last_cpu: Any = None
    for node in nodes.values():
        cfg = getattr(node, "_config", None) or {}
        cd = cfg.get("device")
        if cd is None:
            continue
        if str(cd) != "cpu":
            return cd
        last_cpu = cd
    if last_cpu is not None:
        return last_cpu
    for node in nodes.values():
        for attr in (
            "_unet", "_vae", "_text_encoder", "_transformer", "_controlnet",
        ):
            mod = getattr(node, attr, None)
            if mod is None:
                continue
            try:
                p = next(mod.parameters(), None)
                if p is not None:
                    return p.device
            except Exception:
                continue
    return None


_META_IP_ADAPTER_SDS = "ip_adapter_accumulated_state_dicts"
_META_IP_ADAPTER_ORDER = "ip_adapter_weight_node_ids"


def _build_lora_pipe(graph: Any, *, family: str) -> Any:
    """Create a minimal diffusers LoRA loader host bound to graph modules.

    We do not use the pipeline for inference; only for `load_lora_weights` / `set_adapters`.
    """
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    unet = None
    text_encoder = None
    text_encoder_2 = None

    for nid in getattr(graph, "node_ids", ()) or ():
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        bt = getattr(node, "block_type", "") or ""
        if bt.endswith("/unet") and getattr(node, "_unet", None) is not None:
            unet = resolve_if_lazy(getattr(node, "_unet"))
        if "prompt_encoder" in bt:
            te = getattr(node, "_text_encoder", None)
            if te is not None:
                text_encoder = resolve_if_lazy(te)
            te2 = getattr(node, "_text_encoder_2", None)
            if te2 is not None:
                text_encoder_2 = resolve_if_lazy(te2)

    if unet is None:
        raise RuntimeError("LoRA: no UNet found on graph.")
    if family == "sdxl" and (text_encoder is None or text_encoder_2 is None):
        raise RuntimeError("LoRA: SDXL requires text_encoder and text_encoder_2 on graph.")
    if family == "sd15" and text_encoder is None:
        raise RuntimeError("LoRA: SD1.5 requires text_encoder on graph.")

    if family == "sdxl":
        from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin

        class _Pipe(StableDiffusionXLLoraLoaderMixin):
            def __init__(self, unet, text_encoder, text_encoder_2):
                self.unet = unet
                self.text_encoder = text_encoder
                self.text_encoder_2 = text_encoder_2
                # LoraBaseMixin exposes `lora_scale` as a read-only property.
                self._lora_scale = 1.0
                # Some loader utilities inspect `_pipeline.components` to manage hooks/offload.
                self.components = {
                    "unet": unet,
                    "text_encoder": text_encoder,
                    "text_encoder_2": text_encoder_2,
                }
                # Offload bookkeeping expected by diffusers loader helpers.
                self.hf_device_map = None

        return _Pipe(unet, text_encoder, text_encoder_2)

    from diffusers.loaders.lora_pipeline import StableDiffusionLoraLoaderMixin

    class _Pipe(StableDiffusionLoraLoaderMixin):
        def __init__(self, unet, text_encoder):
            self.unet = unet
            self.text_encoder = text_encoder
            self._lora_scale = 1.0
            self.components = {"unet": unet, "text_encoder": text_encoder}
            self.hf_device_map = None

    return _Pipe(unet, text_encoder)


def _sync_ip_adapter_plus_token_embed_dims_from_unet(graph: Any, unet: Any) -> None:
    """Sync IP-Adapter projection dims from the matching UNet layers.

    - For IP-Adapter Plus / Plus-Face: set ``ip_adapter_plus_token_embed_dim`` so the node can align
      ViT hidden states to ``proj_in.in_features``.
    - For pooled / FaceID: set the node's fallback ``ip_adapter_embed_dim`` used by
      :meth:`yggdrasill.integrations.diffusers.adapters.ip_adapter.IPAdapterNode._inactive_image_embeds`
      when there is no image encoder (e.g. FaceID precomputed embeddings).
    """
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    unet = resolve_if_lazy(unet)
    if unet is None:
        return
    wrap = getattr(unet, "encoder_hid_proj", None)
    layers = getattr(wrap, "image_projection_layers", None) if wrap is not None else None
    if not layers:
        return
    key_plus = C.CFG_IP_ADAPTER_PLUS_TOKEN_EMBED_DIM
    key_pooled = "ip_adapter_embed_dim"
    order = list(graph.metadata.get(_META_IP_ADAPTER_ORDER) or [])
    if len(order) != len(layers):
        ip_sorted = [
            n
            for n in sorted(graph.node_ids)
            if (graph.get_node(n) is not None)
            and getattr(graph.get_node(n), "block_type", "") == "adapter/ip_adapter"
        ]
        if len(ip_sorted) == len(layers):
            order = ip_sorted
    for idx, layer in enumerate(layers):
        nid = order[idx] if idx < len(order) else None
        if nid is None:
            continue
        node = graph.get_node(nid)
        if node is None or getattr(node, "block_type", "") != "adapter/ip_adapter":
            continue

        if not hasattr(node, "_config"):
            node._config = {}

        # Plus-Face/Plus: projection expects token embeddings of this width.
        pin = getattr(layer, "proj_in", None)
        if pin is not None and hasattr(pin, "in_features"):
            plus_dim = int(getattr(pin, "in_features", 0) or 0)
            if plus_dim > 0:
                node._config[key_plus] = plus_dim

        # Pooled / FaceID: even when image encoder is absent, UNet still uses the image projection
        # path, so we need inactive zeros with matching input feature width.
        try:
            import torch.nn as nn

            pooled_dim: int | None = None
            for m in getattr(layer, "modules", lambda: [])():
                if isinstance(m, nn.Linear) and hasattr(m, "in_features"):
                    d = int(getattr(m, "in_features", 0) or 0)
                    if d > 0:
                        pooled_dim = d
                        break
            if pooled_dim is not None:
                node._config[key_pooled] = pooled_dim
        except Exception:
            # Best effort: don't fail graph building if module introspection fails.
            pass


def _load_ip_adapter_weights_into_graph(
    graph: Any,
    *,
    pretrained: str,
    subfolder: Optional[str] = None,
    weight_name: str = "ip-adapter_sd15.bin",
    ip_adapter_scale: Optional[float] = None,
    adapter_node_id: str,
) -> None:
    """Append one IP-Adapter checkpoint and reload **all** accumulated weights on the UNet (diffusers API).

    Diffusers ``_load_ip_adapter_weights`` always rebuilds processors for the full ``state_dicts`` list,
    so each new ``add_component(..., sdxl.ipadapter, ...)`` must merge with previous checkpoints.
    Scales are applied at run time via ``ip_adapter_conditioning_scale``; *ip_adapter_scale* here is
    only forwarded when this is the **first** adapter on the graph (single-adapter backward compat).
    """
    from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

    sd = _load_ip_adapter_state_dict(
        pretrained,
        subfolder=subfolder,
        weight_name=weight_name,
    )
    acc = graph.metadata.setdefault(_META_IP_ADAPTER_SDS, [])
    order = graph.metadata.setdefault(_META_IP_ADAPTER_ORDER, [])
    acc.append(sd)
    order.append(adapter_node_id)

    for nid in graph.node_ids:
        node = graph.get_node(nid)
        bt = getattr(node, "block_type", "") or ""
        if not (bt.endswith("/unet") or bt.endswith("/transformer")):
            continue
        unet = getattr(node, "_unet", None)
        if unet is None:
            continue
        resolved = resolve_if_lazy(unet)
        if resolved is not unet:
            node._unet = resolved
        reload_ip_adapter_weights_on_unet(resolved, acc, low_cpu_mem_usage=True)
        _sync_ip_adapter_plus_token_embed_dims_from_unet(graph, resolved)
        if ip_adapter_scale is not None and len(acc) == 1:
            from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import (
                _set_ip_adapter_scale_on_unet,
            )

            _set_ip_adapter_scale_on_unet(resolved, ip_adapter_scale)
        return
    import logging

    logging.getLogger(__name__).warning(
        "IP-Adapter weights not loaded: no UNet/transformer node in graph. "
        "Add sd15.unet (or sdxl.unet / flux.transformer) before sd15.ipadapter / sdxl.ipadapter."
    )


_IP_ADAPTER_MASK_PREP_NODE_ID = "ip_mask_prep"


def _graph_has_block_type(graph: Any, block_type: str) -> bool:
    for nid in graph.node_ids:
        node = graph.get_node(nid)
        if getattr(node, "block_type", "") == block_type:
            return True
    return False


def _find_backbone_node_id(graph: Any) -> Optional[str]:
    for nid in graph.node_ids:
        node = graph.get_node(nid)
        bt = getattr(node, "block_type", "") or ""
        if bt.endswith("/unet") or bt.endswith("/transformer"):
            return nid
    return None


def _ensure_ip_adapter_mask_prep(graph: Any) -> None:
    """Add ``ip_mask_prep`` → UNet when IP-Adapter is present; run.py pins ``None`` if no masks."""
    if _graph_has_block_type(graph, "common/ip_adapter_mask_prep"):
        return
    if not _graph_has_block_type(graph, "adapter/ip_adapter"):
        return
    unet_nid = _find_backbone_node_id(graph)
    if unet_nid is None:
        return
    if _IP_ADAPTER_MASK_PREP_NODE_ID in graph.node_ids:
        return
    from yggdrasill.foundation.registry import BlockRegistry
    from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect

    reg = BlockRegistry.global_registry()
    mask_node = reg.build({
        "type": "common/ip_adapter_mask_prep",
        "node_id": _IP_ADAPTER_MASK_PREP_NODE_ID,
        "config": {},
    })
    graph.add_node(_IP_ADAPTER_MASK_PREP_NODE_ID, mask_node)
    apply_port_name_auto_connect(graph, _IP_ADAPTER_MASK_PREP_NODE_ID, mask_node)
    getattr(graph, "metadata", {}).setdefault("ip_mask_prep_auto", True)


class DiffusionGraphBuilder:
    """Builds diffusion graphs by adding components with pretrained loading.

    Usage:
        builder = DiffusionGraphBuilder(Hypergraph())
        builder.add_component("unet", "sd15.unet", pretrained="runwayml/stable-diffusion-v1-5")
        builder.add_component("tokenizer", "sd15.tokenizer", ...)
        graph = builder.graph
    """

    def __init__(
        self,
        graph: Optional[Hypergraph] = None,
        *,
        graph_id: Optional[str] = None,
        name: Optional[str] = None,
        train_recipe: Optional[str] = None,
    ) -> None:
        if train_recipe is not None:
            if graph is not None:
                raise ValueError("train_recipe cannot be combined with an explicit graph")
            gid = name or graph_id or train_recipe
            self._graph = Hypergraph(graph_id=gid)
            self._graph.metadata["ygg_diffusion_train_recipe"] = train_recipe
            self._added_groups = {}
            self._completed = True
            return
        graph_id = name or graph_id or "diffusion_graph"
        self._graph = graph or Hypergraph(graph_id=graph_id)
        self._added_groups: Dict[str, str] = {}  # group -> node_id
        self._completed: bool = False

    @classmethod
    def from_template(
        cls,
        template_name: str,
        *,
        task: str = "inference",
        **kwargs: Any,
    ) -> "DiffusionGraphBuilder":
        """Wrap a graph from a template.

        *task* ``\"inference\"`` (default): same as :meth:`Hypergraph.from_template`, or a **named
        train recipe** (``\"sd15_lora_train\"``, ``\"sdxl_lora_train\"``, ``\"flux_lora_train\"``, …):
        placeholder builder; pass training kwargs to :meth:`run` (full LoRA train via
        :class:`~yggdrasill.integrations.diffusers.training.trainer.DiffusionLoRATrainer`).

        *task* ``\"train\"`` is not used for diffusion templates; use a named ``*_lora_train`` recipe
        or build a training graph with :meth:`build_lora_training_hypergraph` / ``engine.run(..., run_mode=\"train\")``.
        """
        key = template_name.strip().lower().replace("-", "_")
        from yggdrasill.integrations.diffusers.train_recipe_registry import is_named_train_recipe

        if is_named_train_recipe(key):
            if task != "inference":
                raise ValueError(
                    f"Named recipe {template_name!r} already implies training; "
                    "use default task='inference'."
                )
            allowed_recipe_kw = frozenset({"name", "graph_id"})
            bad = set(kwargs) - allowed_recipe_kw
            if bad:
                raise ValueError(
                    f"Named recipe {template_name!r} does not accept from_template kwargs {sorted(bad)}; "
                    "pass training arguments to run(). Allowed: name, graph_id."
                )
            return cls(train_recipe=key, name=kwargs.get("name"), graph_id=kwargs.get("graph_id"))
        if task == "train":
            raise ValueError(
                'diffusion from_template no longer supports task="train" with a graph template name; '
                "use a named recipe, e.g. from_template(\"sd15_lora_train\"), then run(...). "
                "For a custom training hypergraph, use DiffusionGraphBuilder.build_lora_training_hypergraph(...) "
                'or engine.run(..., run_mode="train").'
            )
        if task != "inference":
            raise ValueError(f"task must be 'inference' or 'train', got {task!r}")
        graph = Hypergraph.from_template(template_name, **kwargs)
        return cls(graph)

    @staticmethod
    def build_lora_training_hypergraph(*, objective: Any, **kwargs: Any) -> Hypergraph:
        """Level-2 training graph for diffusion LoRA (delegates to :func:`training_hypergraph.build_diffusion_lora_training_hypergraph`)."""
        from yggdrasill.integrations.diffusers.training.training_hypergraph import (
            build_diffusion_lora_training_hypergraph,
        )

        return build_diffusion_lora_training_hypergraph(objective=objective, **kwargs)

    def _apply_graph_device(self) -> None:
        """Move all nodes (and schedulers) to the graph's inferred device."""
        dev = _infer_hypergraph_device(self._graph)
        if dev is not None and hasattr(self._graph, "to"):
            self._graph.to(dev)

    @property
    def graph(self) -> Hypergraph:
        self._ensure_text2img_complete()
        return self._graph

    def to(self, device: Any) -> "DiffusionGraphBuilder":
        """Move the graph to the target device. Returns self for chaining."""
        self.graph.to(device)
        return self

    def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the diffusion graph or a named train recipe.

        **Inference:** prompt, negative_prompt, num_inference_steps, guidance_scale, seed, width, height,
        device, controlnet_image, ip_adapter_image, … — see :func:`yggdrasill.integrations.diffusers.run.run`.
        Returns :class:`~yggdrasill.integrations.diffusers.types.DiffusionOutput`.

        **Named train recipe** (from ``from_template(\"sd15_lora_train\")`` etc.): pass
        :class:`~yggdrasill.integrations.diffusers.training.config.TrainingConfig`-compatible kwargs
        (``data_dir``, ``output_path``, ``task``, ``pretrained`` / ``pretrained_model_name_or_path``, …).
        Returns :class:`~yggdrasill.integrations.diffusers.training.types.TrainResult`.
        """
        recipe = self._graph.metadata.get("ygg_diffusion_train_recipe")
        if recipe is not None:
            return self._run_named_train_recipe(inputs, **kwargs)
        from yggdrasill.integrations.diffusers.run import run as run_diffusion
        return run_diffusion(self.graph, inputs, wrap_output=True, **kwargs)

    def _run_named_train_recipe(
        self,
        inputs: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        from yggdrasill.integrations.diffusers.train_recipe_registry import TRAIN_RECIPE_SPECS
        from yggdrasill.integrations.diffusers.training.config import TrainingConfig
        from yggdrasill.integrations.diffusers.training.trainer import DiffusionLoRATrainer

        recipe = str(self._graph.metadata.get("ygg_diffusion_train_recipe") or "")
        if recipe not in TRAIN_RECIPE_SPECS:
            raise ValueError(f"Unknown train recipe metadata {recipe!r}")
        spec = TRAIN_RECIPE_SPECS[recipe]
        merged: Dict[str, Any] = {}
        if inputs:
            merged.update(inputs)
        merged.update(kwargs)

        allowed = set(TrainingConfig.__dataclass_fields__)
        cfg_kwargs: Dict[str, Any] = {"family": spec["family"]}
        default_pt = spec["default_pretrained"]
        for k, v in merged.items():
            if k == "pretrained":
                cfg_kwargs["pretrained_model_name_or_path"] = v
            elif k in allowed:
                cfg_kwargs[k] = v
        if "pretrained_model_name_or_path" not in cfg_kwargs:
            cfg_kwargs["pretrained_model_name_or_path"] = default_pt

        config = TrainingConfig(**cfg_kwargs)
        return DiffusionLoRATrainer(config).train()

    def _ensure_text2img_complete(self) -> None:
        """Add latent_init, expose I/O, and metadata if this is an incomplete text2img topology."""
        if self._completed:
            return
        has_tokenizer = has_prompt_enc = has_unet = has_vae_decode = False
        has_sched_setup = has_sched_step = has_latent_init = False
        family = ""
        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            bt = getattr(node, "block_type", "") or ""
            if "tokenizer" in bt:
                has_tokenizer = True
            if "prompt_encoder" in bt:
                has_prompt_enc = True
            if bt.endswith("/unet") or bt.endswith("/transformer"):
                has_unet = True
            if "vae_decode" in bt:
                has_vae_decode = True
            if "scheduler_setup" in bt:
                has_sched_setup = True
            if "scheduler_step" in bt:
                has_sched_step = True
            if "latent_init" in bt:
                has_latent_init = True
            # Backbone family for latent_init fallback — must not be overwritten by
            # ``adapter/controlnet`` or ``adapter/ip_adapter`` (last nodes in typical builds).
            if "/" in bt:
                root = bt.split("/")[0]
                if root != "adapter":
                    family = root

        if not (
            has_tokenizer
            and has_prompt_enc
            and has_unet
            and has_vae_decode
            and (has_sched_setup or has_sched_step)
            and not has_latent_init
        ):
            # Graph is already complete (e.g. from_template). Still expose adapter ports
            # (control_image, ip_adapter_image) if ControlNet/IPAdapter nodes were added.
            self.expose_default_io()
            self._completed = True
            return

        # SD1.5 manual stack: one graph for text2img / img2img / inpaint (optional image, mask).
        if family == "sd15":
            from yggdrasill.integrations.diffusers.sd15.universal import (
                try_complete_sd15_universal_diffusion,
            )

            if try_complete_sd15_universal_diffusion(self._graph):
                for nid in self._graph.node_ids:
                    node = self._graph.get_node(nid)
                    bt = getattr(node, "block_type", "") or ""
                    if "scheduler_setup" in bt and hasattr(node, "_config"):
                        node._config = node._config or {}
                        node._config.setdefault("device", "cuda")
                        node._config.setdefault("num_inference_steps", 50)
                self.expose_default_io()
                self._graph.metadata.setdefault("num_loop_steps", 50)
                self._completed = True
                return

        if family == "sdxl":
            from yggdrasill.integrations.diffusers.sdxl.universal import (
                try_complete_sdxl_universal_diffusion,
            )

            if try_complete_sdxl_universal_diffusion(self._graph):
                for nid in self._graph.node_ids:
                    node = self._graph.get_node(nid)
                    bt = getattr(node, "block_type", "") or ""
                    if "scheduler_setup" in bt and hasattr(node, "_config"):
                        node._config = node._config or {}
                        node._config.setdefault("device", "cuda")
                        node._config.setdefault("num_inference_steps", 50)
                        node._config.setdefault("height", 1024)
                        node._config.setdefault("width", 1024)
                self.expose_default_io()
                self._graph.metadata.setdefault("num_loop_steps", 50)
                self._completed = True
                return

        # Fallback: text2img-only completion (no image/mask path).
        latent_type = f"{family}.latent_init" if family else "sd15.latent_init"
        cfg: Dict[str, Any] = {
            "height": 512,
            "width": 512,
            "device": "cuda",
            "dtype": "float16",
        }
        if family == "sdxl":
            cfg["height"] = 1024
            cfg["width"] = 1024
        self.add_component("LatentInit", latent_type, config=cfg)

        # Update scheduler_setup config for device and num_inference_steps
        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            bt = getattr(node, "block_type", "") or ""
            if "scheduler_setup" in bt and hasattr(node, "_config"):
                node._config = node._config or {}
                node._config.setdefault("device", "cuda")
                node._config.setdefault("num_inference_steps", 50)

        self.expose_default_io()
        self._graph.metadata["num_loop_steps"] = 50
        self._completed = True

    def add_component(
        self,
        node_id: str,
        component_type: str,
        *,
        pretrained: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        store: Optional[Any] = None,
        variant: str = "",
        torch_dtype: Optional[Any] = None,
        subfolder: Optional[str] = None,
        weight_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "DiffusionGraphBuilder":
        """Add a component node (or update an existing grouped node).

        Args:
            node_id: Graph node id.
            component_type: E.g. "sd15.unet", "sd15.controlnet", "sd15.ipadapter".
            pretrained: HF repo id or local path for loading.
            config: Node config overrides.
            store: ModelStore instance (optional).
            variant: Model variant (e.g. "fp16").
            torch_dtype: Target dtype for loaded models.
            subfolder: Subfolder in repo (e.g. "models" for IP-Adapter).
            weight_name: Weight filename (e.g. "ip-adapter_sd15.bin"). Passed to config.
            **kwargs: Additional overrides merged into config.

        Returns:
            self for chaining.
        """
        if not is_component_type(component_type):
            raise ValueError(
                f"Expected component type (e.g. 'sd15.unet'), got '{component_type}'"
            )

        spec = resolve_component_type(component_type)
        family = component_type.split(".", 1)[0]
        load_family = getattr(spec, "load_family", None) or family
        family_spec = get_family_spec(family)

        cfg = dict(config or {})
        if subfolder is not None:
            cfg["subfolder"] = subfolder
        if weight_name is not None:
            cfg["weight_name"] = weight_name
        # Default IP-Adapter config when adding ipadapter with h94/IP-Adapter
        if (
            pretrained
            and "h94/IP-Adapter" in str(pretrained)
            and subfolder is None
            and weight_name is None
        ):
            if component_type == "sdxl.ipadapter":
                cfg.setdefault("subfolder", "sdxl_models")
                cfg.setdefault("weight_name", "ip-adapter_sdxl.bin")
            elif component_type in ("sd15.ipadapter_plus",):
                cfg.setdefault("subfolder", "models")
                cfg.setdefault("weight_name", "ip-adapter-plus_sd15.safetensors")
            elif component_type in ("sd15.ipadapter_plus_face",):
                cfg.setdefault("subfolder", "models")
                cfg.setdefault("weight_name", "ip-adapter-plus-face_sd15.safetensors")
            elif component_type in ("sd15.ipadapter", "adapter.ip_adapter"):
                cfg.setdefault("subfolder", "models")
                cfg.setdefault("weight_name", "ip-adapter_sd15.bin")
            elif component_type in ("sd15.ipadapter_faceid", "sdxl.ipadapter_faceid"):
                # FaceID checkpoints store weights in the repo root (subfolder=None) and
                # consume InsightFace embeddings of dim=512.
                cfg.setdefault(
                    "weight_name",
                    "ip-adapter-faceid_sdxl.bin"
                    if component_type == "sdxl.ipadapter_faceid"
                    else "ip-adapter-faceid_sd15.bin",
                )

        # Default IP-Adapter-FaceID config when adding FaceID weights explicitly.
        if (
            pretrained
            and "h94/IP-Adapter-FaceID" in str(pretrained)
            and subfolder is None
            and weight_name is None
        ):
            if component_type == "sdxl.ipadapter_faceid":
                cfg.setdefault("weight_name", "ip-adapter-faceid_sdxl.bin")
            elif component_type == "sd15.ipadapter_faceid":
                cfg.setdefault("weight_name", "ip-adapter-faceid_sd15.bin")
        cfg.update(kwargs)
        # LoRA weights are loaded at runtime via diffusers loader mixins.
        if component_type.endswith(".lora") and pretrained is not None:
            _lw_entry: Dict[str, Any] = {
                "name": str(node_id),
                "path": str(pretrained),
                "weight_name": cfg.get("weight_name"),
                "scale": float(cfg.get("scale", 1.0)),
            }
            if cfg.get("subfolder") is not None:
                _lw_entry["subfolder"] = cfg["subfolder"]
            cfg.setdefault("lora_weights", [_lw_entry])
        # IP-Adapter Plus / Plus-Face: UNet projection expects CLIP vision hidden states, not pooled
        # image_embeds (see diffusers SDXL prepare_ip_adapter_image_embeds / encode_image).
        wn = cfg.get("weight_name")
        if wn is not None and spec.block_types and any(
            "ip_adapter" in str(bt) for bt in spec.block_types
        ):
            wn_blob = " ".join(str(x).lower() for x in wn) if isinstance(wn, (list, tuple)) else str(wn).lower()
            if "plus" in wn_blob:
                cfg.setdefault("ip_adapter_use_hidden_states", True)
        if pretrained is not None:
            cfg.setdefault("pretrained", str(pretrained))
        if "controlnet" in component_type:
            if component_type.startswith("sdxl."):
                cfg.setdefault("width", 1024)
                cfg.setdefault("height", 1024)
            else:
                cfg.setdefault("width", 512)
                cfg.setdefault("height", 512)

        # Resolve torch_dtype: use family default when loading pretrained
        dtype_to_load = torch_dtype
        if dtype_to_load is None and pretrained and spec.load_keys:
            dtype_str = family_spec.torch_dtype_default
            dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
            dtype_to_load = dtype_map.get(dtype_str, torch.float16)

        # AnimateDiff: wrap existing SD1.5 UNet with MotionAdapter (diffusers UNetMotionModel).
        if component_type == "sd15.motionadapter":
            if not pretrained:
                raise ValueError(
                    "sd15.motionadapter requires pretrained= (Hugging Face repo id for MotionAdapter weights)."
                )
            if dtype_to_load is None:
                dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
                dtype_to_load = dtype_map.get(family_spec.torch_dtype_default, torch.float16)
            self._apply_sd15_motion_adapter(str(pretrained), cfg, dtype_to_load)
            self.expose_default_io()
            self._apply_graph_device()
            return self

        if component_type == "sdxl.motionadapter":
            if not pretrained:
                raise ValueError(
                    "sdxl.motionadapter requires pretrained= (Hugging Face repo id for MotionAdapter weights)."
                )
            if dtype_to_load is None:
                dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
                dtype_to_load = dtype_map.get(family_spec.torch_dtype_default, torch.float16)
            self._apply_sdxl_motion_adapter(str(pretrained), cfg, dtype_to_load)
            self.expose_default_io()
            self._apply_graph_device()
            return self

        # Load pretrained components if requested
        components_loaded: Dict[str, Any] = {}
        if pretrained and spec.load_keys:
            load_variant = variant if variant else ("fp16" if dtype_to_load == torch.float16 else "")
            pretrained_map = getattr(spec, "load_pretrained_map", None)
            subfolder_map = getattr(spec, "load_subfolder_map", None)
            variant_map = getattr(spec, "load_variant_map", None)
            # Allow per-call overrides for Diffusers model weights when the component
            # is itself a Diffusers model (e.g. T2I-Adapter single-file weights).
            is_t2i = any(str(bt) == "adapter/t2i_adapter" for bt in (spec.block_types or ()))
            if is_t2i and cfg.get("subfolder"):
                subfolder_map = dict(subfolder_map or {})
                for k in spec.load_keys:
                    subfolder_map.setdefault(k, cfg.get("subfolder"))
            extra_kwargs_map: Dict[str, Dict[str, Any]] = {}
            if is_t2i and cfg.get("weight_name"):
                for k in spec.load_keys:
                    if k == "t2iadapter":
                        extra_kwargs_map.setdefault(k, {})["weight_name"] = cfg.get("weight_name")
            components_loaded = load_components_from_pretrained(
                spec.load_keys,
                pretrained,
                family=load_family,
                store=store,
                torch_dtype=dtype_to_load,
                variant=load_variant,
                pretrained_map=pretrained_map,
                subfolder_map=subfolder_map,
                variant_map=variant_map,
                extra_kwargs_map=extra_kwargs_map or None,
            )

        # Build constructor kwargs from constructor_map + loaded components
        from yggdrasill.foundation.registry import BlockRegistry
        reg = BlockRegistry.global_registry()

        for block_type in spec.block_types:
            const_map = spec.constructor_map.get(block_type, {})
            kwargs: Dict[str, Any] = {}
            for ctor_kwarg, load_key in const_map.items():
                val = components_loaded.get(load_key)
                if val is not None:
                    kwargs[ctor_kwarg] = val

            if config:
                kwargs["config"] = dict(cfg)

            if spec.group and spec.group in self._added_groups:
                # Update existing grouped node (e.g. prompt_encoder with text_encoder_2)
                existing_id = self._added_groups[spec.group]
                existing_node = self._graph._nodes.get(existing_id)
                if existing_node is not None and hasattr(existing_node, "update_from_components"):
                    existing_node.update_from_components(kwargs)
                    self._apply_graph_device()
                    return self

            if len(spec.block_types) == 1:
                nid = node_id
            else:
                # e.g. scheduler_setup -> _setup, scheduler_step -> _step
                suffix = "_" + block_type.split("/")[-1].split("_", 1)[-1]
                nid = f"{node_id}{suffix}"
            build_cfg: Dict[str, Any] = {
                "block_type": block_type,
                "node_id": nid,
                "config": kwargs.get("config", cfg),
            }
            if block_type == "adapter/lora_loader":
                build_cfg["pipe"] = _build_lora_pipe(self._graph, family=family)
            for k, v in kwargs.items():
                if k not in ("config", "type", "block_type", "node_id"):
                    build_cfg[k] = v

            node = reg.build(build_cfg)
            self._graph.add_node(nid, node)
            if block_type == "adapter/lora_loader":
                setattr(node, "_ygg_graph", self._graph)
            if spec.group:
                self._added_groups[spec.group] = nid
            from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect
            apply_port_name_auto_connect(self._graph, nid, node)

        # IP-Adapter: load weights into UNet when pretrained + weight config are provided
        if (
            "ip_adapter" in str(spec.block_types)
            and pretrained
            and (cfg.get("subfolder") or cfg.get("weight_name"))
        ):
            _load_ip_adapter_weights_into_graph(
                self._graph,
                pretrained=pretrained,
                # If subfolder wasn't explicitly provided (or was passed as None),
                # prefer looking in repo root (no /models/). Loader may fall back.
                subfolder=cfg.get("subfolder"),
                weight_name=cfg.get("weight_name", "ip-adapter_sd15.bin"),
                ip_adapter_scale=cfg.get(C.CFG_IP_ADAPTER_SCALE),
                adapter_node_id=nid,
            )

        if "ip_adapter" in str(spec.block_types):
            _ensure_ip_adapter_mask_prep(self._graph)

        # After first .graph / .run, _ensure_text2img_complete() sets _completed and will
        # not call expose_default_io again; new ControlNet / IP-Adapter nodes must expose
        # control_image / ip_adapter_image or run-time dicts never reach EdgeBuffers.
        self.expose_default_io()
        self._apply_graph_device()
        return self

    def _find_unet_node(self) -> Optional[Any]:
        """Find the Backbone (UNet/transformer) node in the graph."""
        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            bt = getattr(node, "block_type", "") or ""
            if bt.endswith("/unet") or bt.endswith("/transformer"):
                return node
        return None

    def _apply_sd15_motion_adapter(
        self,
        pretrained_repo: str,
        cfg: Dict[str, Any],
        torch_dtype: Any,
    ) -> None:
        """Load MotionAdapter and replace ``sd15/unet`` node's module with ``UNetMotionModel``."""
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        try:
            from diffusers.models import MotionAdapter, UNetMotionModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AnimateDiff requires diffusers with MotionAdapter / UNetMotionModel. "
                "Install: pip install 'diffusers>=0.28'"
            ) from exc

        node = self._find_unet_node()
        if node is None:
            raise RuntimeError(
                "sd15.motionadapter requires a graph that already has an sd15/unet node "
                "(e.g. DiffusionGraphBuilder.from_template('sd15_text2img', ...))."
            )
        bt = getattr(node, "block_type", "") or ""
        if bt != "sd15/unet":
            raise RuntimeError(
                f"sd15.motionadapter only supports sd15/unet backbone; found {bt!r}."
            )

        base = resolve_if_lazy(getattr(node, "_unet", None))
        if base is None:
            raise RuntimeError("UNet node has no weights; add sd15.unet (or complete template) before motionadapter.")

        if isinstance(base, UNetMotionModel):
            import warnings

            warnings.warn(
                "UNet is already a UNetMotionModel; skipping second sd15.motionadapter wrap.",
                stacklevel=2,
            )
        else:
            ma = MotionAdapter.from_pretrained(pretrained_repo, torch_dtype=torch_dtype)
            wrapped = UNetMotionModel.from_unet2d(base, ma, load_weights=True)
            p = next(wrapped.parameters(), None)
            if p is not None:
                dev = p.device
                dt = p.dtype
                wrapped = wrapped.to(device=dev, dtype=dt)
            node._unet = wrapped

        num_frames = int(cfg.get(C.CFG_NUM_FRAMES, cfg.get("num_frames", 16)))
        decode_chunk = int(cfg.get(C.CFG_DECODE_CHUNK_SIZE, cfg.get("decode_chunk_size", 16)))
        meta = self._graph.metadata
        slot = meta.setdefault("animatediff", {})
        if not isinstance(slot, dict):
            slot = {}
            meta["animatediff"] = slot
        slot["num_frames"] = num_frames
        slot["motion_adapter_pretrained"] = pretrained_repo

        for nid in self._graph.node_ids:
            n = self._graph.get_node(nid)
            nbt = getattr(n, "block_type", "") or ""
            if "sd15/latent_init" in nbt:
                if not hasattr(n, "_config"):
                    n._config = {}
                n._config[C.CFG_NUM_FRAMES] = num_frames
            if "sd15/vae_decode" in nbt:
                if not hasattr(n, "_config"):
                    n._config = {}
                n._config[C.CFG_NUM_FRAMES] = num_frames
                n._config[C.CFG_DECODE_CHUNK_SIZE] = decode_chunk
                n._config[C.CFG_ANIMATEDIFF_VIDEO_DECODE] = True
            if "sd15/vae_encode" in nbt:
                if not hasattr(n, "_config"):
                    n._config = {}
                n._config[C.CFG_NUM_FRAMES] = num_frames

    def _apply_sdxl_motion_adapter(
        self,
        pretrained_repo: str,
        cfg: Dict[str, Any],
        torch_dtype: Any,
    ) -> None:
        """Load MotionAdapter and wrap ``sdxl/unet`` with ``UNetMotionModel`` (AnimateDiff SDXL)."""
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy

        try:
            from diffusers.models import MotionAdapter, UNetMotionModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AnimateDiff SDXL requires diffusers with MotionAdapter / UNetMotionModel."
            ) from exc

        node = self._find_unet_node()
        if node is None:
            raise RuntimeError(
                "sdxl.motionadapter requires a graph with an sdxl/unet node "
                "(e.g. DiffusionGraphBuilder.from_template('sdxl_text2img', ...))."
            )
        bt = getattr(node, "block_type", "") or ""
        if bt != "sdxl/unet":
            raise RuntimeError(
                f"sdxl.motionadapter only supports sdxl/unet backbone; found {bt!r}."
            )

        base = resolve_if_lazy(getattr(node, "_unet", None))
        if base is None:
            raise RuntimeError("UNet node has no weights; add sdxl.unet before motionadapter.")

        if isinstance(base, UNetMotionModel):
            import warnings

            warnings.warn(
                "UNet is already a UNetMotionModel; skipping second sdxl.motionadapter wrap.",
                stacklevel=2,
            )
        else:
            ma = MotionAdapter.from_pretrained(pretrained_repo, torch_dtype=torch_dtype)
            wrapped = UNetMotionModel.from_unet2d(base, ma, load_weights=True)
            p = next(wrapped.parameters(), None)
            if p is not None:
                wrapped = wrapped.to(device=p.device, dtype=p.dtype)
            node._unet = wrapped

        num_frames = int(cfg.get(C.CFG_NUM_FRAMES, cfg.get("num_frames", 16)))
        decode_chunk = int(cfg.get(C.CFG_DECODE_CHUNK_SIZE, cfg.get("decode_chunk_size", 16)))
        meta = self._graph.metadata
        slot = meta.setdefault("animatediff", {})
        if not isinstance(slot, dict):
            slot = {}
            meta["animatediff"] = slot
        slot["num_frames"] = num_frames
        slot["motion_adapter_pretrained"] = pretrained_repo
        slot["family"] = "sdxl"

        for nid in self._graph.node_ids:
            n = self._graph.get_node(nid)
            nbt = getattr(n, "block_type", "") or ""
            if "sdxl/latent_init" in nbt:
                if not hasattr(n, "_config"):
                    n._config = {}
                n._config[C.CFG_NUM_FRAMES] = num_frames
            if "sdxl/vae_decode" in nbt:
                if not hasattr(n, "_config"):
                    n._config = {}
                n._config[C.CFG_NUM_FRAMES] = num_frames
                n._config[C.CFG_DECODE_CHUNK_SIZE] = decode_chunk
                n._config[C.CFG_ANIMATEDIFF_VIDEO_DECODE] = True
            if "sdxl/vae_encode" in nbt:
                if not hasattr(n, "_config"):
                    n._config = {}
                n._config[C.CFG_NUM_FRAMES] = num_frames

    def add_node(self, node_id: str, node: Any) -> "DiffusionGraphBuilder":
        """Add a pre-built node directly. Auto-connects by port names."""
        from yggdrasill.hypergraph.auto_connect import apply_port_name_auto_connect
        self._graph.add_node(node_id, node)
        apply_port_name_auto_connect(self._graph, node_id, node)
        self.expose_default_io()
        self._apply_graph_device()
        return self

    def add_edge(self, source: str, source_port: str, target: str, target_port: str) -> "DiffusionGraphBuilder":
        """Add an edge. Uses contract port names when passed as strings."""
        self._graph.add_edge(Edge(source, source_port, target, target_port))
        return self

    def expose_default_io(self) -> "DiffusionGraphBuilder":
        """Expose standard diffusion inputs/outputs for text2img topologies.

        Finds tokenizer-like nodes and exposes prompt/negative_prompt;
        finds vae_decode nodes and exposes decoded_image as output_image.
        Safe to call multiple times; skips already-exposed ports.
        """
        from yggdrasill.foundation.node import AbstractGraphNode

        lora_loader_nids: List[str] = []
        for ln in self._graph.node_ids:
            n = self._graph.get_node(ln)
            if n is not None and getattr(n, "block_type", None) == "adapter/lora_loader":
                lora_loader_nids.append(ln)
        multi_lora = len(lora_loader_nids) > 1

        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            if not isinstance(node, AbstractGraphNode):
                continue
            bt = getattr(node, "block_type", "") or ""
            in_names = {p.name for p in node.get_input_ports()}
            out_names = {p.name for p in node.get_output_ports()}

            if "tokenizer" in bt:
                if C.PORT_PROMPT in in_names:
                    self._graph.expose_input(nid, C.PORT_PROMPT, C.PORT_PROMPT)
                if C.PORT_NEGATIVE_PROMPT in in_names:
                    self._graph.expose_input(
                        nid, C.PORT_NEGATIVE_PROMPT, C.PORT_NEGATIVE_PROMPT
                    )
            if "vae_decode" in bt and C.PORT_DECODED_IMAGE in out_names:
                self._graph.expose_output(nid, C.PORT_DECODED_IMAGE, C.PORT_OUTPUT_IMAGE)
            if "adapter/ip_adapter" in bt and C.PORT_IP_ADAPTER_IMAGE in in_names:
                # Use node-scoped key so multi-IPAdapter graphs work with ip_adapter_image={node_id: img}
                self._graph.expose_input(nid, C.PORT_IP_ADAPTER_IMAGE, f"{nid}:{C.PORT_IP_ADAPTER_IMAGE}")
            if "adapter/ip_adapter" in bt and C.PORT_IP_ADAPTER_IMAGE_EMBEDS in in_names:
                self._graph.expose_input(
                    nid,
                    C.PORT_IP_ADAPTER_IMAGE_EMBEDS,
                    f"{nid}:{C.PORT_IP_ADAPTER_IMAGE_EMBEDS}",
                )
            if "adapter/controlnet" in bt and C.PORT_CONTROL_IMAGE in in_names:
                # Use node-scoped key so multi-ControlNet graphs work with controlnet_image={node_id: img}
                self._graph.expose_input(nid, C.PORT_CONTROL_IMAGE, f"{nid}:{C.PORT_CONTROL_IMAGE}")
            if "adapter/t2i_adapter" in bt and C.PORT_T2I_ADAPTER_IMAGE in in_names:
                # Use node-scoped key so multi-adapter graphs work with t2i_adapter_image={node_id: img}
                self._graph.expose_input(
                    nid, C.PORT_T2I_ADAPTER_IMAGE, f"{nid}:{C.PORT_T2I_ADAPTER_IMAGE}"
                )
            if "adapter/lora_loader" in bt and C.PORT_LORA_SCALE in in_names:
                # One LoRA: `builder.run(lora_conditioning_scale=0.7)`. Several: per-node keys like
                # `LoRA1:lora_scale` (same pattern as ControlNet) or a dict on `lora_conditioning_scale`.
                ext = f"{nid}:{C.PORT_LORA_SCALE}" if multi_lora else "lora_conditioning_scale"
                self._graph.expose_input(nid, C.PORT_LORA_SCALE, ext)
            if "ip_adapter_mask_prep" in bt and C.PORT_IP_ADAPTER_MASK_IMAGES in in_names:
                self._graph.expose_input(
                    nid, C.PORT_IP_ADAPTER_MASK_IMAGES, C.PORT_IP_ADAPTER_MASK_IMAGES,
                )

        return self

    def _resolve_scheduler_base_id(self) -> Optional[str]:
        """Return base id (e.g. ``sched``) for ``sched_setup`` / ``sched_step`` pair."""
        for nid in sorted(self._graph.node_ids):
            node = self._graph.get_node(nid)
            if node is None:
                continue
            bt = getattr(node, "block_type", "") or ""
            if "scheduler_setup" in bt and nid.endswith("_setup"):
                return nid[: -len("_setup")]
        return None

    def _resolve_role_to_node_id(self, role_or_id: str) -> str:
        """Resolve canonical role name (e.g. 'Backbone') to actual graph node id."""
        if role_or_id in ("Scheduler", "scheduler"):
            base = self._resolve_scheduler_base_id()
            if base is not None:
                return base
        role_map = {
            "Backbone": ("unet", "transformer"),
            "Conjector": ("prompt_encoder",),
        }
        aliases = role_map.get(role_or_id)
        if aliases is None:
            return role_or_id
        for nid in self._graph.node_ids:
            node = self._graph.get_node(nid)
            bt = getattr(node, "block_type", "") or ""
            if any(a in bt for a in aliases):
                return nid
        return role_or_id

    def replace_component(
        self,
        node_id: str,
        component_type: str,
        *,
        pretrained: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        store: Optional[Any] = None,
        variant: str = "",
        torch_dtype: Optional[Any] = None,
        use_safetensors: Optional[bool] = None,
        **kwargs: Any,
    ) -> "DiffusionGraphBuilder":
        """Replace an existing node with a new component from *component_type*.

        For single-block components (unet, tokenizer, vae_decode), replaces
        the given node_id. For multi-block (e.g. scheduler), pass the base
        id (e.g. "sched") to replace both setup and step nodes.

        Supports canonical role names: ``Backbone`` → unet/transformer node,
        ``Conjector`` → prompt_encoder node, ``Scheduler`` → scheduler pair
        (``sched_setup`` / ``sched_step``).

        Args:
            node_id: Graph node id to replace (or role name: Backbone, Conjector, Scheduler).
            component_type: E.g. "sd15.unet", "sd15.scheduler".
        pretrained: HF repo id or local path for loading.
        config: Node config overrides.
        store: ModelStore instance (optional).
        variant: Model variant (e.g. "fp16").
        torch_dtype: Target dtype for loaded models.
        use_safetensors: If False, load .bin instead of .safetensors (needed for
            repos like Lykon/DreamShaper that have only diffusion_pytorch_model.bin).
        **kwargs: Merged into node config; use ``scheduler_type="euler"`` (etc.) to swap
            the Diffusers scheduler class without reloading the whole repo (SD/SDXL/FLUX).
            Note: ``stabilityai/stable-diffusion-xl-base-1.0`` already ships an Euler scheduler;
            replacing with ``scheduler_type="euler"`` only re-instantiates Euler (no visual change).
            Try ``"dpm_solver"``, ``"ddim"``, ``"unipc"``, etc. to see a different sampler.

        Returns:
            self for chaining.
        """
        cfg = dict(config or {})
        cfg.update(kwargs)
        scheduler_type = cfg.pop("scheduler_type", None)

        node_id = self._resolve_role_to_node_id(node_id)
        if not is_component_type(component_type):
            raise ValueError(
                f"Expected component type (e.g. 'sd15.unet'), got '{component_type}'"
            )

        spec = resolve_component_type(component_type)
        family = component_type.split(".", 1)[0]
        load_family = getattr(spec, "load_family", None) or family

        components_loaded: Dict[str, Any] = {}
        if pretrained and spec.load_keys:
            components_loaded = load_components_from_pretrained(
                spec.load_keys,
                pretrained,
                family=load_family,
                store=store,
                torch_dtype=torch_dtype,
                variant=variant if variant else "",
                use_safetensors=use_safetensors,
            )
            missing = [k for k in spec.load_keys if components_loaded.get(k) is None]
            if missing:
                raise RuntimeError(
                    f"Failed to load {missing} from {pretrained}. "
                    "Check repo structure (unet/, diffusion_pytorch_model.safetensors or .fp16.safetensors or .bin)."
                )

        if scheduler_type and spec.load_keys and "scheduler" in spec.load_keys:
            template = components_loaded.get("scheduler")
            if template is None:
                setup_nid = f"{node_id}_setup"
                old_setup = self._graph.get_node(setup_nid)
                template = getattr(old_setup, "_scheduler", None) if old_setup else None
            components_loaded["scheduler"] = _instantiate_scheduler_by_type(
                scheduler_type, template, family
            )
        elif (
            spec.load_keys == ["scheduler"]
            and not components_loaded
            and not scheduler_type
        ):
            raise RuntimeError(
                f"replace_component({component_type!r}) needs pretrained=... "
                f"and/or scheduler_type=... (e.g. scheduler_type='euler')."
            )

        from yggdrasill.foundation.registry import BlockRegistry
        reg = BlockRegistry.global_registry()

        replaced_ids: list[str] = []

        for block_type in spec.block_types:
            const_map = spec.constructor_map.get(block_type, {})
            ctor_kwargs: Dict[str, Any] = {}
            for ctor_kwarg, load_key in const_map.items():
                val = components_loaded.get(load_key)
                if val is not None:
                    ctor_kwargs[ctor_kwarg] = val

            if len(spec.block_types) == 1:
                nid = node_id
            else:
                suffix = "_" + block_type.split("/")[-1].split("_", 1)[-1]
                nid = f"{node_id}{suffix}"

            if nid not in self._graph.node_ids:
                continue

            # Preserve node config from the replaced node (device, num_inference_steps,
            # denoising_*, etc.). Otherwise replace_component(..., scheduler_type=...) alone
            # wipes _config and scheduler defaults to cpu / wrong schedule vs latents.
            old_node = self._graph.get_node(nid)
            merged_cfg = dict(cfg)
            if old_node is not None:
                prev = getattr(old_node, "_config", None) or {}
                merged_cfg = {**dict(prev), **merged_cfg}
            ctor_kwargs["config"] = merged_cfg

            build_cfg: Dict[str, Any] = {
                "block_type": block_type,
                "node_id": nid,
                "config": ctor_kwargs.get("config", merged_cfg),
            }
            for k, v in ctor_kwargs.items():
                if k not in ("config", "block_type", "node_id"):
                    build_cfg[k] = v

            new_node = reg.build(build_cfg)
            self._graph.replace_node(nid, node=new_node)
            replaced_ids.append(nid)

        if len(spec.block_types) > 1 and not replaced_ids:
            expected = [
                f"{node_id}_" + bt.split("/")[-1].split("_", 1)[-1]
                for bt in spec.block_types
            ]
            raise ValueError(
                f"replace_component: no scheduler nodes matched base id {node_id!r}. "
                f"Expected graph node ids like {expected!r} (from add_component(\"sched\", ...) "
                f"→ sched_setup / sched_step), or pass that base explicitly. "
                f"Role name 'Scheduler' only resolves if a *scheduler_setup* node id ends with '_setup'."
            )
        if len(spec.block_types) > 1 and len(replaced_ids) != len(spec.block_types):
            raise ValueError(
                f"replace_component: partial scheduler replace — matched {replaced_ids!r}, "
                f"expected {len(spec.block_types)} nodes for {component_type!r}."
            )

        _meta = getattr(self._graph, "metadata", None) or {}
        if (
            component_type in ("sd15.unet", "sd15.backbone")
            and (
                self._graph.graph_id == "sd15_inpaint"
                or _meta.get("sd15_universal")
            )
        ):
            from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
            from yggdrasill.integrations.diffusers.presets.sd15 import (
                reconfigure_sd15_inpaint_for_unet_in_channels,
            )

            un_n = self._graph.get_node(node_id)
            inner = getattr(un_n, "_unet", None) if un_n is not None else None
            inner = resolve_if_lazy(inner) if inner is not None else None
            uc = getattr(inner, "config", None) if inner is not None else None
            in_ch = int(getattr(uc, "in_channels", 4)) if uc is not None else 4
            reconfigure_sd15_inpaint_for_unet_in_channels(self._graph, in_channels=in_ch)

        if (
            component_type in ("sdxl.unet", "sdxl.backbone")
            and (
                self._graph.graph_id == "sdxl_inpaint"
                or _meta.get("sdxl_universal")
            )
        ):
            from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
            from yggdrasill.integrations.diffusers.presets.sdxl import (
                reconfigure_sdxl_inpaint_for_unet_in_channels,
            )

            un_n = self._graph.get_node(node_id)
            inner = getattr(un_n, "_unet", None) if un_n is not None else None
            inner = resolve_if_lazy(inner) if inner is not None else None
            uc = getattr(inner, "config", None) if inner is not None else None
            in_ch = int(getattr(uc, "in_channels", 4)) if uc is not None else 4
            reconfigure_sdxl_inpaint_for_unet_in_channels(self._graph, in_channels=in_ch)

        if component_type in _BACKBONE_REPLACE_COMPONENT_TYPES and replaced_ids:
            _reclaim_cuda_after_backbone_replace()

        if "ip_adapter" in str(spec.block_types):
            _ensure_ip_adapter_mask_prep(self._graph)

        self.expose_default_io()
        self._apply_graph_device()
        return self
