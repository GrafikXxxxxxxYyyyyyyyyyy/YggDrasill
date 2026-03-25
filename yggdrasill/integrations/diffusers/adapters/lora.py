"""LoRA adapter support for SD1.5/SDXL UNet and text encoders."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractInjector


def _lora_loader_node_ids_sorted(graph: Any) -> List[str]:
    out: List[str] = []
    for nid in sorted(getattr(graph, "node_ids", ()) or ()):
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        if getattr(node, "block_type", None) == "adapter/lora_loader":
            out.append(nid)
    return out


def _merged_lora_scale_for_node(merged: Dict[str, Any], node_id: str, n_lora_nodes: int) -> Optional[float]:
    key = f"{node_id}:{C.PORT_LORA_SCALE}"
    if key in merged and merged[key] is not None:
        return float(merged[key])
    if n_lora_nodes == 1 and C.PORT_LORA_SCALE in merged and merged[C.PORT_LORA_SCALE] is not None:
        return float(merged[C.PORT_LORA_SCALE])
    return None


def collect_all_lora_adapter_names_weights(graph: Any, merged: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """Build (names, weights) for diffusers ``set_adapters`` across every LoRA loader on the graph."""
    nids = _lora_loader_node_ids_sorted(graph)
    names: List[str] = []
    weights: List[float] = []
    for nid in nids:
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        cfg = getattr(node, "_config", None) or {}
        lora_weights = cfg.get("lora_weights") or []
        scale_override = _merged_lora_scale_for_node(merged, nid, len(nids))
        for i, lora in enumerate(lora_weights):
            name = lora.get("name", f"lora_{i}")
            w = scale_override if scale_override is not None else float(lora.get("scale", 1.0))
            names.append(name)
            weights.append(w)
    return names, weights


def _is_adapter_present_on_pipe(pipe: Any, name: str) -> bool:
    for mod_attr in ("unet", "text_encoder", "text_encoder_2"):
        mod = getattr(pipe, mod_attr, None)
        cfg = getattr(mod, "peft_config", None) if mod is not None else None
        if isinstance(cfg, dict) and name in cfg:
            return True
    return False


def _load_one_lora_entry(pipe: Any, lora: Dict[str, Any], index: int) -> None:
    name = lora.get("name", f"lora_{index}")
    if _is_adapter_present_on_pipe(pipe, name):
        return
    weight_path = lora.get("path", "")
    weight_name = lora.get("weight_name")
    subfolder = lora.get("subfolder")
    kwargs: Dict[str, Any] = {"adapter_name": name}
    if weight_name:
        kwargs["weight_name"] = weight_name
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    try:
        pipe.load_lora_weights(weight_path, **kwargs)
    except ValueError as e:
        msg = str(e).lower()
        if "peft backend is required" in msg:
            raise RuntimeError(
                "Diffusers LoRA loading requires the `peft` package, but it is not installed in this "
                "environment. Install it (recommended via the project extra): "
                "`pip install -e .[diffusion]` or `pip install peft>=0.13.1`."
            ) from e
        if "adapter name" in msg and "already in use" in msg:
            return
        raise


def ensure_all_graph_lora_weights_loaded(pipe: Any, graph: Any) -> None:
    """Load every LoRA declared on the graph so ``set_adapters`` is valid regardless of node run order."""
    for nid in _lora_loader_node_ids_sorted(graph):
        node = graph.get_node(nid) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        cfg = getattr(node, "_config", None) or {}
        for i, lora in enumerate(cfg.get("lora_weights") or []):
            _load_one_lora_entry(pipe, lora, i)


class LoRAInjectorNode(AbstractInjector):
    """Injector: loads and applies LoRA weights to backbone modules.

    This is a runtime overlay node: it mutates model weights in-place
    rather than producing output tensors through an edge. It runs once
    during graph setup, before the denoising loop.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        pipe: Any = None,
    ) -> None:
        cfg = dict(config or {})
        if pipe is None and "pipe" in cfg:
            pipe = cfg.pop("pipe")
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)
        self._pipe = pipe
        self._loaded = False

    @property
    def block_type(self) -> str:
        return "adapter/lora_loader"

    def declare_ports(self) -> List[Port]:
        return [
            Port("trigger", PortDirection.IN, PortType.ANY, optional=True),
            Port(C.PORT_LORA_SCALE, PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._pipe is None:
            raise RuntimeError(
                f"{type(self).__name__}(node_id={self._node_id!r}): pipe is None. "
                "This node must be constructed by DiffusionGraphBuilder so it can "
                "bind UNet/text encoders for LoRA loading."
            )
        lora_weights = self._config.get("lora_weights", [])
        adapter_names: List[str] = []
        adapter_weights: List[float] = []
        for i, lora in enumerate(lora_weights):
            name = lora.get("name", f"lora_{i}")
            scale = float(lora.get("scale", 1.0))
            adapter_names.append(name)
            adapter_weights.append(scale)

        graph = getattr(self, "_ygg_graph", None)
        merged: Dict[str, Any] = {}
        if graph is not None:
            merged = dict(getattr(graph, "_yggdrasill_lora_merged", None) or {})
            ls_in = inputs.get(C.PORT_LORA_SCALE)
            if ls_in is not None:
                merged[f"{self._node_id}:{C.PORT_LORA_SCALE}"] = float(ls_in)

        out_weights = list(adapter_weights)
        if graph is not None:
            ensure_all_graph_lora_weights_loaded(self._pipe, graph)
            all_names, all_weights = collect_all_lora_adapter_names_weights(graph, merged)
            if all_names:
                self._pipe.set_adapters(all_names, adapter_weights=all_weights)
                name_to_w = dict(zip(all_names, all_weights))
                out_weights = [name_to_w[n] for n in adapter_names if n in name_to_w]
        else:
            if adapter_names:
                self._pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            lora_scale = inputs.get(C.PORT_LORA_SCALE)
            if lora_scale is not None and len(adapter_names) == 1:
                self._pipe.set_adapters(adapter_names, adapter_weights=[float(lora_scale)])
                out_weights = [float(lora_scale)]

        self._loaded = True

        return {"result": {"loaded_loras": adapter_names, "weights": out_weights}}

    def unfuse(self) -> None:
        """Remove fused LoRA weights from the pipeline."""
        if self._pipe is not None and hasattr(self._pipe, "unfuse_lora"):
            self._pipe.unfuse_lora()
            self._pipe.unload_lora_weights()


