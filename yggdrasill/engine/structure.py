from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from yggdrasill.engine.edge import Edge
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import PortDirection

_instance_counter = itertools.count()


def _make_progress_callback() -> Any:
    """Return a callback that shows tqdm progress for denoising cycles."""
    pbar_ref: List[Any] = []

    def _cb(phase: str, info: Dict[str, Any]) -> None:
        if phase == "loop_start":
            try:
                from tqdm import tqdm
                total = info.get("steps", 1)
                pbar_ref.append(tqdm(total=total, desc="Generating", unit="step"))
            except ImportError:
                pass
        elif phase == "cycle_step" and pbar_ref:
            pbar_ref[0].update(1)
        elif phase == "loop_end" and pbar_ref:
            pbar_ref[0].close()
            pbar_ref.clear()

    return _cb


def _resolve_config_ref(
    config: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """If *config* is ``{"ref": "path/to/file"}`` load from that file.

    Path traversal (``..``) is rejected for security. When *base_dir* is set,
    ref must resolve inside it; otherwise any path (relative or absolute) is
    allowed.
    """
    if set(config.keys()) == {"ref"}:
        ref_path = Path(config["ref"])
        if ".." in ref_path.parts:
            raise ValueError(
                f"Config ref must not contain '..' path traversal: {ref_path}"
            )
        if base_dir is not None:
            resolved = (base_dir / ref_path).resolve()
            base_resolved = base_dir.resolve()
            if not str(resolved).startswith(str(base_resolved)):
                raise ValueError(
                    f"Config ref resolves outside base dir: {ref_path}"
                )
            ref_path = resolved
        elif not ref_path.is_absolute():
            ref_path = ref_path.resolve()
        if ref_path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
                with open(ref_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                raise ImportError(
                    f"PyYAML required to load YAML config ref: {ref_path}"
                )
        if ref_path.suffix not in (".json",):
            raise ValueError(
                f"Unsupported config ref file extension '{ref_path.suffix}'; "
                f"expected .json, .yaml, or .yml"
            )
        with open(ref_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return config


class Hypergraph:
    """Stores nodes, edges, and exposed inputs/outputs.

    Serves as the structural backbone that the engine (validator, planner,
    executor) operates on.  At the task-hypergraph level, nodes are task-node
    objects (Block+Node).  At the workflow level a separate Workflow class
    wraps the same protocol with Hypergraph instances as "nodes".
    """

    def __init__(
        self,
        graph_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        self._instance_id = next(_instance_counter)
        self._graph_id = name or graph_id or "graph"
        self._graph_kind: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

        self._nodes: Dict[str, Any] = {}
        self._edges: List[Edge] = []
        self._in_edges: Dict[str, List[Edge]] = {}
        self._out_edges: Dict[str, List[Edge]] = {}

        self._exposed_inputs: List[Dict[str, Any]] = []
        self._exposed_outputs: List[Dict[str, Any]] = []

        self._execution_version: int = 0
        self._node_trainable: Dict[str, bool] = {}

    # --- identity --------------------------------------------------------

    @property
    def graph_id(self) -> str:
        return self._graph_id

    @property
    def graph_kind(self) -> Optional[str]:
        return self._graph_kind

    @graph_kind.setter
    def graph_kind(self, value: Optional[str]) -> None:
        self._graph_kind = value

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = dict(value)

    @property
    def execution_version(self) -> int:
        return self._execution_version

    # --- nodes -----------------------------------------------------------

    @property
    def node_ids(self) -> Set[str]:
        return set(self._nodes.keys())

    def get_node(self, node_id: str) -> Optional[Any]:
        return self._nodes.get(node_id)

    def add_node(
        self,
        node_id: str,
        node: Any = None,
        *,
        type: Optional[str] = None,  # noqa: A002 — shadows builtin intentionally
        pretrained: Optional[str] = None,
        auto_connect: bool = True,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Add a node to the graph.

        Three calling conventions are supported:

        1. **Raw node object** — ``graph.add_node("id", node_object)``
        2. **Block-level type** — ``graph.add_node("id", type="sdxl/unet", pretrained="repo")``
        3. **Component-level type** — ``graph.add_node("id", type="sdxl.unet", pretrained="repo")``

        When *auto_connect* is ``True`` (the default for typed calls),
        port-name-based auto-wiring is applied after the node is added.
        """
        if not node_id or not node_id.strip():
            raise ValueError("node_id must be non-empty")
        node_id = node_id.strip()

        if node is not None:
            self._add_raw_node(node_id, node)
            return

        if type is None:
            raise ValueError(
                "add_node requires either a node object as the second "
                "positional argument, or type= keyword argument"
            )

        from yggdrasill.diffusion.components import (
            is_block_type, is_component_type,
        )
        if is_block_type(type):
            self._add_block_type_node(
                node_id, type,
                pretrained=pretrained,
                auto_connect=auto_connect,
                config=config,
                **kwargs,
            )
        elif is_component_type(type):
            self._add_component_type_node(
                node_id, type,
                pretrained=pretrained,
                auto_connect=auto_connect,
                config=config,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unrecognised type format '{type}'. "
                "Use 'family/block' for block-level or 'family.component' "
                "for component-level types."
            )

    def _add_raw_node(self, node_id: str, node: Any) -> None:
        """Internal: add a pre-constructed node object."""
        self._nodes[node_id] = node
        self._in_edges.setdefault(node_id, [])
        self._out_edges.setdefault(node_id, [])
        self._execution_version += 1

    def _add_block_type_node(
        self,
        node_id: str,
        block_type: str,
        *,
        pretrained: Optional[str] = None,
        auto_connect: bool = True,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Build a node from a block-level type and add it."""
        from yggdrasill.foundation.registry import BlockRegistry

        reg = BlockRegistry.global_registry()
        build_cfg: Dict[str, Any] = {"block_type": block_type, "node_id": node_id}
        if config:
            build_cfg.update(config)
        build_cfg.update(kwargs)

        if pretrained is not None:
            loaded = self._load_pretrained_components(
                block_type, pretrained, **kwargs,
            )
            build_cfg.update(loaded)

        node = reg.build(build_cfg)
        self._add_raw_node(node_id, node)
        if auto_connect:
            self._auto_connect_port_names(node_id, node)

    def _add_component_type_node(
        self,
        node_id: str,
        component_type: str,
        *,
        pretrained: Optional[str] = None,
        auto_connect: bool = True,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Resolve a component-level type and create the corresponding node(s)."""
        from yggdrasill.diffusion.components import resolve_component_type
        from yggdrasill.foundation.registry import BlockRegistry

        spec = resolve_component_type(component_type)
        reg = BlockRegistry.global_registry()

        loaded: Dict[str, Any] = {}
        if pretrained and spec.load_keys:
            family = component_type.split(".", 1)[0]
            loaded = self._load_pretrained_components_by_keys(
                spec.load_keys, pretrained, family=family, **kwargs,
            )

        if spec.group:
            self._handle_grouped_component(
                node_id, spec, loaded,
                auto_connect=auto_connect,
                config=config,
                registry=reg,
                **kwargs,
            )
            self._ensure_implicit_component_nodes(
                component_type,
                pretrained=pretrained,
                config=config,
                **kwargs,
            )
            return

        _block_defaults: Dict[str, Dict[str, Any]] = {
            "sd15/scheduler_setup": {"num_inference_steps": 50},
            "sd15/unet": {"guidance_scale": 7.5},
            "sd15/vae_decode": {"output_type": "pil"},
            "sd15/latent_init": {"dtype": "float32"},
        }
        if len(spec.block_types) == 1:
            bt = spec.block_types[0]
            ctor_map = spec.constructor_map.get(bt, {})
            build_cfg: Dict[str, Any] = {"block_type": bt, "node_id": node_id}
            for ctor_kwarg, comp_key in ctor_map.items():
                if comp_key in loaded:
                    build_cfg[ctor_kwarg] = loaded[comp_key]
            defaults = _block_defaults.get(bt, {})
            build_cfg.update(defaults)
            if config:
                build_cfg.update(config)

            node = reg.build(build_cfg)
            self._add_raw_node(node_id, node)
            if auto_connect:
                self._auto_connect_port_names(node_id, node)
        else:
            for i, bt in enumerate(spec.block_types):
                suffix = bt.rsplit("/", 1)[-1]
                sub_id = f"{node_id}_{suffix}"
                ctor_map = spec.constructor_map.get(bt, {})
                build_cfg = {"block_type": bt, "node_id": sub_id}
                for ctor_kwarg, comp_key in ctor_map.items():
                    if comp_key in loaded:
                        build_cfg[ctor_kwarg] = loaded[comp_key]
                defaults = _block_defaults.get(bt, {})
                build_cfg.update(defaults)
                if config:
                    build_cfg.update(config)

                node = reg.build(build_cfg)
                self._add_raw_node(sub_id, node)
                if auto_connect:
                    self._auto_connect_port_names(sub_id, node)

        self._ensure_implicit_component_nodes(
            component_type,
            pretrained=pretrained,
            config=config,
            **kwargs,
        )

    def _ensure_implicit_component_nodes(
        self,
        component_type: str,
        *,
        pretrained: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Auto-insert helper nodes required by ergonomic component APIs.

        When adding prompt_encoder (Conjector), ensure tokenizer (Converter)
        exists (canon: strict separation). Also adds latent_init, added_conditioning.
        """
        family = component_type.split(".", 1)[0]

        # Canon: Tokenizer (Converter) must exist when Conjector (prompt_encoder) is used
        tokenizer_type = f"{family}.tokenizer"
        if component_type in (f"{family}.prompt_encoder", f"{family}.text_encoder"):
            if self._find_node_id_by_block_type(f"{family}/tokenizer") is None and pretrained:
                self.add_node(
                    "__auto_" + family + "_tokenizer",
                    type=tokenizer_type,
                    pretrained=pretrained,
                    auto_connect=True,
                    config=config,
                    **kwargs,
                )

        implicit_specs: Dict[str, List[tuple[str, str]]] = {
            "sd15": [
                ("sd15/latent_init", "__auto_sd15_latent_init"),
            ],
            "sdxl": [
                ("sdxl/latent_init", "__auto_sdxl_latent_init"),
                ("sdxl/added_conditioning", "__auto_sdxl_added_cond"),
            ],
            "flux": [
                ("flux/latent_init", "__auto_flux_latent_init"),
            ],
        }
        _base_latent_init_config: Dict[str, Any] = {
            "height": 512,
            "width": 512,
            "batch_size": 1,
            "dtype": "float32",
            "num_latent_channels": 4,
            "seed": None,
        }
        for block_type, auto_node_id in implicit_specs.get(family, []):
            if self._find_node_id_by_block_type(block_type) is not None:
                continue
            implicit_config = dict(_base_latent_init_config) if "latent_init" in block_type else {}
            if config:
                implicit_config.update(config)
            self._add_block_type_node(
                auto_node_id,
                block_type,
                auto_connect=True,
                config=implicit_config or config,
            )
            if "latent_init" in block_type and not self.metadata.get("num_loop_steps"):
                self.metadata["num_loop_steps"] = 50

    def _find_node_id_by_block_type(self, block_type: str) -> Optional[str]:
        """Return the first node id with the given block type, if any."""
        for nid, node in self._nodes.items():
            if getattr(node, "block_type", None) == block_type:
                return nid
        return None

    def _handle_grouped_component(
        self,
        node_id: str,
        spec: Any,
        loaded: Dict[str, Any],
        *,
        auto_connect: bool = True,
        config: Optional[Dict[str, Any]] = None,
        registry: Any = None,
        **kwargs: Any,
    ) -> None:
        """Handle component types that share a group (e.g. tokenizer + text_encoder)."""
        if not hasattr(self, "_component_groups"):
            self._component_groups: Dict[str, str] = {}

        group = spec.group
        assert group is not None

        existing_nid = self._component_groups.get(group)
        bt = spec.block_types[0]

        if existing_nid and existing_nid in self._nodes:
            existing_node = self._nodes[existing_nid]
            ctor_map = spec.constructor_map.get(bt, {})
            for ctor_kwarg, comp_key in ctor_map.items():
                if comp_key in loaded:
                    setattr(existing_node, f"_{ctor_kwarg}", loaded[comp_key])
            self._component_groups[f"{group}:{node_id}"] = existing_nid
        else:
            ctor_map = spec.constructor_map.get(bt, {})
            build_cfg: Dict[str, Any] = {"block_type": bt, "node_id": node_id}
            for ctor_kwarg, comp_key in ctor_map.items():
                if comp_key in loaded:
                    build_cfg[ctor_kwarg] = loaded[comp_key]
            if config:
                build_cfg.update(config)

            node = registry.build(build_cfg)
            self._add_raw_node(node_id, node)
            self._component_groups[group] = node_id
            if auto_connect:
                self._auto_connect_port_names(node_id, node)

    def _load_pretrained_components(
        self,
        block_type: str,
        pretrained: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load only the components needed for this block type. Each loads separately; cached."""
        from yggdrasill.integrations.diffusers.component_loaders import get_block_type_load_keys
        from yggdrasill.integrations.diffusers.model_store import ModelStore

        load_keys = get_block_type_load_keys(block_type)
        if not load_keys:
            return {}
        family = block_type.split("/", 1)[0]
        ms = ModelStore.default()
        return ms.load_components_by_keys(
            family, load_keys, pretrained,
            variant=kwargs.get("variant", ""),
            revision=kwargs.get("revision"),
            torch_dtype=kwargs.get("torch_dtype"),
        )

    _FAMILY_FULL_KEYS: Dict[str, List[str]] = {
        "sd15": ["tokenizer", "text_encoder", "unet", "vae", "scheduler"],
        "sdxl": ["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "unet", "vae", "scheduler"],
        "flux": ["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "transformer", "vae", "scheduler"],
    }

    def _load_pretrained_components_by_keys(
        self,
        load_keys: List[str],
        pretrained: str,
        *,
        family: str,
        lazy: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load or defer components. Batches load for diffusion families to match from_template speed."""
        if not load_keys:
            return {}
        cache = getattr(self, "_component_load_cache", None)
        if cache is None:
            self._component_load_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
            cache = self._component_load_cache
        cache_key = (family, pretrained)
        if cache_key not in cache and family in self._FAMILY_FULL_KEYS:
            full_keys = self._FAMILY_FULL_KEYS[family]
            from yggdrasill.integrations.diffusers.model_store import ModelStore
            ms = ModelStore.default()
            import torch
            torch_dtype = kwargs.get("torch_dtype")
            if torch_dtype is None and family in self._FAMILY_FULL_KEYS:
                torch_dtype = torch.float32 if family == "sd15" else torch.float16
            cache[cache_key] = ms.load_components_by_keys(
                family, full_keys, pretrained,
                variant=kwargs.get("variant", ""),
                revision=kwargs.get("revision"),
                torch_dtype=torch_dtype,
            )
        if cache_key in cache:
            loaded = {k: cache[cache_key].get(k) for k in load_keys if cache[cache_key].get(k) is not None}
            if loaded:
                return loaded
        import torch
        _dtype = kwargs.get("torch_dtype")
        if _dtype is None and family in self._FAMILY_FULL_KEYS:
            _dtype = torch.float32 if family == "sd15" else torch.float16
        if lazy:
            from yggdrasill.integrations.diffusers.lazy_component import LazyComponent
            return {
                k: LazyComponent(
                    family, k, pretrained,
                    variant=kwargs.get("variant", ""),
                    revision=kwargs.get("revision"),
                    torch_dtype=_dtype,
                )
                for k in load_keys
            }
        from yggdrasill.integrations.diffusers.model_store import ModelStore
        ms = ModelStore.default()
        return ms.load_components_by_keys(
            family, load_keys, pretrained,
            variant=kwargs.get("variant", ""),
            revision=kwargs.get("revision"),
            torch_dtype=_dtype,
        )

    def _auto_connect_port_names(self, node_id: str, node: Any) -> None:
        """Run port-name-based auto-connect for a newly added node."""
        from yggdrasill.task_nodes.auto_connect import apply_port_name_auto_connect
        apply_port_name_auto_connect(self, node_id, node)

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        del self._nodes[node_id]
        self._edges = [
            e for e in self._edges
            if e.source_node != node_id and e.target_node != node_id
        ]
        self._in_edges.pop(node_id, None)
        self._out_edges.pop(node_id, None)
        for nid in list(self._in_edges):
            self._in_edges[nid] = [
                e for e in self._in_edges[nid] if e.source_node != node_id
            ]
        for nid in list(self._out_edges):
            self._out_edges[nid] = [
                e for e in self._out_edges[nid] if e.target_node != node_id
            ]
        self._exposed_inputs = [
            ei for ei in self._exposed_inputs if ei["node_id"] != node_id
        ]
        self._exposed_outputs = [
            eo for eo in self._exposed_outputs if eo["node_id"] != node_id
        ]
        self._node_trainable.pop(node_id, None)
        self._execution_version += 1

    def replace_node(
        self,
        node_id: str,
        *,
        pretrained: Optional[str] = None,
        type: Optional[str] = None,  # noqa: A002
        node: Optional[Any] = None,
        auto_connect: bool = True,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Replace an existing node in the graph.

        Three modes:

        * **Weights-only** (only *pretrained*): reload model weights on
          the existing node without changing the graph topology.
        * **Full swap with type** (*type* and/or *pretrained*): remove
          the old node, create a new one, and re-wire edges whose port
          names still match.
        * **Full swap with object** (*node*): replace with a pre-built
          node object, re-wiring compatible edges.
        """
        target_node_ids = self._resolve_replace_targets(node_id)
        if not target_node_ids:
            raise ValueError(f"Node '{node_id}' not in graph")

        if len(target_node_ids) > 1 and node is not None:
            raise ValueError(
                "Replacing multiple underlying nodes with a single node object "
                "is not supported"
            )

        if len(target_node_ids) > 1:
            for target_node_id in target_node_ids:
                self.replace_node(
                    target_node_id,
                    pretrained=pretrained,
                    type=type,
                    node=None,
                    auto_connect=auto_connect,
                    config=config,
                    **kwargs,
                )
            return

        node_id = target_node_ids[0]

        if pretrained is not None and type is None and node is None:
            self._replace_pretrained_only(node_id, pretrained, **kwargs)
            return

        old_in_edges = list(self.get_edges_in(node_id))
        old_out_edges = list(self.get_edges_out(node_id))
        old_exposed_in = [
            ei for ei in self._exposed_inputs if ei["node_id"] == node_id
        ]
        old_exposed_out = [
            eo for eo in self._exposed_outputs if eo["node_id"] == node_id
        ]

        self.remove_node(node_id)

        if node is not None:
            self._add_raw_node(node_id, node)
        elif type is not None:
            self.add_node(
                node_id,
                type=type,
                pretrained=pretrained,
                auto_connect=False,
                config=config,
                **kwargs,
            )
        else:
            raise ValueError(
                "replace_node requires at least one of: pretrained, type, or node"
            )

        new_node = self._nodes.get(node_id)
        if new_node is None:
            return

        new_in_names: Set[str] = set()
        new_out_names: Set[str] = set()
        if isinstance(new_node, AbstractGraphNode):
            new_in_names = {p.name for p in new_node.get_input_ports()}
            new_out_names = {p.name for p in new_node.get_output_ports()}

        for edge in old_in_edges:
            if edge.target_port in new_in_names:
                try:
                    self.add_edge(edge)
                except (ValueError, KeyError):
                    pass

        for edge in old_out_edges:
            if edge.source_port in new_out_names:
                try:
                    self.add_edge(edge)
                except (ValueError, KeyError):
                    pass

        for ei in old_exposed_in:
            if ei["port_name"] in new_in_names:
                self.expose_input(node_id, ei["port_name"], ei.get("name"))
        for eo in old_exposed_out:
            if eo["port_name"] in new_out_names:
                self.expose_output(node_id, eo["port_name"], eo.get("name"))

        if auto_connect:
            self._auto_connect_port_names(node_id, new_node)

    def _replace_pretrained_only(
        self,
        node_id: str,
        pretrained: str,
        **kwargs: Any,
    ) -> None:
        """Reload model weights on an existing node without changing topology."""
        existing = self._nodes[node_id]
        bt = getattr(existing, "block_type", None)
        if bt is None:
            raise ValueError(
                f"Node '{node_id}' has no block_type; cannot determine "
                "which components to reload."
            )

        from yggdrasill.integrations.diffusers.model_store import ModelStore

        attr_map = {
            "unet": "_unet",
            "vae": "_vae",
            "transformer": "_transformer",
            "controlnet": "_controlnet",
            "scheduler": "_scheduler",
            "tokenizer": "_tokenizer",
            "tokenizer_2": "_tokenizer_2",
            "text_encoder": "_text_encoder",
            "text_encoder_2": "_text_encoder_2",
            "image_encoder": "_image_encoder",
            "feature_extractor": "_feature_extractor",
        }
        needed_keys = [k for k, attr in attr_map.items() if hasattr(existing, attr)]
        if not needed_keys:
            return
        family = bt.split("/", 1)[0]
        ms = ModelStore.default()
        components = ms.load_components_by_keys(
            family, needed_keys, pretrained,
            variant=kwargs.get("variant", ""),
            revision=kwargs.get("revision"),
            torch_dtype=kwargs.get("torch_dtype"),
        )
        for comp_name, attr_name in attr_map.items():
            if hasattr(existing, attr_name) and comp_name in components:
                setattr(existing, attr_name, components[comp_name])

    def _resolve_replace_targets(self, node_ref: str) -> List[str]:
        """Resolve a user-facing node reference into concrete node ids."""
        if node_ref in self._nodes:
            return [node_ref]

        aliases = self.metadata.get("node_aliases", {})
        target = aliases.get(node_ref)
        if target is None:
            return []
        if isinstance(target, str):
            return [target]
        if isinstance(target, list):
            return [nid for nid in target if nid in self._nodes]
        return []

    # --- edges -----------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        if edge.source_node not in self._nodes:
            raise ValueError(f"Source node '{edge.source_node}' not in graph")
        if edge.target_node not in self._nodes:
            raise ValueError(f"Target node '{edge.target_node}' not in graph")

        src_node = self._nodes[edge.source_node]
        dst_node = self._nodes[edge.target_node]

        src_port = None
        dst_port = None

        if isinstance(src_node, AbstractGraphNode):
            src_port = src_node.get_port(edge.source_port)
            if src_port is None:
                raise ValueError(
                    f"Port '{edge.source_port}' not found on node '{edge.source_node}'"
                )
            if src_port.direction != PortDirection.OUT:
                raise ValueError(
                    f"Port '{edge.source_port}' on '{edge.source_node}' is not an output"
                )

        if isinstance(dst_node, AbstractGraphNode):
            dst_port = dst_node.get_port(edge.target_port)
            if dst_port is None:
                raise ValueError(
                    f"Port '{edge.target_port}' not found on node '{edge.target_node}'"
                )
            if dst_port.direction != PortDirection.IN:
                raise ValueError(
                    f"Port '{edge.target_port}' on '{edge.target_node}' is not an input"
                )

        if isinstance(src_node, AbstractGraphNode) and isinstance(dst_node, AbstractGraphNode):
            if src_port is not None and dst_port is not None:
                if not src_port.compatible_with(dst_port):
                    raise ValueError(
                        f"Incompatible port types: {edge.source_node}.{edge.source_port} "
                        f"({src_port.dtype}) -> {edge.target_node}.{edge.target_port} "
                        f"({dst_port.dtype})"
                    )

        if edge in self._edges:
            return  # idempotent

        self._edges.append(edge)
        self._in_edges.setdefault(edge.target_node, []).append(edge)
        self._out_edges.setdefault(edge.source_node, []).append(edge)
        self._execution_version += 1

    def remove_edge(self, edge: Edge) -> None:
        if edge not in self._edges:
            return
        self._edges.remove(edge)
        in_list = self._in_edges.get(edge.target_node, [])
        if edge in in_list:
            in_list.remove(edge)
        out_list = self._out_edges.get(edge.source_node, [])
        if edge in out_list:
            out_list.remove(edge)
        self._execution_version += 1

    def get_edges(self) -> List[Edge]:
        return list(self._edges)

    def get_edges_in(self, node_id: str) -> List[Edge]:
        return list(self._in_edges.get(node_id, []))

    def get_edges_out(self, node_id: str) -> List[Edge]:
        return list(self._out_edges.get(node_id, []))

    # --- exposed inputs / outputs ----------------------------------------

    def expose_input(
        self, node_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        node = self._nodes[node_id]
        if isinstance(node, AbstractGraphNode):
            port = node.get_port(port_name)
            if port is None:
                raise ValueError(f"Port '{port_name}' not found on node '{node_id}'")
            if port.direction != PortDirection.IN:
                raise ValueError(f"Port '{port_name}' on node '{node_id}' is not an input")
        entry: Dict[str, Any] = {"node_id": node_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_inputs:
            if existing["node_id"] == node_id and existing["port_name"] == port_name:
                return
        self._exposed_inputs.append(entry)
        self._execution_version += 1

    def expose_output(
        self, node_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        node = self._nodes[node_id]
        if isinstance(node, AbstractGraphNode):
            port = node.get_port(port_name)
            if port is None:
                raise ValueError(f"Port '{port_name}' not found on node '{node_id}'")
            if port.direction != PortDirection.OUT:
                raise ValueError(f"Port '{port_name}' on node '{node_id}' is not an output")
        entry: Dict[str, Any] = {"node_id": node_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_outputs:
            if existing["node_id"] == node_id and existing["port_name"] == port_name:
                return
        self._exposed_outputs.append(entry)
        self._execution_version += 1

    def get_input_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        result = []
        for entry in self._exposed_inputs:
            rec: Dict[str, Any] = dict(entry)
            if include_dtype:
                node = self._nodes.get(entry["node_id"])
                if node is not None and isinstance(node, AbstractGraphNode):
                    port = node.get_port(entry["port_name"])
                    if port is not None:
                        rec["dtype"] = port.dtype.value if hasattr(port.dtype, "value") else str(port.dtype)
            result.append(rec)
        return result

    def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        result = []
        for entry in self._exposed_outputs:
            rec: Dict[str, Any] = dict(entry)
            if include_dtype:
                node = self._nodes.get(entry["node_id"])
                if node is not None and isinstance(node, AbstractGraphNode):
                    port = node.get_port(entry["port_name"])
                    if port is not None:
                        rec["dtype"] = port.dtype.value if hasattr(port.dtype, "value") else str(port.dtype)
            result.append(rec)
        return result

    # --- Phase 3: config-driven API --------------------------------------

    def add_node_from_config(
        self,
        node_id: str,
        block_type: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        block_id: Optional[str] = None,
        pretrained: Optional[Any] = None,
        trainable: bool = True,
        registry: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Create a task-node via the registry and add it to the graph."""
        from yggdrasill.foundation.registry import BlockRegistry

        if not node_id or not node_id.strip():
            raise ValueError("node_id must be non-empty")
        node_id = node_id.strip()
        if node_id in self._nodes:
            raise ValueError(f"node_id '{node_id}' already exists in graph")

        reg = registry or BlockRegistry.global_registry()
        build_cfg: Dict[str, Any] = {"block_type": block_type, "node_id": node_id}
        if block_id is not None:
            build_cfg["block_id"] = block_id
        if config:
            safe_cfg = {
                k: v for k, v in config.items()
                if k not in ("block_type", "node_id", "block_id")
            }
            build_cfg.update(safe_cfg)

        node = reg.build(build_cfg)
        self.add_node(node_id, node)
        self._node_trainable[node_id] = trainable

        if pretrained is not None:
            if isinstance(pretrained, (str, Path)):
                from yggdrasill.hypergraph.serialization import _read_checkpoint
                pretrained = _read_checkpoint(Path(pretrained))
            if isinstance(pretrained, dict):
                node.load_state_dict(pretrained, strict=False)

        if kwargs.get("auto_connect") and hasattr(self, "_auto_connect_fn"):
            self._auto_connect_fn(self, node_id, node)

        return node_id

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
    ) -> "Hypergraph":
        """Build a Hypergraph from a config dict."""
        import warnings

        from yggdrasill.foundation.registry import BlockRegistry

        sv = config.get("schema_version")
        if sv is not None and sv != "1.0":
            warnings.warn(
                f"Config schema_version '{sv}' differs from supported '1.0'; "
                f"loading may produce unexpected results",
                stacklevel=2,
            )

        reg = registry or BlockRegistry.global_registry()
        g = cls(graph_id=config.get("graph_id", "graph"))
        g.graph_kind = config.get("graph_kind")
        g.metadata = dict(config.get("metadata", {}))

        for nc in config.get("nodes", []):
            nid = nc["node_id"]
            bt = nc["block_type"]
            node_cfg = _resolve_config_ref(nc.get("config") or {})
            bid = nc.get("block_id")
            build_cfg: Dict[str, Any] = {"block_type": bt, "node_id": nid}
            if bid is not None:
                build_cfg["block_id"] = bid
            if node_cfg:
                safe_cfg = {
                    k: v for k, v in node_cfg.items()
                    if k not in ("block_type", "node_id", "block_id")
                }
                build_cfg.update(safe_cfg)
            node = reg.build(build_cfg)
            g.add_node(nid, node)
            g._node_trainable[nid] = nc.get("trainable", True)

        for ec in config.get("edges", []):
            g.add_edge(Edge(
                source_node=ec["source_node"],
                source_port=ec["source_port"],
                target_node=ec["target_node"],
                target_port=ec["target_port"],
            ))

        for ei in config.get("exposed_inputs", []):
            g.expose_input(ei["node_id"], ei["port_name"], ei.get("name"))

        for eo in config.get("exposed_outputs", []):
            g.expose_output(eo["node_id"], eo["port_name"], eo.get("name"))

        if validate:
            from yggdrasill.engine.validator import validate as _validate
            result = _validate(g)
            if not result.valid:
                raise ValueError(f"Config validation failed: {result.errors}")

        return g

    @classmethod
    def from_template(
        cls,
        template_name: str,
        *,
        repo_id: Optional[str] = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "Hypergraph":
        """Build a hypergraph from a named high-level template.

        Example:
            ``Hypergraph.from_template("sd15_text2img", repo_id="runwayml/stable-diffusion-v1-5")``
        """
        from yggdrasill.templates import build_template, list_templates

        key = template_name.strip().lower()
        if key not in list_templates():
            raise ValueError(
                f"Unknown template '{template_name}'. Available: {list(list_templates())}"
            )
        opts: Dict[str, Any] = dict(kwargs)
        if repo_id is not None:
            opts["repo_id"] = repo_id
        opts.setdefault("device", device)
        structure = build_template(key, **opts)
        if not isinstance(structure, cls):
            raise TypeError(
                f"Template '{template_name}' produced {type(structure).__name__}, "
                f"expected {cls.__name__}. "
                "Use Workflow.from_template(...) for workflow templates."
            )
        return structure

    def to_config(self) -> Dict[str, Any]:
        """Export the structure as a JSON-serialisable dict (no weights)."""
        nodes = []
        for nid, node in self._nodes.items():
            entry: Dict[str, Any] = {"node_id": nid}
            if hasattr(node, "block_type"):
                entry["block_type"] = node.block_type
            if hasattr(node, "block_id"):
                entry["block_id"] = node.block_id
            if hasattr(node, "config"):
                entry["config"] = node.config
            entry["trainable"] = self._node_trainable.get(nid, True)
            nodes.append(entry)

        edges = [
            {
                "source_node": e.source_node,
                "source_port": e.source_port,
                "target_node": e.target_node,
                "target_port": e.target_port,
            }
            for e in self._edges
        ]

        result: Dict[str, Any] = {
            "schema_version": "1.0",
            "graph_id": self._graph_id,
            "nodes": nodes,
            "edges": edges,
            "exposed_inputs": [dict(ei) for ei in self._exposed_inputs],
            "exposed_outputs": [dict(eo) for eo in self._exposed_outputs],
        }
        if self._graph_kind is not None:
            result["graph_kind"] = self._graph_kind
        if self._metadata:
            result["metadata"] = dict(self._metadata)
        return result

    def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        *,
        training: bool = False,
        num_loop_steps: Optional[int] = None,
        device: Optional[Any] = None,
        callbacks: Optional[list] = None,
        dry_run: bool = False,
        validate_before: bool = True,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute the hypergraph and return results.

        Accepts either a dict of inputs or keyword arguments that are
        automatically routed to the appropriate graph inputs and executor
        parameters.

        For diffusion graphs the return value is a
        :class:`~yggdrasill.diffusion.output.DiffusionOutput` with an
        ``.images`` attribute; for other graphs a plain ``dict`` is
        returned.

        Keyword argument routing
        ------------------------
        * ``num_inference_steps`` -> ``num_loop_steps`` (executor)
        * ``seed`` -> ``seed`` (executor)
        * ``max_steps`` -> ``max_steps`` (executor, for agent loops)
        * ``show_progress=True`` — tqdm progress bar for denoising steps
        * Names matching an exposed input port -> ``inputs`` dict
        * Remaining kwargs -> ``pin_data`` config overrides on nodes that
          recognise the key in their config.
        """
        from yggdrasill.engine.executor import run as _run

        if not self._exposed_inputs and not self._exposed_outputs and self._nodes:
            self.infer_exposed_ports()

        cbs = list(callbacks) if callbacks else []
        if show_progress and not dry_run:
            cbs.insert(0, _make_progress_callback())

        resolved_inputs, executor_kwargs = self._resolve_run_kwargs(
            inputs, kwargs,
            num_loop_steps=num_loop_steps,
            device=device,
        )

        raw = _run(
            self,
            resolved_inputs,
            training=training,
            num_loop_steps=executor_kwargs.get("num_loop_steps", num_loop_steps),
            device=device,
            callbacks=cbs if cbs else callbacks,
            dry_run=dry_run,
            validate_before=validate_before,
            seed=executor_kwargs.get("seed"),
            pin_data=executor_kwargs.get("pin_data"),
            max_steps=executor_kwargs.get("max_steps"),
        )

        if isinstance(raw, dict):
            return self._maybe_wrap_output(raw)
        return raw

    def _resolve_run_kwargs(
        self,
        inputs: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
        *,
        num_loop_steps: Optional[int] = None,
        device: Optional[Any] = None,
    ) -> tuple:
        """Split kwargs into (inputs_dict, executor_kwargs)."""
        resolved: Dict[str, Any] = dict(inputs or {})
        executor_kw: Dict[str, Any] = {}

        # Propagate device to nodes that create tensors (latent_init, scheduler_setup)
        effective_device = device
        if effective_device is None:
            try:
                from yggdrasill.integrations.diffusers.model_store import ModelStore
                ms = ModelStore.default()
                if ms.device and ms.device != "cpu":
                    effective_device = ms.device
            except Exception:
                pass
        if effective_device is None:
            for nid, node in self._nodes.items():
                for attr in ("_unet", "_vae", "_scheduler", "_text_encoder"):
                    model = getattr(node, attr, None)
                    if model is not None and hasattr(model, "parameters"):
                        try:
                            p = next(model.parameters(), None)
                            if p is not None:
                                effective_device = str(p.device)
                                break
                        except (StopIteration, TypeError, RuntimeError):
                            pass
                if effective_device is not None:
                    break
        if effective_device is not None:
            dev_str = str(effective_device)
            for nid, node in self._nodes.items():
                bt = getattr(node, "block_type", None) or ""
                if "latent_init" in bt or "scheduler_setup" in bt:
                    if not hasattr(node, "_config"):
                        node._config = {}
                    node._config["device"] = dev_str

        num_inference_steps = kwargs.pop("num_inference_steps", None)
        if num_inference_steps is not None:
            executor_kw["num_loop_steps"] = num_inference_steps
            for nid, node in self._nodes.items():
                bt = getattr(node, "block_type", None)
                if bt and "scheduler_setup" in str(bt):
                    if not hasattr(node, "_config"):
                        node._config = {}
                    node._config["num_inference_steps"] = num_inference_steps
                    break
        if "seed" in kwargs:
            seed_val = kwargs.pop("seed")
            executor_kw["seed"] = seed_val
            target_nid = None
            for nid, node in self._nodes.items():
                node_cfg = getattr(node, "_config", None) or {}
                if "seed" not in node_cfg:
                    continue
                bt = getattr(node, "block_type", "") or ""
                if "latent_init" in bt:
                    target_nid = nid
                    break
                if target_nid is None:
                    target_nid = nid
            if target_nid is not None:
                node = self._nodes[target_nid]
                if not hasattr(node, "_config"):
                    node._config = {}
                node._config["seed"] = seed_val
        if "max_steps" in kwargs:
            executor_kw["max_steps"] = kwargs.pop("max_steps")

        input_spec = self.get_input_spec()
        exposed_names = set()
        for spec_entry in input_spec:
            key = self._spec_key(spec_entry)
            exposed_names.add(key)
            port_name = spec_entry["port_name"]
            exposed_names.add(port_name)

        for key, val in list(kwargs.items()):
            if key in exposed_names:
                resolved[key] = val
            else:
                for nid, node in self._nodes.items():
                    node_cfg = getattr(node, "_config", None) or {}
                    if key in node_cfg:
                        if not hasattr(node, "_config"):
                            node._config = {}
                        node._config[key] = val
                        break

        return resolved, executor_kw

    def _maybe_wrap_output(self, raw: Dict[str, Any]) -> Any:
        """Wrap raw output in DiffusionOutput if graph has diffusion outputs."""
        output_spec = self.get_output_spec()
        output_port_names = {
            spec.get("name", spec["port_name"]) for spec in output_spec
        }

        from yggdrasill.diffusion.contracts import PORT_DECODED_IMAGE, PORT_OUTPUT_IMAGE

        diffusion_keys = {PORT_DECODED_IMAGE, PORT_OUTPUT_IMAGE}
        if output_port_names & diffusion_keys:
            from yggdrasill.diffusion.output import DiffusionOutput

            image_key = (
                PORT_DECODED_IMAGE if PORT_DECODED_IMAGE in raw else
                PORT_OUTPUT_IMAGE if PORT_OUTPUT_IMAGE in raw else
                next((k for k in raw if k.endswith(":" + PORT_DECODED_IMAGE) or k.endswith(":" + PORT_OUTPUT_IMAGE)), None)
            )
            if image_key is not None:
                return DiffusionOutput.from_executor_output(raw, image_key=image_key)

        return raw

    def _is_canonical_diffusion_text2img(self, family: str) -> bool:
        """True if graph has all required nodes for diffusion text2img."""
        backbone = f"{family}/transformer" if family == "flux" else f"{family}/unet"
        req = {
            f"{family}/tokenizer",
            f"{family}/prompt_encoder",
            backbone,
            f"{family}/vae_decode",
            f"{family}/latent_init",
            f"{family}/scheduler_setup",
            f"{family}/scheduler_step",
        }
        if family == "sdxl":
            req.add(f"{family}/added_conditioning")
        present = {
            getattr(n, "block_type", None) for n in self._nodes.values()
        }
        return req.issubset(present)

    def infer_exposed_ports(self) -> None:
        """Auto-detect exposed inputs/outputs. For canonical diffusion text2img,
        only expose prompt, negative_prompt, decoded_image."""
        from yggdrasill.diffusion import contracts as C

        self._exposed_inputs.clear()
        self._exposed_outputs.clear()

        for family in ("sd15", "sdxl", "flux"):
            if self._is_canonical_diffusion_text2img(family):
                tok_id = self._find_node_id_by_block_type(f"{family}/tokenizer")
                vae_id = self._find_node_id_by_block_type(f"{family}/vae_decode")
                if tok_id:
                    self._exposed_inputs.append(
                        {"node_id": tok_id, "port_name": C.PORT_PROMPT, "name": C.PORT_PROMPT}
                    )
                    self._exposed_inputs.append(
                        {"node_id": tok_id, "port_name": C.PORT_NEGATIVE_PROMPT, "name": C.PORT_NEGATIVE_PROMPT}
                    )
                if vae_id:
                    self._exposed_outputs.append(
                        {"node_id": vae_id, "port_name": C.PORT_DECODED_IMAGE, "name": C.PORT_OUTPUT_IMAGE}
                    )
                self._execution_version += 1
                return

        for nid, node in self._nodes.items():
            if not isinstance(node, AbstractGraphNode):
                continue
            in_edges = self.get_edges_in(nid)
            covered_in = {e.target_port for e in in_edges}
            for port in node.get_input_ports():
                if port.name not in covered_in:
                    self._exposed_inputs.append({"node_id": nid, "port_name": port.name})

            out_edges = self.get_edges_out(nid)
            covered_out = {e.source_port for e in out_edges}
            for port in node.get_output_ports():
                if port.name not in covered_out:
                    self._exposed_outputs.append({"node_id": nid, "port_name": port.name})
        self._execution_version += 1

    # --- state dict (preparation for Phase 5) ----------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict, deduplicated by block_id for shared blocks."""
        result: Dict[str, Any] = {}
        seen_block_ids: Dict[str, str] = {}
        aliases: Dict[str, str] = {}
        for nid in sorted(self._nodes.keys()):
            node = self._nodes[nid]
            if not hasattr(node, "state_dict"):
                continue
            sd = node.state_dict()
            if not sd:
                continue
            bid = getattr(node, "block_id", nid)
            if bid in seen_block_ids:
                aliases[nid] = seen_block_ids[bid]
            else:
                seen_block_ids[bid] = nid
                result[nid] = sd
        if aliases:
            result["_aliases"] = aliases
        return result

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        aliases: Dict[str, str] = state.get("_aliases", {})
        expanded = {k: v for k, v in state.items() if k != "_aliases"}
        for nid, alias_target in aliases.items():
            if alias_target in expanded:
                expanded[nid] = expanded[alias_target]

        if strict:
            extra = set(expanded.keys()) - set(self._nodes.keys())
            if extra:
                raise KeyError(f"state_dict has keys not in graph: {extra}")
        for nid, node in self._nodes.items():
            if nid in expanded and hasattr(node, "load_state_dict"):
                node.load_state_dict(expanded[nid], strict=strict)

    # --- device / trainable ----------------------------------------------

    def to(self, device: Any) -> "Hypergraph":
        for node in self._nodes.values():
            if hasattr(node, "_config") and node._config is not None:
                node._config["device"] = device
            if hasattr(node, "to") and callable(node.to):
                node.to(device)
        return self

    def set_trainable(self, node_id: str, trainable: bool) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        self._node_trainable[node_id] = trainable

    def trainable_parameters(self) -> Iterator[Any]:
        """Yield trainable parameters, deduplicated by block_id."""
        seen_block_ids: Set[str] = set()
        for nid, node in self._nodes.items():
            if not self._node_trainable.get(nid, True):
                continue
            block_id = getattr(node, "block_id", nid)
            if block_id in seen_block_ids:
                continue
            seen_block_ids.add(block_id)
            if hasattr(node, "trainable_parameters") and callable(node.trainable_parameters):
                yield from node.trainable_parameters()

    # --- serialization (Phase 5) delegates to hypergraph.serialization ----

    def save(
        self,
        directory: "str | Path",
        *,
        config_filename: str = "config.json",
        checkpoint_filename: str = "checkpoint.pkl",
    ) -> Path:
        from yggdrasill.hypergraph.serialization import save_hypergraph
        return save_hypergraph(
            self, directory,
            config_filename=config_filename,
            checkpoint_filename=checkpoint_filename,
        )

    def save_config(
        self,
        directory: "str | Path",
        *,
        filename: str = "config.json",
    ) -> Path:
        from yggdrasill.hypergraph.serialization import save_config as _save_cfg
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        cfg = self.to_config()
        cfg["schema_version"] = "1.0"
        _save_cfg(cfg, d / filename)
        return d

    def save_checkpoint(
        self,
        directory: "str | Path",
        *,
        filename: str = "checkpoint.pkl",
    ) -> Path:
        from yggdrasill.hypergraph.serialization import save_checkpoint as _save_ckpt
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        from yggdrasill.hypergraph.serialization import _deduplicate_state
        full_state = self.state_dict()
        deduped = _deduplicate_state(self, full_state)
        _save_ckpt(deduped, d / filename)
        return d

    @classmethod
    def load(
        cls,
        directory: "str | Path",
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
        config_filename: str = "config.json",
        checkpoint_filename: str = "checkpoint.pkl",
        load_checkpoint: bool = True,
    ) -> "Hypergraph":
        from yggdrasill.hypergraph.serialization import load_hypergraph
        return load_hypergraph(
            directory, registry=registry, validate=validate,
            config_filename=config_filename,
            checkpoint_filename=checkpoint_filename,
            load_checkpoint_flag=load_checkpoint,
        )

    @classmethod
    def load_config(
        cls,
        directory: "str | Path",
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
        config_filename: str = "config.json",
    ) -> "Hypergraph":
        from yggdrasill.hypergraph.serialization import load_hypergraph
        return load_hypergraph(
            directory, registry=registry, validate=validate,
            config_filename=config_filename,
            load_checkpoint_flag=False,
        )

    def load_from_checkpoint(
        self,
        directory: "str | Path",
        *,
        checkpoint_filename: str = "checkpoint.pkl",
    ) -> None:
        """Load weights into an already-built hypergraph."""
        from yggdrasill.hypergraph.serialization import (
            _expand_deduped_state, _read_checkpoint,
        )
        p = Path(directory) / checkpoint_filename
        state = _read_checkpoint(p)
        expanded = _expand_deduped_state(self, state)
        self.load_state_dict(expanded, strict=False)

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _spec_key(entry: Dict[str, Any]) -> str:
        if "name" in entry and entry["name"] is not None:
            return entry["name"]
        nid = entry.get("node_id") or entry.get("graph_id", "")
        return f"{nid}:{entry['port_name']}"
