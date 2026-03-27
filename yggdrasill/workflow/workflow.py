"""Workflow: a hypergraph of hypergraphs.

Nodes are Hypergraph instances.  Edges connect their exposed outputs
to other hypergraphs' exposed inputs.  The same engine (Validator,
Planner, Executor) runs the workflow without any changes.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.hypergraph.serialization import (
    load_checkpoint as _load_checkpoint_file,
    load_config as _load_config_file,
    save_checkpoint as _save_checkpoint_file,
    save_config as _save_config_file,
)
from yggdrasill.hypergraph.structure import _make_progress_callback

_wf_instance_counter = itertools.count()


class Workflow:
    """Hypergraph of hypergraphs -- implements the same structural protocol
    that the engine expects (node_ids, get_node, get_edges, etc.)."""

    @classmethod
    def from_template(cls, template_name: str, **kwargs: Any) -> "Workflow":
        """Build a diffusion workflow from a template name. Requires diffusers addon."""
        from yggdrasill.integrations.diffusers.templates import build_template

        built = build_template(template_name, **kwargs)
        if not isinstance(built, Workflow):
            raise TypeError(
                f"Template '{template_name}' returned {type(built).__name__}, "
                "expected Workflow. Use Hypergraph.from_template for graph templates."
            )
        return built

    def __init__(self, workflow_id: Optional[str] = None) -> None:
        self._instance_id = next(_wf_instance_counter)
        self._workflow_id = workflow_id or "workflow"
        self._workflow_kind: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

        self._nodes: Dict[str, Hypergraph] = {}
        self._edges: List[Edge] = []
        self._in_edges: Dict[str, List[Edge]] = {}
        self._out_edges: Dict[str, List[Edge]] = {}

        self._exposed_inputs: List[Dict[str, Any]] = []
        self._exposed_outputs: List[Dict[str, Any]] = []

        self._execution_version: int = 0
        self._node_trainable: Dict[str, bool] = {}

    # --- identity / metadata -----------------------------------------------

    @property
    def workflow_id(self) -> str:
        return self._workflow_id

    @property
    def graph_id(self) -> str:
        return self._workflow_id

    @property
    def workflow_kind(self) -> Optional[str]:
        return self._workflow_kind

    @workflow_kind.setter
    def workflow_kind(self, value: Optional[str]) -> None:
        self._workflow_kind = value

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self._metadata = dict(value)

    @property
    def execution_version(self) -> int:
        return self._execution_version

    # --- structural protocol (engine-facing) --------------------------------

    @property
    def node_ids(self) -> Set[str]:
        return set(self._nodes.keys())

    def get_node(self, node_id: str) -> Optional[Hypergraph]:
        return self._nodes.get(node_id)

    def get_edges(self) -> List[Edge]:
        return list(self._edges)

    def get_edges_in(self, node_id: str) -> List[Edge]:
        return list(self._in_edges.get(node_id, []))

    def get_edges_out(self, node_id: str) -> List[Edge]:
        return list(self._out_edges.get(node_id, []))

    def get_input_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for entry in self._exposed_inputs:
            rec: Dict[str, Any] = dict(entry)
            if include_dtype:
                gid = entry.get("graph_id") or entry.get("node_id")
                graph = self._nodes.get(gid)
                if graph is not None:
                    for sp in graph.get_input_spec(include_dtype=True):
                        if sp["port_name"] == entry["port_name"]:
                            if "dtype" in sp:
                                rec["dtype"] = sp["dtype"]
                            break
            result.append(rec)
        return result

    def get_output_spec(self, include_dtype: bool = False) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for entry in self._exposed_outputs:
            rec: Dict[str, Any] = dict(entry)
            if include_dtype:
                gid = entry.get("graph_id") or entry.get("node_id")
                graph = self._nodes.get(gid)
                if graph is not None:
                    for sp in graph.get_output_spec(include_dtype=True):
                        if sp["port_name"] == entry["port_name"]:
                            if "dtype" in sp:
                                rec["dtype"] = sp["dtype"]
                            break
            result.append(rec)
        return result

    # --- node management ----------------------------------------------------

    def add_node(self, graph_id: str, hypergraph: Hypergraph) -> str:
        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id must be non-empty")
        graph_id = graph_id.strip()
        self._nodes[graph_id] = hypergraph
        self._in_edges.setdefault(graph_id, [])
        self._out_edges.setdefault(graph_id, [])
        self._node_trainable.setdefault(graph_id, True)
        self._execution_version += 1
        return graph_id

    def add_node_from_config(
        self,
        graph_id: str,
        config: Dict[str, Any],
        *,
        registry: Optional[Any] = None,
        trainable: bool = True,
    ) -> str:
        """Build a Hypergraph from *config* (or ``{"ref": "path"}``) and add it."""
        from yggdrasill.hypergraph.structure import _resolve_config_ref
        resolved = _resolve_config_ref(config)
        hg = Hypergraph.from_config(resolved, registry=registry)
        self.add_node(graph_id, hg)
        self._node_trainable[graph_id] = trainable
        return graph_id

    def remove_node(self, graph_id: str) -> None:
        if graph_id not in self._nodes:
            return
        del self._nodes[graph_id]
        self._edges = [
            e for e in self._edges
            if e.source_node != graph_id and e.target_node != graph_id
        ]
        self._in_edges.pop(graph_id, None)
        self._out_edges.pop(graph_id, None)
        for nid in list(self._in_edges):
            self._in_edges[nid] = [
                e for e in self._in_edges[nid] if e.source_node != graph_id
            ]
        for nid in list(self._out_edges):
            self._out_edges[nid] = [
                e for e in self._out_edges[nid] if e.target_node != graph_id
            ]
        self._exposed_inputs = [
            ei for ei in self._exposed_inputs
            if (ei.get("graph_id") or ei.get("node_id")) != graph_id
        ]
        self._exposed_outputs = [
            eo for eo in self._exposed_outputs
            if (eo.get("graph_id") or eo.get("node_id")) != graph_id
        ]
        self._node_trainable.pop(graph_id, None)
        self._execution_version += 1

    # --- edge management ----------------------------------------------------

    def add_edge(
        self,
        source_graph_id: str,
        source_port: str,
        target_graph_id: str,
        target_port: str,
    ) -> None:
        if source_graph_id not in self._nodes:
            raise ValueError(f"Source graph '{source_graph_id}' not in workflow")
        if target_graph_id not in self._nodes:
            raise ValueError(f"Target graph '{target_graph_id}' not in workflow")

        src_graph = self._nodes[source_graph_id]
        dst_graph = self._nodes[target_graph_id]

        src_port_names = {
            sp["port_name"] for sp in src_graph.get_output_spec()
        }
        if source_port not in src_port_names:
            raise ValueError(
                f"Port '{source_port}' not in output spec of graph '{source_graph_id}'"
            )

        dst_port_names = {
            sp["port_name"] for sp in dst_graph.get_input_spec()
        }
        if target_port not in dst_port_names:
            raise ValueError(
                f"Port '{target_port}' not in input spec of graph '{target_graph_id}'"
            )

        edge = Edge(
            source_node=source_graph_id,
            source_port=source_port,
            target_node=target_graph_id,
            target_port=target_port,
        )
        if edge in self._edges:
            return
        self._edges.append(edge)
        self._in_edges.setdefault(edge.target_node, []).append(edge)
        self._out_edges.setdefault(edge.source_node, []).append(edge)
        self._execution_version += 1

    def remove_edge(
        self,
        source_graph_id: str,
        source_port: str,
        target_graph_id: str,
        target_port: str,
    ) -> None:
        edge = Edge(
            source_node=source_graph_id,
            source_port=source_port,
            target_node=target_graph_id,
            target_port=target_port,
        )
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

    # --- exposed ports ------------------------------------------------------

    def expose_input(
        self, graph_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if graph_id not in self._nodes:
            raise ValueError(f"Graph '{graph_id}' not in workflow")
        graph = self._nodes[graph_id]
        input_port_names = {sp["port_name"] for sp in graph.get_input_spec()}
        if port_name not in input_port_names:
            raise ValueError(
                f"Port '{port_name}' not in input spec of graph '{graph_id}'"
            )
        entry: Dict[str, Any] = {"graph_id": graph_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_inputs:
            eid = existing.get("graph_id") or existing.get("node_id")
            if eid == graph_id and existing.get("port_name") == port_name:
                if name is not None and existing.get("name") != name:
                    existing["name"] = name
                    self._execution_version += 1
                return
        self._exposed_inputs.append(entry)
        self._execution_version += 1

    def expose_output(
        self, graph_id: str, port_name: str, name: Optional[str] = None,
    ) -> None:
        if graph_id not in self._nodes:
            raise ValueError(f"Graph '{graph_id}' not in workflow")
        graph = self._nodes[graph_id]
        output_port_names = {sp["port_name"] for sp in graph.get_output_spec()}
        if port_name not in output_port_names:
            raise ValueError(
                f"Port '{port_name}' not in output spec of graph '{graph_id}'"
            )
        entry: Dict[str, Any] = {"graph_id": graph_id, "port_name": port_name}
        if name is not None:
            entry["name"] = name
        for existing in self._exposed_outputs:
            eid = existing.get("graph_id") or existing.get("node_id")
            if eid == graph_id and existing.get("port_name") == port_name:
                if name is not None and existing.get("name") != name:
                    existing["name"] = name
                    self._execution_version += 1
                return
        self._exposed_outputs.append(entry)
        self._execution_version += 1

    # --- infer exposed ports ------------------------------------------------

    def infer_exposed_ports(self) -> None:
        """Auto-detect exposed inputs/outputs by absence of workflow edges."""
        self._exposed_inputs.clear()
        self._exposed_outputs.clear()

        for graph_id, graph in self._nodes.items():
            in_edges = self.get_edges_in(graph_id)
            covered_in = {e.target_port for e in in_edges}
            for sp in graph.get_input_spec():
                if sp["port_name"] not in covered_in:
                    self._exposed_inputs.append({
                        "graph_id": graph_id,
                        "port_name": sp["port_name"],
                    })

            out_edges = self.get_edges_out(graph_id)
            covered_out = {e.source_port for e in out_edges}
            for sp in graph.get_output_spec():
                if sp["port_name"] not in covered_out:
                    self._exposed_outputs.append({
                        "graph_id": graph_id,
                        "port_name": sp["port_name"],
                    })

        self._execution_version += 1

    # --- run (delegates to engine) ------------------------------------------

    def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        *,
        training: bool = False,
        num_loop_steps: Optional[int] = None,
        device: Optional[Any] = None,
        callbacks: Optional[List[Callable[..., None]]] = None,
        dry_run: bool = False,
        validate_before: bool = True,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from yggdrasill.engine.executor import run as _run
        cbs = list(callbacks) if callbacks else []
        if show_progress and not dry_run:
            cbs.insert(0, _make_progress_callback())

        resolved_inputs, executor_kwargs = self._resolve_run_kwargs(
            inputs, kwargs,
            num_loop_steps=num_loop_steps,
            device=device,
        )
        return _run(
            self, resolved_inputs,
            training=training,
            num_loop_steps=executor_kwargs.get("num_loop_steps", num_loop_steps),
            device=executor_kwargs.get("device", device),
            callbacks=cbs if cbs else callbacks,
            dry_run=dry_run,
            validate_before=validate_before,
            seed=executor_kwargs.get("seed"),
            pin_data=executor_kwargs.get("pin_data"),
            max_steps=executor_kwargs.get("max_steps"),
            run_data=executor_kwargs.get("run_data"),
            destination_node_id=executor_kwargs.get("destination_node_id"),
            dirty_node_ids=executor_kwargs.get("dirty_node_ids"),
            interrupt_on=executor_kwargs.get("interrupt_on"),
            skip_node_ids=executor_kwargs.get("skip_node_ids"),
        )

    def _resolve_run_kwargs(
        self,
        inputs: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
        *,
        num_loop_steps: Optional[int] = None,
        device: Optional[Any] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Split workflow run kwargs into user inputs and executor kwargs."""
        resolved: Dict[str, Any] = dict(inputs or {})
        executor_kw: Dict[str, Any] = {}

        if num_loop_steps is not None:
            executor_kw["num_loop_steps"] = num_loop_steps
        if device is not None:
            executor_kw["device"] = device

        num_inference_steps = kwargs.pop("num_inference_steps", None)
        if num_inference_steps is not None:
            executor_kw["num_loop_steps"] = num_inference_steps
        if "seed" in kwargs:
            executor_kw["seed"] = kwargs.pop("seed")
        if "max_steps" in kwargs:
            executor_kw["max_steps"] = kwargs.pop("max_steps")
        if "pin_data" in kwargs:
            executor_kw["pin_data"] = kwargs.pop("pin_data")
        if "run_data" in kwargs:
            executor_kw["run_data"] = kwargs.pop("run_data")
        if "destination_node_id" in kwargs:
            executor_kw["destination_node_id"] = kwargs.pop("destination_node_id")
        if "dirty_node_ids" in kwargs:
            dirty = kwargs.pop("dirty_node_ids")
            if dirty is not None:
                executor_kw["dirty_node_ids"] = list(dirty)
        if "interrupt_on" in kwargs:
            interrupt = kwargs.pop("interrupt_on")
            if interrupt is not None:
                executor_kw["interrupt_on"] = list(interrupt)
        if "skip_node_ids" in kwargs:
            skip = kwargs.pop("skip_node_ids")
            if skip is not None:
                executor_kw["skip_node_ids"] = set(skip) if not isinstance(skip, set) else skip

        input_spec = self.get_input_spec()
        exposed_names = set()
        port_name_counts: Dict[str, int] = {}
        for spec_entry in input_spec:
            pname = spec_entry["port_name"]
            port_name_counts[pname] = port_name_counts.get(pname, 0) + 1
        for spec_entry in input_spec:
            key = spec_entry.get("name") or spec_entry["port_name"]
            exposed_names.add(key)
            port_name = spec_entry["port_name"]
            if port_name_counts.get(port_name, 0) == 1:
                exposed_names.add(port_name)

        for key, val in list(kwargs.items()):
            if key in exposed_names:
                resolved[key] = val

        return resolved, executor_kw

    # --- state dict ---------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for nid, hg in self._nodes.items():
            result[nid] = hg.state_dict()
        return result

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if strict:
            extra = set(state.keys()) - set(self._nodes.keys())
            if extra:
                raise KeyError(f"state_dict has keys not in workflow: {extra}")
        for nid, hg in self._nodes.items():
            if nid in state:
                hg.load_state_dict(state[nid], strict=strict)

    # --- device / trainable -------------------------------------------------

    def to(self, device: Any) -> "Workflow":
        for hg in self._nodes.values():
            if hasattr(hg, "to") and callable(hg.to):
                hg.to(device)
        return self

    def set_trainable(self, graph_id: str, trainable: bool) -> None:
        if graph_id not in self._nodes:
            raise ValueError(f"Graph '{graph_id}' not in workflow")
        self._node_trainable[graph_id] = trainable

    def trainable_parameters(self) -> Iterator[Any]:
        for gid, hg in self._nodes.items():
            if not self._node_trainable.get(gid, True):
                continue
            if hasattr(hg, "trainable_parameters"):
                yield from hg.trainable_parameters()

    # --- config -------------------------------------------------------------

    def to_config(self) -> Dict[str, Any]:
        graphs = []
        for gid, hg in self._nodes.items():
            entry: Dict[str, Any] = {
                "graph_id": gid,
                "config": hg.to_config(),
            }
            if not self._node_trainable.get(gid, True):
                entry["trainable"] = False
            graphs.append(entry)

        edges = [
            {
                "source_graph": e.source_node,
                "source_port": e.source_port,
                "target_graph": e.target_node,
                "target_port": e.target_port,
            }
            for e in self._edges
        ]

        cfg: Dict[str, Any] = {
            "schema_version": "1.0",
            "workflow_id": self._workflow_id,
            "graphs": graphs,
            "edges": edges,
            "exposed_inputs": [dict(ei) for ei in self._exposed_inputs],
            "exposed_outputs": [dict(eo) for eo in self._exposed_outputs],
        }
        if self._workflow_kind is not None:
            cfg["workflow_kind"] = self._workflow_kind
        if self._metadata:
            cfg["metadata"] = dict(self._metadata)
        return cfg

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
    ) -> "Workflow":
        import warnings
        sv = config.get("schema_version")
        if sv is not None and sv != "1.0":
            warnings.warn(
                f"Workflow config schema_version='{sv}' differs from expected '1.0'",
                stacklevel=2,
            )

        w = cls(workflow_id=config.get("workflow_id", "workflow"))
        w.workflow_kind = config.get("workflow_kind")
        w.metadata = dict(config.get("metadata", {}))

        from yggdrasill.hypergraph.structure import _resolve_config_ref

        for gc in config.get("graphs", []):
            gid = gc["graph_id"]
            if "ref" in gc and "config" not in gc:
                graph_config = _resolve_config_ref({"ref": gc["ref"]})
            else:
                graph_config = gc["config"]
            hg = Hypergraph.from_config(graph_config, registry=registry)
            w.add_node(gid, hg)
            if "trainable" in gc:
                w._node_trainable[gid] = gc["trainable"]

        for ec in config.get("edges", []):
            w.add_edge(
                ec["source_graph"],
                ec["source_port"],
                ec["target_graph"],
                ec["target_port"],
            )

        for ei in config.get("exposed_inputs", []):
            w.expose_input(
                ei.get("node_id") or ei.get("graph_id"),
                ei["port_name"],
                ei.get("name"),
            )

        for eo in config.get("exposed_outputs", []):
            w.expose_output(
                eo.get("node_id") or eo.get("graph_id"),
                eo["port_name"],
                eo.get("name"),
            )

        if validate:
            from yggdrasill.engine.validator import validate as _validate
            result = _validate(w)
            if not result.valid:
                raise ValueError(f"Workflow validation failed: {result.errors}")

        return w

    # --- save / load --------------------------------------------------------

    def save(
        self,
        directory: str | Path,
        *,
        config_filename: str = "config.json",
        checkpoint_filename: str = "checkpoint.pkl",
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        cfg = self.to_config()
        _save_config_file(cfg, directory / config_filename)

        state = self.state_dict()
        _save_checkpoint_file(state, directory / checkpoint_filename)

        return directory

    def save_config(
        self,
        directory: str | Path,
        *,
        filename: str = "config.json",
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        cfg = self.to_config()
        _save_config_file(cfg, directory / filename)
        return directory

    def save_checkpoint(
        self,
        directory: str | Path,
        *,
        filename: str = "checkpoint.pkl",
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        _save_checkpoint_file(state, directory / filename)
        return directory

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
        config_filename: str = "config.json",
        checkpoint_filename: str = "checkpoint.pkl",
        load_checkpoint_flag: bool = True,
    ) -> "Workflow":
        directory = Path(directory)
        config = _load_config_file(directory / config_filename)

        w = cls.from_config(config, registry=registry, validate=validate)

        if load_checkpoint_flag:
            ckpt_path = directory / checkpoint_filename
            if ckpt_path.exists():
                state = _load_checkpoint_file(ckpt_path)
                w.load_state_dict(state, strict=False)

        return w

    @classmethod
    def load_config(
        cls,
        directory: str | Path,
        *,
        registry: Optional[Any] = None,
        validate: bool = False,
        config_filename: str = "config.json",
    ) -> "Workflow":
        """Load only the workflow config (no checkpoint)."""
        directory = Path(directory)
        config = _load_config_file(directory / config_filename)
        return cls.from_config(config, registry=registry, validate=validate)

    def load_from_checkpoint(
        self,
        directory: str | Path,
        *,
        checkpoint_filename: str = "checkpoint.pkl",
    ) -> None:
        """Load checkpoint from disk.

        Warning: uses ``pickle.load`` internally -- only load checkpoints
        from trusted sources.
        """
        directory = Path(directory)
        ckpt_path = directory / checkpoint_filename
        state = _load_checkpoint_file(ckpt_path)
        self.load_state_dict(state, strict=False)
