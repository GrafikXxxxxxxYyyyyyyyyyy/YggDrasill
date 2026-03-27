from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from yggdrasill.foundation.port import PortAggregation


class EdgeBuffers:
    """Per-run storage keyed by (node_id, port_name).

    Supports single-value writes, multi-edge appends with source tracking,
    and aggregation according to ``PortAggregation`` policy.
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], Any] = {}
        self._multi: Dict[Tuple[str, str], List[Tuple[Optional[str], Any]]] = {}
        #: Optional scratch for training-step executors (not part of port buffers / serialization).
        self.scratch: Dict[str, Any] = {}

    # --- single-value operations ---

    def write(self, node_id: str, port_name: str, value: Any) -> None:
        key = (node_id, port_name)
        self._data[key] = value

    def read(self, node_id: str, port_name: str) -> Any:
        return self._data.get((node_id, port_name))

    def has(self, node_id: str, port_name: str) -> bool:
        return (node_id, port_name) in self._data

    # --- multi-edge operations ---

    def append(
        self,
        node_id: str,
        port_name: str,
        value: Any,
        *,
        source_node: Optional[str] = None,
    ) -> None:
        """Append a value for multi-edge aggregation, optionally tracking the source."""
        key = (node_id, port_name)
        self._multi.setdefault(key, []).append((source_node, value))

    def aggregate(
        self,
        node_id: str,
        port_name: str,
        policy: PortAggregation = PortAggregation.SINGLE,
    ) -> Any:
        key = (node_id, port_name)
        multi = self._multi.get(key, [])
        if not multi:
            return self._data.get(key)
        values = [v for _, v in multi]
        if policy == PortAggregation.FIRST:
            return values[0]
        if policy == PortAggregation.CONCAT:
            return values
        if policy == PortAggregation.SUM:
            return sum(values)
        if policy == PortAggregation.DICT:
            return {src: val for src, val in multi}
        # SINGLE -- return the last written
        return values[-1] if values else self._data.get(key)

    def clear_multi(self, node_id: str, port_name: str) -> None:
        """Remove accumulated multi-edge entries for a key (needed between cycle iterations)."""
        self._multi.pop((node_id, port_name), None)

    def has_multi(self, node_id: str, port_name: str) -> bool:
        return bool(self._multi.get((node_id, port_name)))

    # --- snapshot ---

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of buffer data grouped by node_id."""
        result: Dict[str, Dict[str, Any]] = {}
        for (nid, pname), val in self._data.items():
            result.setdefault(nid, {})[pname] = val
        return result

    # --- bulk initialisation ---

    @classmethod
    def init_from_inputs(
        cls,
        spec: List[Dict[str, Any]],
        inputs: Dict[str, Any],
    ) -> "EdgeBuffers":
        """Create a buffer pre-populated from *spec* (exposed_inputs) and *inputs*."""
        buf = cls()
        for entry in spec:
            nid = entry.get("node_id") or entry.get("graph_id")
            pname = entry["port_name"]
            name = entry.get("name")
            candidates: List[str] = []
            if name is not None:
                candidates.append(name)
            candidates.append(pname)
            candidates.append(f"{nid}:{pname}")
            written = False
            # Prefer a non-None value so a stale ``{node}:{port}=None`` (e.g. from a prior run or
            # template) does not shadow a valid bare port name like ``ip_adapter_image``. This
            # matches :func:`yggdrasill.integrations.diffusers.run._merged_provides_input_for_node_port`.
            for k in candidates:
                if k in inputs and inputs[k] is not None:
                    buf.write(nid, pname, inputs[k])
                    written = True
                    break
            if not written:
                for k in candidates:
                    if k in inputs:
                        buf.write(nid, pname, inputs[k])
                        written = True
                        break
            if not written and (nid, pname) in inputs:
                buf.write(nid, pname, inputs[(nid, pname)])
        return buf
