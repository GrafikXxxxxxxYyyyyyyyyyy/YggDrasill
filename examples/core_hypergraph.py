"""Minimal core-only example for the stable YggDrasill API surface.

Run:
    python examples/core_hypergraph.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortDirection, PortType


class PrefixNode(AbstractBaseBlock, AbstractGraphNode):
    def __init__(self, node_id: str, prefix: str) -> None:
        AbstractBaseBlock.__init__(self)
        AbstractGraphNode.__init__(self, node_id=node_id)
        self.prefix = prefix

    @property
    def block_type(self) -> str:
        return "example/prefix"

    def declare_ports(self) -> List[Port]:
        return [
            Port("text", PortDirection.IN, PortType.TEXT),
            Port("result", PortDirection.OUT, PortType.TEXT),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"{self.prefix}{inputs['text']}"}


def build_graph() -> Hypergraph:
    graph = Hypergraph(graph_id="core_example")
    graph.add_node("step_1", PrefixNode("step_1", "Hello, "))
    graph.add_node("step_2", PrefixNode("step_2", "YggDrasill says: "))
    graph.add_edge(Edge("step_1", "result", "step_2", "text"))
    graph.expose_input("step_1", "text", "name")
    graph.expose_output("step_2", "result", "message")
    return graph


if __name__ == "__main__":
    result = build_graph().run({"name": "core runtime"}, validate_before=False)
    print(result["message"])
