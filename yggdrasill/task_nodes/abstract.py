"""Seven abstract task-node roles.

Each inherits both AbstractBaseBlock (material) and AbstractGraphNode (ideal),
following the dual-inheritance principle: one object, two origins.
Port contracts follow the canonical specification (canon 02).
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from yggdrasill.foundation.block import AbstractBaseBlock
from yggdrasill.foundation.node import AbstractGraphNode
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.roles import Role


class _TaskNodeBase(AbstractBaseBlock, AbstractGraphNode):
    """Shared boilerplate for all seven roles."""

    _role: Role

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        AbstractBaseBlock.__init__(self, block_id=block_id, config=config)
        AbstractGraphNode.__init__(self, node_id=node_id)

    @property
    def role(self) -> Role:
        return self._role


# ---------------------------------------------------------------------------
# 1. Backbone -- core prediction step (canon 02 §4)
# ---------------------------------------------------------------------------

class AbstractBackbone(_TaskNodeBase):
    """Primary transformation: one prediction step (e.g., UNet, Transformer)."""

    _role = Role.BACKBONE

    @property
    def block_type(self) -> str:
        return "backbone"

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, PortType.TENSOR),
            Port("timestep", PortDirection.IN, PortType.TENSOR),
            Port("condition", PortDirection.IN, PortType.ANY, optional=True),
            Port("pred", PortDirection.OUT, PortType.TENSOR),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 2. Injector -- injects conditioning into the backbone (canon 02 §5)
# ---------------------------------------------------------------------------

class AbstractInjector(_TaskNodeBase):
    """Adapts the backbone's computation (e.g. LoRA): tied to weights / internal behavior.

    Use :class:`AbstractConjector` for **standalone conditioning encoders** (text, reference
    image, etc.) that do not alter backbone weights or architecture the way LoRA does.
    """

    _role = Role.INJECTOR

    @property
    def block_type(self) -> str:
        return "injector"

    def declare_ports(self) -> List[Port]:
        return [
            Port("condition", PortDirection.IN, PortType.ANY),
            Port("hidden", PortDirection.IN, PortType.TENSOR, optional=True),
            Port("adapted", PortDirection.OUT, PortType.TENSOR),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 3. Conjector -- stands beside the backbone, supplies condition (canon 02 §6)
# ---------------------------------------------------------------------------

class AbstractConjector(_TaskNodeBase):
    """Supplies conditioning beside the backbone (text encoder, IP-Adapter image encoder, …)."""

    _role = Role.CONJECTOR

    @property
    def block_type(self) -> str:
        return "conjector"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("condition", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 4. Inner Module -- inside the loop, one transition step (canon 02 §7)
# ---------------------------------------------------------------------------

class AbstractInnerModule(_TaskNodeBase):
    """Internal processing module (e.g., DDIM solver step)."""

    _role = Role.INNER_MODULE

    @property
    def block_type(self) -> str:
        return "inner_module"

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, PortType.TENSOR),
            Port("timestep", PortDirection.IN, PortType.TENSOR),
            Port("pred", PortDirection.IN, PortType.TENSOR),
            Port("control", PortDirection.IN, PortType.ANY, optional=True),
            Port("next_latent", PortDirection.OUT, PortType.TENSOR),
            Port("next_timestep", PortDirection.OUT, PortType.TENSOR, optional=True),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 5. Outer Module -- before/after the loop (canon 02 §8)
# ---------------------------------------------------------------------------

class AbstractOuterModule(_TaskNodeBase):
    """External module: before entry to loop or after exit (e.g., scheduler)."""

    _role = Role.OUTER_MODULE

    @property
    def block_type(self) -> str:
        return "outer_module"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY, optional=True),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 6. Helper -- auxiliary utility (canon 02 §10)
# ---------------------------------------------------------------------------

class AbstractHelper(_TaskNodeBase):
    """Utility node (RAG, file I/O, API calls, etc.).

    Training also maps here without new roles: optimiser steps, LR schedulers,
    grad-scaler orchestration, checkpoint hooks (see ``training/*`` block types).
    """

    _role = Role.HELPER

    @property
    def block_type(self) -> str:
        return "helper"

    def declare_ports(self) -> List[Port]:
        return [
            Port("query", PortDirection.IN, PortType.ANY),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 7. Converter -- data format conversion (canon 02 §9)
# ---------------------------------------------------------------------------

class AbstractConverter(_TaskNodeBase):
    """Converts between representations (e.g., VAE encode/decode)."""

    _role = Role.CONVERTER

    @property
    def block_type(self) -> str:
        return "converter"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...
