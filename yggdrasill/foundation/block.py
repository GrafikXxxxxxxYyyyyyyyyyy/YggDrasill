from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, TypeVar

_B = TypeVar("_B", bound="AbstractBaseBlock")


class AbstractBaseBlock(ABC):
    """Material origin: identity, data, computation.

    Does not declare ports -- that is the responsibility of the graph node.
    Provides forward(inputs)->outputs, state_dict/load_state_dict, and
    identity attributes (block_type, block_id, config).
    """

    def __init__(
        self,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._block_id = block_id or f"{type(self).__name__}_{id(self):x}"
        self._config: Dict[str, Any] = dict(config or {})
        self._training = True
        self._frozen = False

    @property
    def block_type(self) -> str:
        return type(self).__name__

    @property
    def block_id(self) -> str:
        return self._block_id

    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy()

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute outputs from inputs. Key names are defined by the node's ports."""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict, auto-aggregating sub-block state with dotted prefix."""
        result: Dict[str, Any] = {}
        for name, sub in self.get_sub_blocks().items():
            for k, v in sub.state_dict().items():
                result[f"{name}.{k}"] = v
        return result

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if strict:
            own_keys = set(self.state_dict().keys())
            extra = set(state.keys()) - own_keys
            missing = own_keys - set(state.keys())
            if extra or missing:
                raise KeyError(
                    f"state_dict mismatch: extra={extra}, missing={missing}"
                )
        sub_blocks = self.get_sub_blocks()
        if sub_blocks:
            sub_states: Dict[str, Dict[str, Any]] = {n: {} for n in sub_blocks}
            for key, val in state.items():
                prefix, dot, suffix = key.partition(".")
                if dot and prefix in sub_blocks:
                    sub_states[prefix][suffix] = val
            for name, sub in sub_blocks.items():
                if sub_states[name]:
                    sub.load_state_dict(sub_states[name], strict=strict)

    def get_sub_blocks(self) -> Dict[str, "AbstractBaseBlock"]:
        return {}

    # --- train / eval ---

    @property
    def training(self) -> bool:
        return self._training

    def train(self: _B, mode: bool = True) -> _B:
        self._training = mode
        for sub in self.get_sub_blocks().values():
            sub.train(mode)
        return self

    def eval(self: _B) -> _B:
        return self.train(False)

    # --- freeze / unfreeze ---

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self: _B) -> _B:
        self._frozen = True
        for sub in self.get_sub_blocks().values():
            sub.freeze()
        return self

    def unfreeze(self: _B) -> _B:
        self._frozen = False
        for sub in self.get_sub_blocks().values():
            sub.unfreeze()
        return self

    # --- device placement ---

    def to(self: _B, device: Any) -> _B:
        """Move block to *device* (no-op by default; override for GPU-backed blocks)."""
        for sub in self.get_sub_blocks().values():
            sub.to(device)
        return self

    # --- trainable parameters ---

    def trainable_parameters(self) -> Iterator[Any]:
        """Yield trainable parameters (empty by default; override for GPU-backed blocks)."""
        return iter(())

    @property
    def supports_backward(self) -> bool:
        """Whether this block participates in autograd when training (loss flows through it)."""
        return False

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serialisable config sufficient to recreate this block via the registry.

        For task-nodes (Block+Node), node_id is included automatically.
        """
        cfg: Dict[str, Any] = {
            "block_type": self.block_type,
            "block_id": self.block_id,
            "config": self.config,
        }
        if hasattr(self, "_node_id"):
            cfg["node_id"] = self._node_id
        return cfg
