from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, Union

from yggdrasill.foundation.block import AbstractBaseBlock


Factory = Union[Type[AbstractBaseBlock], Callable[..., AbstractBaseBlock]]


class BlockRegistry:
    """Maps block_type strings to factories (classes or callables) that produce blocks."""

    _global: Optional["BlockRegistry"] = None

    @classmethod
    def global_registry(cls) -> "BlockRegistry":
        if cls._global is None:
            cls._global = cls()
        return cls._global

    @classmethod
    def reset_global(cls) -> None:
        """Reset the global registry (useful in tests)."""
        cls._global = None

    def __init__(self) -> None:
        self._factories: Dict[str, Factory] = {}

    def register(self, block_type: str, factory: Factory) -> None:
        if not block_type or not block_type.strip():
            raise ValueError("block_type must be a non-empty string")
        self._factories[block_type.strip()] = factory

    def build(self, config: Dict[str, Any]) -> AbstractBaseBlock:
        block_type = config.get("block_type") or config.get("type")
        if not block_type:
            raise KeyError(
                "config must contain 'block_type' or 'type'. "
                f"Registered types: {sorted(self._factories.keys())}"
            )
        block_type = block_type.strip()
        factory = self._factories.get(block_type)
        if factory is None:
            raise KeyError(
                f"Unknown block_type '{block_type}'. "
                f"Registered types: {sorted(self._factories.keys())}"
            )
        _meta_keys = ("block_type", "type", "schema_version")
        rest = {k: v for k, v in config.items() if k not in _meta_keys}
        node_id = rest.pop("node_id", None)
        block_id = rest.pop("block_id", None)
        # Flatten nested "config" key from get_config() round-trip
        if "config" in rest and isinstance(rest["config"], dict):
            inner = rest.pop("config")
            rest.update(inner)
        # Pop trainable -- it's graph-level metadata, not a constructor arg
        rest.pop("trainable", None)

        kwargs: Dict[str, Any] = {}
        if block_id is not None:
            kwargs["block_id"] = block_id
        if node_id is not None:
            kwargs["node_id"] = node_id
        if rest:
            kwargs["config"] = rest

        return factory(**kwargs)

    def get(self, block_type: str) -> Optional[Factory]:
        return self._factories.get(block_type.strip())

    def __contains__(self, block_type: str) -> bool:
        return block_type.strip() in self._factories

    @property
    def registered_types(self) -> list[str]:
        return sorted(self._factories.keys())


def register_block(
    block_type: str,
    registry: Optional[BlockRegistry] = None,
) -> Callable[[Type[AbstractBaseBlock]], Type[AbstractBaseBlock]]:
    """Decorator: register a class under *block_type* in the given (or global) registry."""

    def decorator(cls: Type[AbstractBaseBlock]) -> Type[AbstractBaseBlock]:
        (registry or BlockRegistry.global_registry()).register(block_type, cls)
        return cls

    return decorator
