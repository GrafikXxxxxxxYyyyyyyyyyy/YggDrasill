"""Lazy component: defer model loading until first use."""
from __future__ import annotations

from typing import Any, Optional


class LazyComponent:
    """Defers ModelStore load until resolve() is called."""

    __slots__ = ("_family", "_load_key", "_repo_id", "_variant", "_revision", "_torch_dtype", "_resolved")

    def __init__(
        self,
        family: str,
        load_key: str,
        repo_id: str,
        *,
        variant: str = "",
        revision: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
    ) -> None:
        self._family = family
        self._load_key = load_key
        self._repo_id = repo_id
        self._variant = variant
        self._revision = revision
        self._torch_dtype = torch_dtype
        self._resolved: Optional[Any] = None

    def resolve(self) -> Any:
        """Load from ModelStore (cached) and return the component."""
        if self._resolved is not None:
            return self._resolved
        from yggdrasill.integrations.diffusers.model_store import ModelStore
        ms = ModelStore.default()
        self._resolved = ms.load_component_by_key(
            self._family,
            self._load_key,
            self._repo_id,
            variant=self._variant,
            revision=self._revision,
            torch_dtype=self._torch_dtype,
        )
        return self._resolved

    def __repr__(self) -> str:
        return f"LazyComponent({self._family}/{self._load_key} from {self._repo_id})"


def resolve_if_lazy(val: Any) -> Any:
    """If val is LazyComponent, resolve and return; otherwise return val."""
    if isinstance(val, LazyComponent):
        return val.resolve()
    return val
