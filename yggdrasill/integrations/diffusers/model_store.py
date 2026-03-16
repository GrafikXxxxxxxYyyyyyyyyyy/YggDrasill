"""Centralized model store for Diffusers components.

Provides lazy loading, caching, shared component reuse, and device/dtype
management for all Diffusers-backed models used by YggDrasill nodes.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple, Type

logger = logging.getLogger(__name__)

from yggdrasill.diffusion.types import ModelDType


def _import_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for Diffusers integration. "
            "Install with: pip install torch"
        ) from exc


def _import_diffusers() -> Any:
    try:
        import diffusers
        return diffusers
    except ImportError as exc:
        raise ImportError(
            "Diffusers is required for this integration. "
            "Install with: pip install diffusers[torch]"
        ) from exc


class ModelStore:
    """Thread-safe cache for loaded Diffusers components.

    Components are keyed by (source, subfolder, component_class_name) and
    reused across graph nodes to avoid duplicate memory usage.
    """

    _instance: Optional["ModelStore"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, str], Any] = {}
        self._device: str = "cpu"
        self._dtype: Optional[ModelDType] = None

    @classmethod
    def default(cls) -> "ModelStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value

    @property
    def dtype(self) -> Optional[ModelDType]:
        return self._dtype

    @dtype.setter
    def dtype(self, value: Optional[ModelDType]) -> None:
        self._dtype = value

    def get_torch_dtype(self) -> Any:
        if self._dtype is None:
            return None
        return self._dtype.to_torch()

    def cache_key(
        self,
        source: str,
        subfolder: str = "",
        cls_name: str = "",
    ) -> Tuple[str, str, str]:
        return (source, subfolder, cls_name)

    def get(self, key: Tuple[str, str, str]) -> Optional[Any]:
        return self._cache.get(key)

    def put(self, key: Tuple[str, str, str], component: Any) -> None:
        self._cache[key] = component

    def load_component(
        self,
        cls: Type[Any],
        source: str,
        subfolder: str = "",
        *,
        variant: str = "",
        revision: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        use_safetensors: bool = True,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        force_reload: bool = False,
    ) -> Any:
        """Load a Diffusers component with caching.

        Uses ``cls.from_pretrained(source, subfolder=..., ...)`` and caches
        the result under ``(source, subfolder, cls.__name__)``.
        """
        key = self.cache_key(source, subfolder, cls.__name__)
        if not force_reload:
            cached = self.get(key)
            if cached is not None:
                logger.info("Component cache hit: %s %s/%s", cls.__name__, source, subfolder or ".")
                return cached

        logger.info("Loading %s from %s (subfolder=%s)", cls.__name__, source, subfolder or ".")
        kwargs: Dict[str, Any] = {}
        if subfolder:
            kwargs["subfolder"] = subfolder
        if variant:
            kwargs["variant"] = variant
        if revision:
            kwargs["revision"] = revision

        dtype = torch_dtype or self.get_torch_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        if use_safetensors and hasattr(cls, "from_pretrained"):
            kwargs["use_safetensors"] = use_safetensors

        if extra_kwargs:
            kwargs.update(extra_kwargs)

        component = cls.from_pretrained(source, **kwargs)
        self.put(key, component)
        logger.info("Loaded %s from %s", cls.__name__, source)
        return component

    def load_component_by_key(
        self,
        family: str,
        load_key: str,
        repo_id: str,
        *,
        variant: str = "",
        revision: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        force_reload: bool = False,
    ) -> Optional[Any]:
        """Load a single component by (family, load_key). Cached per (repo_id, subfolder, cls_name).

        Each model loads separately via its own from_pretrained — no full pipeline.
        Scheduler (sd15) uses repo config for parity with diffusers pipeline.
        """
        from yggdrasill.integrations.diffusers.component_loaders import (
            get_loader,
            load_scheduler_from_repo,
        )

        loader = get_loader(family, load_key)
        if loader is None:
            return None
        cls, subfolder = loader
        if cls is None and load_key == "scheduler":
            key = self.cache_key(repo_id, subfolder, "Scheduler")
            if not force_reload:
                cached = self.get(key)
                if cached is not None:
                    logger.info("Component cache hit: Scheduler %s/%s", repo_id, subfolder or ".")
                    return cached
            logger.info("Loading scheduler from %s (subfolder=%s)", repo_id, subfolder or ".")
            component = load_scheduler_from_repo(repo_id, subfolder)
            self.put(key, component)
            return component
        return self.load_component(
            cls,
            repo_id,
            subfolder=subfolder,
            variant=variant,
            revision=revision,
            torch_dtype=torch_dtype,
            force_reload=force_reload,
        )

    def load_components_by_keys(
        self,
        family: str,
        load_keys: list,
        repo_id: str,
        *,
        variant: str = "",
        revision: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load only the requested components. Each loads separately; results are cached."""
        logger.info("Loading components %s (family=%s, repo=%s)", load_keys, family, repo_id)
        result: Dict[str, Any] = {}
        for key in load_keys:
            comp = self.load_component_by_key(
                family,
                key,
                repo_id,
                variant=variant,
                revision=revision,
                torch_dtype=torch_dtype,
            )
            if comp is not None:
                result[key] = comp
        return result

    def move_to_device(self, component: Any, device: Optional[str] = None) -> Any:
        """Move a component to the target device if it supports .to()."""
        target = device or self._device
        if hasattr(component, "to") and callable(component.to):
            component.to(target)
        return component

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
