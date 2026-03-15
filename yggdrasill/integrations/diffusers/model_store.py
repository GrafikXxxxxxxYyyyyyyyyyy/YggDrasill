"""Centralized model store for Diffusers components.

Provides lazy loading, caching, shared component reuse, and device/dtype
management for all Diffusers-backed models used by YggDrasill nodes.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple, Type

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
                return cached

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
        return component

    def load_pipeline_components(
        self,
        repo_id: str,
        *,
        variant: str = "",
        revision: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load all components from a single pipeline repo.

        Returns a dict of component name -> loaded object, suitable for
        passing to individual node constructors.
        """
        _import_diffusers()
        from diffusers import DiffusionPipeline

        dtype = torch_dtype or self.get_torch_dtype()
        kwargs: Dict[str, Any] = {}
        if variant:
            kwargs["variant"] = variant
        if revision:
            kwargs["revision"] = revision
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        pipe = DiffusionPipeline.from_pretrained(repo_id, **kwargs)

        components: Dict[str, Any] = {}
        for attr in ("vae", "text_encoder", "text_encoder_2",
                      "tokenizer", "tokenizer_2", "unet", "transformer",
                      "scheduler", "safety_checker", "feature_extractor",
                      "image_encoder", "controlnet"):
            val = getattr(pipe, attr, None)
            if val is not None:
                components[attr] = val
                key = self.cache_key(repo_id, "", type(val).__name__)
                self.put(key, val)

        return components

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
