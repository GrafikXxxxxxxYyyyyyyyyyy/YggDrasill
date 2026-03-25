"""Centralized model store for Diffusers components.

Provides lazy loading, caching, shared component reuse, and device/dtype
management for all Diffusers-backed models used by YggDrasill nodes.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Dict, Optional, Tuple, Type

from yggdrasill.integrations.diffusers.types import ModelDType

logger = logging.getLogger(__name__)

# Hub weight layout: fp16 checkpoints live under variant ``fp16`` for SD1.5/SDXL.
_DTYPE_DEFAULT_FAMILIES = frozenset({"sd15", "sdxl", "flux"})


def _ensure_logging_handler() -> None:
    """Ensure loading progress is visible when root logger has no INFO handler."""
    if logger.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("[yggdrasill] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


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
        _ensure_logging_handler()
        key = self.cache_key(source, subfolder, cls.__name__)
        if not force_reload:
            cached = self.get(key)
            if cached is not None:
                logger.info("Component cache hit: %s from %s/%s", cls.__name__, source, subfolder or ".")
                return cached

        path = f"{source}/{subfolder}" if subfolder else source
        logger.info("Loading %s from %s ...", cls.__name__, path)
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

        component: Any = None
        try:
            component = cls.from_pretrained(source, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_file_not_found = (
                "does not appear to have" in err_str
                or "entry not found" in err_str
                or "404" in err_str
            )
            if not is_file_not_found:
                raise
            fallback_path = f"{source}/{subfolder}" if subfolder else source

            # Some community repos store components in repo root (no subfolder),
            # even when the canonical pipeline expects e.g. "vae/".
            # If we were asked to load from a subfolder and got a 404, retry without it.
            if component is None and subfolder:
                logger.info("Retrying %s from repo root (no subfolder) ...", cls.__name__)
                kwargs_root = {k: v for k, v in kwargs.items() if k != "subfolder"}
                try:
                    component = cls.from_pretrained(source, **kwargs_root)
                    subfolder = ""
                    fallback_path = source
                    kwargs = kwargs_root
                except Exception as root_err:
                    logger.info("Repo-root retry failed: %s", root_err)

            # Community ControlNet / single-file repos often ship only
            # ``diffusion_pytorch_model.safetensors`` (no ``.fp16.`` variant).
            if variant == "fp16" and use_safetensors:
                logger.info(
                    "Trying safetensors without variant for %s "
                    "(repo may only publish diffusion_pytorch_model.safetensors) ...",
                    fallback_path,
                )
                kwargs_nv = {k: v for k, v in kwargs.items() if k != "variant"}
                try:
                    component = cls.from_pretrained(source, **kwargs_nv)
                except Exception as nv_err:
                    logger.info("Safetensors without variant failed: %s", nv_err)

            # Try variant="fp16" when first attempt had no variant
            if component is None and not variant and use_safetensors:
                logger.info("Trying variant=fp16 for %s ...", fallback_path)
                kwargs_fp16 = dict(kwargs)
                kwargs_fp16["variant"] = "fp16"
                try:
                    component = cls.from_pretrained(source, **kwargs_fp16)
                except Exception as fp16_err:
                    logger.info("Variant fp16 failed: %s", fp16_err)
            # Try .bin if still no component
            if component is None:
                kwargs_bin = {k: v for k, v in kwargs.items() if k != "variant"}
                if use_safetensors:
                    logger.info("Loading .bin from %s (download may take a while) ...", fallback_path)
                    kwargs_bin["use_safetensors"] = False
                try:
                    component = cls.from_pretrained(source, **kwargs_bin)
                except Exception as retry_err:
                    logger.error("Fallback failed: %s", retry_err)
                    raise RuntimeError(
                        f"Failed to load {cls.__name__} from {fallback_path}. "
                        "Tried safetensors (requested variant), safetensors without variant, "
                        "variant=fp16 when applicable, and .bin."
                    ) from retry_err
        self.put(key, component)
        path = f"{source}/{subfolder}" if subfolder else source
        logger.info("Loaded %s from %s", cls.__name__, path)
        return component

    def load_component_by_key(
        self,
        family: str,
        load_key: str,
        repo_id: str,
        *,
        subfolder_override: Optional[str] = None,
        variant: str = "",
        revision: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        use_safetensors: Optional[bool] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        force_reload: bool = False,
    ) -> Optional[Any]:
        """Load a single component by (family, load_key). Cached per (repo_id, subfolder, cls_name).

        Each model loads separately via its own from_pretrained — no full pipeline.
        Scheduler (sd15) uses repo config for parity with diffusers pipeline.
        subfolder_override: when set, overrides the loader's default subfolder.
        """
        from yggdrasill.integrations.diffusers.component_loaders import (
            get_loader,
            load_scheduler_from_repo,
        )

        loader = get_loader(family, load_key)
        if loader is None:
            return None
        cls, subfolder = loader
        if subfolder_override is not None:
            subfolder = subfolder_override
        if cls is None and load_key == "scheduler":
            _ensure_logging_handler()
            key = self.cache_key(repo_id, subfolder, "Scheduler")
            if not force_reload:
                cached = self.get(key)
                if cached is not None:
                    logger.info("Component cache hit: Scheduler from %s/%s", repo_id, subfolder or ".")
                    return cached
            path = f"{repo_id}/{subfolder}" if subfolder else repo_id
            logger.info("Loading Scheduler from %s ...", path)
            component = load_scheduler_from_repo(repo_id, subfolder)
            self.put(key, component)
            logger.info("Loaded Scheduler from %s", path)
            return component
        use_safetensors_kw = (
            {} if use_safetensors is None
            else {"use_safetensors": use_safetensors}
        )
        return self.load_component(
            cls,
            repo_id,
            subfolder=subfolder,
            variant=variant,
            revision=revision,
            torch_dtype=torch_dtype,
            extra_kwargs=extra_kwargs,
            force_reload=force_reload,
            **use_safetensors_kw,
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
        use_safetensors: Optional[bool] = None,
        pretrained_map: Optional[Dict[str, str]] = None,
        subfolder_map: Optional[Dict[str, str]] = None,
        variant_map: Optional[Dict[str, str]] = None,
        extra_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Load only the requested components. Each loads separately; results are cached.

        pretrained_map: optional {load_key: repo_id} to load each key from a different repo.
        subfolder_map: optional {load_key: subfolder} to override subfolder per key.
        variant_map: optional {load_key: variant} to override variant per key.

        For families ``sd15``, ``sdxl``, and ``flux``, if *torch_dtype* is omitted it
        follows :class:`~yggdrasill.integrations.diffusers.family_registry.FamilySpec`
        (SDXL/SD15 → ``float16``, FLUX → ``bfloat16``). If *variant* is omitted and
        the resolved dtype is ``float16``, *variant* defaults to ``fp16`` so Hub fp16
        weights are used (matches :func:`~yggdrasill.integrations.diffusers.factory.build_sdxl_pipeline`).
        """
        _ensure_logging_handler()
        if family in _DTYPE_DEFAULT_FAMILIES:
            from yggdrasill.integrations.diffusers.family_registry import get_family_spec

            fspec = get_family_spec(family)
            torch_mod = _import_torch()
            dm = {
                "float16": torch_mod.float16,
                "float32": torch_mod.float32,
                "bfloat16": torch_mod.bfloat16,
            }
            if torch_dtype is None:
                torch_dtype = dm.get(fspec.torch_dtype_default, torch_mod.float16)
            if not variant and torch_dtype == torch_mod.float16:
                variant = "fp16"
        result: Dict[str, Any] = {}
        for key in load_keys:
            key_repo = (pretrained_map or {}).get(key, repo_id)
            key_subfolder = (subfolder_map or {}).get(key) if subfolder_map else None
            key_variant = (variant_map or {}).get(key, variant)
            key_extra = (extra_kwargs_map or {}).get(key) if extra_kwargs_map else None
            if not result and not pretrained_map:
                logger.info("Loading components from %s: %s", key_repo, ", ".join(load_keys))
            comp = self.load_component_by_key(
                family,
                key,
                key_repo,
                subfolder_override=key_subfolder,
                variant=key_variant,
                revision=revision,
                torch_dtype=torch_dtype,
                use_safetensors=use_safetensors,
                extra_kwargs=key_extra,
            )
            if comp is not None:
                result[key] = comp
        if result:
            logger.info("Loaded %d/%d components: %s", len(result), len(load_keys), ", ".join(result.keys()))
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
