"""Yggdrasill -- hypergraph framework from block to universe."""

__version__ = "0.1.0"

from yggdrasill.hypergraph.structure import Hypergraph  # noqa: F401 — avoid loading engine (training/torch) on bare import

__all__ = ["__version__", "Hypergraph", "DiffusionGraphBuilder"]


def __getattr__(name: str):
    if name == "DiffusionGraphBuilder":
        from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder

        return DiffusionGraphBuilder
    raise AttributeError(f"module 'yggdrasill' has no attribute {name!r}")
