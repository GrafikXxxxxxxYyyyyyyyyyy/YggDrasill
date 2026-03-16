"""Backward-compatible re-export. Hypergraph lives in hypergraph/structure."""
from __future__ import annotations

from yggdrasill.hypergraph.structure import Hypergraph, _resolve_config_ref

__all__ = ["Hypergraph", "_resolve_config_ref"]
