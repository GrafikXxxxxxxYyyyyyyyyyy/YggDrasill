"""Merge utilities for adapter residuals (ControlNet, IP-Adapter, etc.)."""
from __future__ import annotations

from typing import Any


def merge_residuals(value: Any) -> Any:
    """Merge residuals from multiple adapters.

    When ``CONCAT`` aggregation is used, the executor delivers a list of
    residual tuples (one per adapter).  This helper element-wise sums
    them into a single residual tuple suitable for the UNet/Transformer.
    If the value is not a list (single adapter), it is returned as-is.
    """
    if not isinstance(value, list):
        return value
    if len(value) == 1:
        return value[0]
    import torch
    merged = value[0]
    if isinstance(merged, (list, tuple)):
        merged = list(merged)
        for extra in value[1:]:
            for i, t in enumerate(extra):
                merged[i] = merged[i] + t
        return tuple(merged)
    for extra in value[1:]:
        merged = merged + extra
    return merged
