"""Regression: run_diffusion must not pop ip_adapter_conditioning_scale before patched Hypergraph.run.

If run_kw loses the scale, the patch applies {"default": 1.0} and user scale is ignored."""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_run_diffusion_applies_ip_adapter_scale_only_via_patched_graph_run() -> None:
    pytest.importorskip("torch")
    import yggdrasill.integrations.diffusers  # noqa: F401 — patches Hypergraph.run

    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers.run import _inject_ip_adapter_scale
    from yggdrasill.integrations.diffusers.run import run as run_diffusion

    g = Hypergraph()
    scale_maps: list[dict] = []

    def capture(graph: object, scale_map: dict, merged: dict) -> None:
        scale_maps.append(dict(scale_map))
        return _inject_ip_adapter_scale(graph, scale_map, merged)

    with patch("yggdrasill.engine.executor.run", return_value={}):
        with patch(
            "yggdrasill.integrations.diffusers.run._inject_ip_adapter_scale",
            side_effect=capture,
        ):
            run_diffusion(
                g,
                None,
                ip_adapter_conditioning_scale={"IPAdapter": 0.33},
            )

    assert len(scale_maps) == 1, "expected a single inject (patched run only), not run_diffusion + patch"
    assert scale_maps[0].get("IPAdapter") == 0.33
