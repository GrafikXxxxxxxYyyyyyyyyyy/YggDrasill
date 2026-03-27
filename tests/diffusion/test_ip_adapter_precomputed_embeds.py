"""Precomputed IP-Adapter embeddings: run() merge + cond-only helper for Diffusers pipeline tensors."""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_pipeline_ip_adapter_embeds_cond_only_strips_uncond_half() -> None:
    pytest.importorskip("torch")
    import torch

    from yggdrasill.integrations.diffusers.common.ip_adapter_embeds import (
        pipeline_ip_adapter_embeds_cond_only,
    )

    neg = torch.zeros(1, 1, 16)
    pos = torch.ones(1, 1, 16)
    packed = torch.cat([neg, pos], dim=0)
    out = pipeline_ip_adapter_embeds_cond_only([packed])
    assert len(out) == 1
    assert torch.equal(out[0], pos)


def test_run_diffusion_merges_ip_adapter_image_embeds_dict() -> None:
    pytest.importorskip("torch")
    import torch

    import yggdrasill.integrations.diffusers  # noqa: F401

    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers import contracts as C
    from yggdrasill.integrations.diffusers.run import run as run_diffusion

    captured: list = []

    def cap(structure, inputs, **kwargs):
        captured.append(dict(inputs))
        return {}

    g = Hypergraph()
    t = torch.zeros(1, 4)
    with patch("yggdrasill.engine.executor.run", cap):
        run_diffusion(g, {}, ip_adapter_image_embeds={"slot_a": t}, wrap_output=False)

    assert captured
    assert captured[0].get(f"slot_a:{C.PORT_IP_ADAPTER_IMAGE_EMBEDS}") is t
