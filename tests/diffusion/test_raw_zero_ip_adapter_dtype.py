"""raw_zero_ip_adapter_image_embeds_for_unet must match fp16 UNet for Plus proj_in."""
from __future__ import annotations

import pytest


def test_raw_zero_matches_unet_dtype_when_dtype_none() -> None:
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from yggdrasill.integrations.diffusers.common.ip_adapter_embeds import (
        raw_zero_ip_adapter_image_embeds_for_unet,
    )

    class FakeProj(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_projection_layers = nn.ModuleList([nn.Linear(8, 8)])

    class FakeUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_hid_proj = FakeProj()

    u = FakeUNet().to(dtype=torch.float16)
    out = raw_zero_ip_adapter_image_embeds_for_unet(
        u, 1, device=torch.device("cpu"), dtype=None,
    )
    assert len(out) == 1
    assert out[0].dtype == torch.float16


def test_raw_zero_plus_shape_runs_through_multi_ip_projection() -> None:
    """Inactive Plus placeholders must be 4D so IPAdapterPlusImageProjectionBlock gets 3D x."""
    pytest.importorskip("diffusers")
    import torch
    import torch.nn as nn

    from diffusers.models.embeddings import (
        IPAdapterPlusImageProjection,
        MultiIPAdapterImageProjection,
    )

    from yggdrasill.integrations.diffusers.common.ip_adapter_embeds import (
        format_ip_adapter_image_embeds,
        raw_zero_ip_adapter_image_embeds_for_unet,
    )

    class FakeUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_hid_proj = MultiIPAdapterImageProjection(
                [
                    IPAdapterPlusImageProjection(
                        embed_dims=32,
                        output_dims=64,
                        hidden_dims=48,
                        depth=1,
                        dim_head=8,
                        heads=2,
                        num_queries=4,
                        ffn_ratio=2,
                    )
                ]
            )

    u = FakeUNet()
    raw = raw_zero_ip_adapter_image_embeds_for_unet(
        u, 1, device=torch.device("cpu"), dtype=torch.float32
    )
    assert raw[0].shape == (1, 1, 1, 32)
    packed = format_ip_adapter_image_embeds(
        raw,
        device=torch.device("cpu"),
        dtype=torch.float32,
        do_classifier_free_guidance=True,
    )
    assert packed[0].shape == (2, 1, 1, 32)
    proj_out = u.encoder_hid_proj(packed)
    assert len(proj_out) == 1
    assert proj_out[0].shape[0] == 2
