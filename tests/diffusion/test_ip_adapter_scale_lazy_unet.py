"""IP-Adapter scale must apply to the resolved UNet when the node still holds a LazyComponent."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn.functional as F

from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import _set_ip_adapter_scale_on_unet
from yggdrasill.integrations.diffusers.lazy_component import LazyComponent


@pytest.mark.skipif(
    not hasattr(F, "scaled_dot_product_attention"),
    reason="IPAdapterAttnProcessor2_0 requires scaled_dot_product_attention",
)
def test_set_ip_adapter_scale_resolves_lazy_unet_wrapper() -> None:
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0

    proc = IPAdapterAttnProcessor2_0(320, 1024, num_tokens=(4,))
    inner = SimpleNamespace(
        down_blocks=(),
        up_blocks=(),
        config=SimpleNamespace(layers_per_block=2),
        attn_processors={"mid.block.1.attentions.0.transformer_blocks.0.attn2.processor": proc},
    )
    lazy = LazyComponent("sdxl", "unet", "dummy/repo")
    lazy._resolved = inner

    _set_ip_adapter_scale_on_unet(lazy, 0.35)
    assert proc.scale == [0.35]
