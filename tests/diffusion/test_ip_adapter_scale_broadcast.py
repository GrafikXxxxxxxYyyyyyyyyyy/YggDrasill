"""IP-Adapter UNet scale: single float must broadcast to each entry in processor.scale (SD1.5 / multi-token)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn.functional as F

from yggdrasill.integrations.diffusers.adapters.ip_adapter_loader import _set_ip_adapter_scale_on_unet


@pytest.mark.skipif(
    not hasattr(F, "scaled_dot_product_attention"),
    reason="IPAdapterAttnProcessor2_0 requires scaled_dot_product_attention",
)
def test_set_ip_adapter_scale_broadcasts_one_float_to_multi_token_slots() -> None:
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0

    proc = IPAdapterAttnProcessor2_0(320, 1024, num_tokens=(4, 4))
    assert len(proc.scale) == 2
    unet = SimpleNamespace(
        down_blocks=(),
        up_blocks=(),
        config=SimpleNamespace(layers_per_block=2),
        attn_processors={"mid.block.1.attentions.0.transformer_blocks.0.attn2.processor": proc},
    )
    _set_ip_adapter_scale_on_unet(unet, [0.0])
    assert proc.scale == [0.0, 0.0]
