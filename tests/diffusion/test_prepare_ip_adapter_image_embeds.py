"""prepare_ip_adapter_image_embeds(graph) without Diffusers pipeline."""
from __future__ import annotations

import pytest

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.adapters.ip_adapter import (
    IPAdapterNode,
    align_vision_hidden_states_for_ip_adapter_plus,
)
from yggdrasill.integrations.diffusers.common.ip_adapter_embeds import (
    format_ip_adapter_image_embeds,
    prepare_ip_adapter_image_embeds,
)

from tests.diffusion.conftest import FakeFeatureExtractor, FakeImageEncoder, FakeTensor, requires_torch


def test_prepare_raises_when_no_ip_adapter_node() -> None:
    g = Hypergraph()
    g.add_node("u", object())
    with pytest.raises(ValueError, match="no adapter/ip_adapter"):
        prepare_ip_adapter_image_embeds(g, "x.png")


@requires_torch
def test_prepare_uses_graph_node_encoder() -> None:
    import torch

    g = Hypergraph()
    node = IPAdapterNode(
        "IP",
        image_encoder=FakeImageEncoder(),
        feature_extractor=FakeFeatureExtractor(),
    )
    g.add_node("IP", node)

    out = prepare_ip_adapter_image_embeds(
        g,
        FakeTensor((1, 3, 224, 224)),
        num_images_per_prompt=2,
    )
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape[0] == 2


@requires_torch
def test_prepare_node_id_subset() -> None:
    g = Hypergraph()
    g.add_node("A", IPAdapterNode("A", image_encoder=FakeImageEncoder(), feature_extractor=FakeFeatureExtractor()))
    g.add_node("B", IPAdapterNode("B", image_encoder=FakeImageEncoder(), feature_extractor=FakeFeatureExtractor()))
    out = prepare_ip_adapter_image_embeds(
        g,
        FakeTensor((1, 3, 224, 224)),
        node_id="B",
    )
    assert len(out) == 1


@requires_torch
def test_encode_ip_adapter_image_matches_forward() -> None:
    node = IPAdapterNode(
        "IP",
        image_encoder=FakeImageEncoder(),
        feature_extractor=FakeFeatureExtractor(),
    )
    img = FakeTensor((1, 3, 224, 224))
    a = node.encode_ip_adapter_image(img)
    b = node.forward({C.PORT_IP_ADAPTER_IMAGE: img})[C.PORT_IMAGE_EMBEDS]
    assert a.shape == b.shape


@requires_torch
def test_align_plus_hidden_projects_when_unet_expects_contrastive_dim() -> None:
    """Wide ViT hidden states → ``visual_projection`` when target matches ``proj_in`` width."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    class Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=1664, projection_dim=1280)
            self.visual_projection = nn.Linear(1664, 1280, bias=False)

    enc = Enc()
    h = torch.randn(2, 5, 1664)
    out = align_vision_hidden_states_for_ip_adapter_plus(
        enc, h, token_embed_dim=1280,
    )
    assert tuple(out.shape) == (2, 5, 1280)


@requires_torch
def test_align_plus_hidden_identity_when_already_target_width() -> None:
    """SD1.5-style ViT-H: tokens are already ``proj_in.in_features`` wide."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    class Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=1280, projection_dim=1024)
            self.visual_projection = nn.Linear(1280, 1024, bias=False)

    enc = Enc()
    h = torch.randn(1, 4, 1280)
    out = align_vision_hidden_states_for_ip_adapter_plus(
        enc, h, token_embed_dim=1280,
    )
    assert out is h
    assert tuple(out.shape) == (1, 4, 1280)


@requires_torch
def test_align_plus_hidden_no_token_dim_no_projection() -> None:
    """Without UNet-derived width, do not guess (avoids SD1.5 vs SDXL ambiguity)."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    class Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=1664, projection_dim=1280)
            self.visual_projection = nn.Linear(1664, 1280, bias=False)

    enc = Enc()
    h = torch.randn(1, 3, 1664)
    out = align_vision_hidden_states_for_ip_adapter_plus(enc, h, token_embed_dim=None)
    assert out is h


@requires_torch
def test_encode_ip_adapter_plus_hidden_states_shape() -> None:
    """Plus-style weights need CFG-packed [2, n_img, seq, dim] (Diffusers encode_image)."""
    import torch
    from types import SimpleNamespace

    class FE:
        def __call__(self, images, return_tensors="pt"):
            n = len(images) if isinstance(images, list) else 1
            return SimpleNamespace(pixel_values=torch.zeros(n, 3, 224, 224))

    enc = FakeImageEncoder()
    enc.dtype = torch.float32  # type: ignore[assignment]
    enc.device = torch.device("cpu")  # type: ignore[assignment]
    node = IPAdapterNode(
        "IP",
        config={"ip_adapter_use_hidden_states": True},
        image_encoder=enc,
        feature_extractor=FE(),
    )
    from PIL import Image

    out = node.encode_ip_adapter_image(
        [Image.new("RGB", (4, 4), 0), Image.new("RGB", (4, 4), 1)],
    )
    assert tuple(out.shape) == (2, 2, 16, 1024)


@requires_torch
def test_format_ip_adapter_image_embeds_multi_ref_one_slot() -> None:
    """Match diffusers MultiIPAdapterImageProjection: [N, D] → [1, N, D], not [N, 1, D]."""
    import torch

    dev = torch.device("cpu")
    emb = torch.randn(2, 768)
    out = format_ip_adapter_image_embeds(
        emb, device=dev, dtype=None, do_classifier_free_guidance=False,
    )
    assert len(out) == 1
    assert tuple(out[0].shape) == (1, 2, 768)

    out_cfg = format_ip_adapter_image_embeds(
        emb, device=dev, dtype=None, do_classifier_free_guidance=True,
    )
    assert tuple(out_cfg[0].shape) == (2, 2, 768)


@requires_torch
def test_prepare_plus_preserves_encoder_uncond_not_zeros_like() -> None:
    """Stripping Plus to cond-only made format_ip_adapter_image_embeds pad with zeros_like — wrong."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    from yggdrasill.engine.structure import Hypergraph

    class FE:
        def __call__(self, images, return_tensors="pt"):
            im = images[0]
            bright = False
            if hasattr(im, "getpixel"):
                bright = im.getpixel((0, 0))[:3] != (0, 0, 0)
            pv = torch.ones if bright else torch.zeros
            return SimpleNamespace(pixel_values=pv(1, 3, 224, 224))

    class Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dtype = torch.float32
            self.device = torch.device("cpu")

        def forward(self, pixel_values, output_hidden_states=False):
            if output_hidden_states:
                # Uncond path must not equal zeros_like(cond): models emit non-zero features on zero input.
                is_zero = bool(pixel_values.abs().sum().item() < 1e-6)
                fill = 0.25 if is_zero else 4.0
                h = torch.full((1, 4, 512), fill)
                return SimpleNamespace(hidden_states=[h, h])
            return SimpleNamespace(image_embeds=torch.zeros(1, 1024))

    g = Hypergraph()
    g.add_node(
        "IP",
        IPAdapterNode(
            "IP",
            config={"ip_adapter_use_hidden_states": True},
            image_encoder=Enc(),
            feature_extractor=FE(),
        ),
    )
    from PIL import Image

    out = prepare_ip_adapter_image_embeds(g, Image.new("RGB", (8, 8), (255, 255, 255)))
    assert len(out) == 1
    assert tuple(out[0].shape) == (2, 1, 4, 512)
    assert abs(float(out[0][0, 0, 0, 0].item()) - 0.25) < 1e-5
    assert abs(float(out[0][1, 0, 0, 0].item()) - 4.0) < 1e-5

    wrong_pad = format_ip_adapter_image_embeds(
        out[0][1:2], device=torch.device("cpu"), dtype=None, do_classifier_free_guidance=True,
    )
    assert abs(float(wrong_pad[0][0, 0, 0, 0].item())) < 1e-5
    assert abs(float(wrong_pad[0][0, 0, 0, 0].item()) - 0.25) > 1e-3


def test_format_ip_adapter_plus_prepacked_skips_zero_pad() -> None:
    """4D [2, N, seq, dim] from encode must not get a second CFG zero block."""
    import torch

    dev = torch.device("cpu")
    packed = torch.randn(2, 2, 8, 512)
    out = format_ip_adapter_image_embeds(
        packed, device=dev, dtype=None, do_classifier_free_guidance=True,
    )
    assert len(out) == 1
    assert tuple(out[0].shape) == (2, 2, 8, 512)

    out_nc = format_ip_adapter_image_embeds(
        packed, device=dev, dtype=None, do_classifier_free_guidance=False,
    )
    assert tuple(out_nc[0].shape) == (1, 2, 8, 512)
