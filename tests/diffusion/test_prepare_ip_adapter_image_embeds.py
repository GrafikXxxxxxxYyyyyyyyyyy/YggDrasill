"""prepare_ip_adapter_image_embeds(graph) without Diffusers pipeline."""
from __future__ import annotations

import pytest

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
from yggdrasill.integrations.diffusers.common.ip_adapter_embeds import (
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
    from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode

    node = IPAdapterNode(
        "IP",
        image_encoder=FakeImageEncoder(),
        feature_extractor=FakeFeatureExtractor(),
    )
    img = FakeTensor((1, 3, 224, 224))
    a = node.encode_ip_adapter_image(img)
    b = node.forward({C.PORT_IP_ADAPTER_IMAGE: img})[C.PORT_IMAGE_EMBEDS]
    assert a.shape == b.shape
