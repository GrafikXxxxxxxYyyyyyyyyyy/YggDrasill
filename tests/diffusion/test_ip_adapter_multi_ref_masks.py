"""Multi-reference IP-Adapter requires spatial masks (enforced in run._prepare_diffusion_run)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.run import _enforce_ip_adapter_multi_ref_with_masks


def _fake_graph_with_ip() -> MagicMock:
    g = MagicMock()
    g.node_ids = ["ip"]
    node = MagicMock()
    node.block_type = "adapter/ip_adapter"
    g.get_node = lambda nid: node if nid == "ip" else None
    return g


def test_two_refs_without_masks_raises() -> None:
    g = _fake_graph_with_ip()
    merged = {f"ip:{C.PORT_IP_ADAPTER_IMAGE}": ["a.png", "b.png"]}
    with pytest.warns(UserWarning, match="unstable without spatial masks"):
        _enforce_ip_adapter_multi_ref_with_masks(g, merged)


def test_three_refs_without_masks_ok() -> None:
    """Batched style folders (3+ images) should not require spatial masks."""
    g = _fake_graph_with_ip()
    merged = {f"ip:{C.PORT_IP_ADAPTER_IMAGE}": ["a.png", "b.png", "c.png"]}
    _enforce_ip_adapter_multi_ref_with_masks(g, merged)


def test_style_batch_plus_face_two_tuple_ok() -> None:
    """``[many_style_paths, face_path]`` has len 2 but is not two same-type references."""
    g = _fake_graph_with_ip()
    merged = {f"ip:{C.PORT_IP_ADAPTER_IMAGE}": [["a.png", "b.png", "c.png"], "face.png"]}
    _enforce_ip_adapter_multi_ref_with_masks(g, merged)


def test_multi_ref_with_mask_images_ok() -> None:
    g = _fake_graph_with_ip()
    merged = {
        f"ip:{C.PORT_IP_ADAPTER_IMAGE}": ["a.png", "b.png"],
        C.PORT_IP_ADAPTER_MASK_IMAGES: ["m1.png", "m2.png"],
    }
    _enforce_ip_adapter_multi_ref_with_masks(g, merged)


def test_single_ref_without_masks_ok() -> None:
    g = _fake_graph_with_ip()
    merged = {f"ip:{C.PORT_IP_ADAPTER_IMAGE}": "one.png"}
    _enforce_ip_adapter_multi_ref_with_masks(g, merged)


def test_no_ip_node_skips() -> None:
    g = MagicMock()
    g.node_ids = ["unet"]
    n = MagicMock()
    n.block_type = "sdxl/unet"
    g.get_node = lambda nid: n if nid == "unet" else None
    merged = {f"ip:{C.PORT_IP_ADAPTER_IMAGE}": ["a.png", "b.png"]}
    _enforce_ip_adapter_multi_ref_with_masks(g, merged)
