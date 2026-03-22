"""Multiple IP-Adapter checkpoints accumulate into one UNet._load_ip_adapter_weights call."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yggdrasill.integrations.diffusers.builder import (
    _META_IP_ADAPTER_ORDER,
    _META_IP_ADAPTER_SDS,
    _load_ip_adapter_weights_into_graph,
)


def test_second_ip_adapter_append_reloads_with_two_state_dicts() -> None:
    unet = MagicMock()
    unet._load_ip_adapter_weights = MagicMock()
    graph = MagicMock()
    graph.metadata = {}
    graph.node_ids = ["u"]
    u_node = MagicMock()
    u_node.block_type = "sdxl/unet"
    u_node._unet = unet

    def get_node(nid: str):
        if nid == "u":
            return u_node
        return None

    graph.get_node = get_node

    fake_sd = {"image_proj": {}, "ip_adapter": {}}
    with patch(
        "yggdrasill.integrations.diffusers.builder._load_ip_adapter_state_dict",
        return_value=fake_sd,
    ):
        _load_ip_adapter_weights_into_graph(
            graph,
            pretrained="h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="a.safetensors",
            adapter_node_id="ip_a",
        )
        _load_ip_adapter_weights_into_graph(
            graph,
            pretrained="h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="b.safetensors",
            adapter_node_id="ip_b",
        )

    assert len(graph.metadata[_META_IP_ADAPTER_SDS]) == 2
    assert graph.metadata[_META_IP_ADAPTER_ORDER] == ["ip_a", "ip_b"]
    assert unet._load_ip_adapter_weights.call_count == 2
    assert len(unet._load_ip_adapter_weights.call_args_list[-1][0][0]) == 2


def test_merge_ip_adapter_image_list_uses_weight_load_order_metadata() -> None:
    from yggdrasill.integrations.diffusers import contracts as C
    from yggdrasill.integrations.diffusers.builder import _META_IP_ADAPTER_ORDER
    from yggdrasill.integrations.diffusers.run import merge_ip_adapter_image_kwarg

    g = MagicMock()
    g.node_ids = ["style_ip", "face_ip"]
    g.metadata = {_META_IP_ADAPTER_ORDER: ["style_ip", "face_ip"]}
    s_node = MagicMock()
    s_node.block_type = "adapter/ip_adapter"
    f_node = MagicMock()
    f_node.block_type = "adapter/ip_adapter"

    def get_node(nid: str):
        return {"style_ip": s_node, "face_ip": f_node}[nid]

    g.get_node = get_node
    merged: dict = {}
    merge_ip_adapter_image_kwarg(merged, g, ["style_batch", "face_img"])
    assert merged[f"style_ip:{C.PORT_IP_ADAPTER_IMAGE}"] == "style_batch"
    assert merged[f"face_ip:{C.PORT_IP_ADAPTER_IMAGE}"] == "face_img"


def test_assign_to_single_exposed_routes_ip_list_when_multiple_exposed_match() -> None:
    """Regression: two IP nodes → same port_name in spec; list must not become bare ``ip_adapter_image``."""
    from yggdrasill.integrations.diffusers import contracts as C
    from yggdrasill.integrations.diffusers.builder import _META_IP_ADAPTER_ORDER
    from yggdrasill.integrations.diffusers.run import _assign_to_single_exposed

    g = MagicMock()
    g.node_ids = ["style_ip", "face_ip"]
    g.metadata = {_META_IP_ADAPTER_ORDER: ["style_ip", "face_ip"]}
    _ip = MagicMock()
    _ip.block_type = "adapter/ip_adapter"
    g.get_node = lambda nid: _ip if nid in ("style_ip", "face_ip") else None
    g.get_input_spec = lambda: [
        {"node_id": "style_ip", "port_name": C.PORT_IP_ADAPTER_IMAGE, "name": "x1"},
        {"node_id": "face_ip", "port_name": C.PORT_IP_ADAPTER_IMAGE, "name": "x2"},
    ]
    merged: dict = {}
    _assign_to_single_exposed(merged, g, C.PORT_IP_ADAPTER_IMAGE, [["s1", "s2"], "f1"])
    assert C.PORT_IP_ADAPTER_IMAGE not in merged
    assert merged[f"style_ip:{C.PORT_IP_ADAPTER_IMAGE}"] == ["s1", "s2"]
    assert merged[f"face_ip:{C.PORT_IP_ADAPTER_IMAGE}"] == "f1"


def test_merge_ip_adapter_image_dict_rejects_unknown_node_ids() -> None:
    from yggdrasill.integrations.diffusers.run import merge_ip_adapter_image_kwarg

    g = MagicMock()
    g.node_ids = ["style_ip", "face_ip"]
    g.metadata = {}
    n = MagicMock()
    n.block_type = "adapter/ip_adapter"

    def get_node(nid: str):
        if nid in ("style_ip", "face_ip"):
            return n
        return None

    g.get_node = get_node
    merged: dict = {}
    with pytest.raises(ValueError, match="not IP-Adapter node ids"):
        merge_ip_adapter_image_kwarg(
            merged, g, {"style_ip": object(), "typo_face": object()},
        )


def test_merge_ip_adapter_image_falls_back_to_sorted_node_ids() -> None:
    from yggdrasill.integrations.diffusers import contracts as C
    from yggdrasill.integrations.diffusers.run import merge_ip_adapter_image_kwarg

    g = MagicMock()
    g.node_ids = ["z_ip", "a_ip"]
    g.metadata = {}
    z = MagicMock()
    z.block_type = "adapter/ip_adapter"
    a = MagicMock()
    a.block_type = "adapter/ip_adapter"
    g.get_node = lambda nid: {"z_ip": z, "a_ip": a}[nid]
    merged: dict = {}
    merge_ip_adapter_image_kwarg(merged, g, ["first", "second"])
    assert merged[f"a_ip:{C.PORT_IP_ADAPTER_IMAGE}"] == "first"
    assert merged[f"z_ip:{C.PORT_IP_ADAPTER_IMAGE}"] == "second"
