"""IP-Adapter UNet scales: inactive slots must get 0.0 each run (matches no-image runs)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.run import _inject_ip_adapter_scale


class _Graph:
    def __init__(self) -> None:
        self.node_ids = ("IP", "U")

    def get_node(self, nid: str):
        if nid == "IP":
            return SimpleNamespace(block_type="adapter/ip_adapter")
        if nid == "U":
            return SimpleNamespace(block_type="sdxl/unet", _unet=object())
        return None

    def get_input_spec(self):
        return []


def test_inject_ip_adapter_scale_zeros_inactive_without_explicit_kwarg_path() -> None:
    g = _Graph()
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, {"default": 1.0}, {})
    set_scale.assert_called_once()
    args, _kw = set_scale.call_args
    assert args[1] == 0.0


def test_inject_ip_adapter_scale_single_slot_active_uses_default() -> None:
    g = _Graph()
    merged = {"IP:ip_adapter_image": object()}
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, {"default": 1.0}, merged)
    set_scale.assert_called_once()
    args, _kw = set_scale.call_args
    assert args[1] == 1.0


def test_inject_ip_adapter_scale_single_element_list_normalized_to_float() -> None:
    g = _Graph()
    merged = {"IP:ip_adapter_image": object()}
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, {"IP": [0.55]}, merged)
    assert set_scale.call_args.args[1] == 0.55


def test_inject_ip_adapter_scale_list_collapses_to_mean_without_masks() -> None:
    """No spatial masks → diffusers needs a scalar scale per adapter (not ``[[a,b]]``)."""
    g = _Graph()
    merged = {"IP:ip_adapter_image": object()}
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, {"IP": [0.5, 0.9]}, merged)
    assert set_scale.call_args.args[1] == 0.7


def test_inject_ip_adapter_scale_nested_list_for_multi_reference_masks() -> None:
    """Match diffusers ``set_ip_adapter_scale([[0.7, 0.7]])`` — one adapter slot, two reference strengths."""
    g = _Graph()
    merged = {
        "IP:ip_adapter_image": object(),
        C.PORT_IP_ADAPTER_MASK_IMAGES: [object(), object()],
    }
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, {"IP": [0.7, 0.7]}, merged)
    set_scale.assert_called_once()
    args, _kw = set_scale.call_args
    assert args[1] == [[0.7, 0.7]]


def test_inject_ip_adapter_scale_active_when_only_precomputed_embeds() -> None:
    g = _Graph()
    merged = {"IP:ip_adapter_image_embeds": object()}
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, {"default": 0.7}, merged)
    set_scale.assert_called_once()
    args, _kw = set_scale.call_args
    assert args[1] == 0.7


def test_inject_ip_adapter_scale_instant_style_global_dict() -> None:
    """InstantStyle map (only down/up/mid keys) must not be mistaken for {node_id: scale}."""
    g = _Graph()
    merged = {"IP:ip_adapter_image": object()}
    layout = {
        "down": {"block_2": [0.0, 1.0]},
        "up": {"block_0": [0.0, 1.0, 0.0]},
    }
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, layout, merged)
    set_scale.assert_called_once()
    args, _kw = set_scale.call_args
    assert args[1] is layout


def test_inject_ip_adapter_scale_style_only_up_block_dict() -> None:
    g = _Graph()
    merged = {"IP:ip_adapter_image": object()}
    layout = {"up": {"block_0": [0.0, 1.0, 0.0]}}
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, layout, merged)
    assert set_scale.call_args.args[1] is layout


class _GraphTwoIP:
    node_ids = ("A", "B", "U")

    def get_node(self, nid: str):
        if nid in ("A", "B"):
            return SimpleNamespace(block_type="adapter/ip_adapter")
        if nid == "U":
            return SimpleNamespace(block_type="sdxl/unet", _unet=object())
        return None

    def get_input_spec(self):
        return []


def test_inject_ip_adapter_scale_list_per_adapter_in_order() -> None:
    g = _GraphTwoIP()
    merged = {
        "A:ip_adapter_image": object(),
        "B:ip_adapter_image": object(),
    }
    d0 = {"down": {"block_2": [0.0, 1.0]}}
    d1 = {"up": {"block_0": [0.0, 1.0, 0.0]}}
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, [d0, d1], merged)
    set_scale.assert_called_once()
    assert set_scale.call_args.args[1] == [d0, d1]


def test_inject_ip_adapter_scale_per_node_layout_under_node_id() -> None:
    """``{node_id: {\"down\": ...}}`` uses per-node branch, not global layout."""
    g = _Graph()
    merged = {"IP:ip_adapter_image": object()}
    inner = {"down": {"block_2": [0.0, 1.0]}, "up": {"block_0": [0.0, 1.0, 0.0]}}
    with patch(
        "yggdrasill.integrations.diffusers.adapters.ip_adapter_loader._set_ip_adapter_scale_on_unet",
    ) as set_scale:
        _inject_ip_adapter_scale(g, {"IP": inner, "default": 0.5}, merged)
    assert set_scale.call_args.args[1] is inner
