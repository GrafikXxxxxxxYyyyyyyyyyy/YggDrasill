"""IP-Adapter UNet scales: inactive slots must get 0.0 each run (matches no-image runs)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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
