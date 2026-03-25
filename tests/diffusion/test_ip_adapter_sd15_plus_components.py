from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder


def _fake_sd15_graph_with_unet(unet: object) -> Hypergraph:
    class BackboneNode:
        block_type = "sd15/unet"

        def __init__(self, unet: object) -> None:
            self._unet = unet

    g = Hypergraph(graph_id="t")
    g.add_node("unet", BackboneNode(unet))
    return g


def test_builder_sd15_ipadapter_plus_defaults_weight_name_and_sets_hidden_states_flag() -> None:
    unet = MagicMock()
    unet.encoder_hid_proj = None
    unet.parameters = lambda: iter([])  # type: ignore[assignment]
    unet._load_ip_adapter_weights = MagicMock()

    g = _fake_sd15_graph_with_unet(unet)
    b = DiffusionGraphBuilder(g)

    with patch(
        "yggdrasill.integrations.diffusers.builder._load_ip_adapter_state_dict",
        return_value={"image_proj": {}, "ip_adapter": {}},
    ) as load_sd:
        b.add_component(
            "IP",
            "sd15.ipadapter_plus",
            pretrained="h94/IP-Adapter",
        )

    assert load_sd.called
    _args, kwargs = load_sd.call_args
    assert kwargs.get("subfolder") == "models"
    assert kwargs.get("weight_name") == "ip-adapter-plus_sd15.safetensors"

    node = g.get_node("IP")
    assert node is not None
    assert getattr(node, "_config", {}).get("ip_adapter_use_hidden_states") is True


def test_builder_sd15_ipadapter_plus_face_defaults_weight_name_and_sets_hidden_states_flag() -> None:
    unet = MagicMock()
    unet.encoder_hid_proj = None
    unet.parameters = lambda: iter([])  # type: ignore[assignment]
    unet._load_ip_adapter_weights = MagicMock()

    g = _fake_sd15_graph_with_unet(unet)
    b = DiffusionGraphBuilder(g)

    with patch(
        "yggdrasill.integrations.diffusers.builder._load_ip_adapter_state_dict",
        return_value={"image_proj": {}, "ip_adapter": {}},
    ) as load_sd:
        b.add_component(
            "IP",
            "sd15.ipadapter_plus_face",
            pretrained="h94/IP-Adapter",
        )

    assert load_sd.called
    _args, kwargs = load_sd.call_args
    assert kwargs.get("subfolder") == "models"
    assert kwargs.get("weight_name") == "ip-adapter-plus-face_sd15.safetensors"

    node = g.get_node("IP")
    assert node is not None
    assert getattr(node, "_config", {}).get("ip_adapter_use_hidden_states") is True

