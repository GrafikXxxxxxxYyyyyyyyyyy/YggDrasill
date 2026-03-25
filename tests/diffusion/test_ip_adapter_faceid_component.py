from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder


def test_builder_sdxl_ipadapter_faceid_defaults_weight_name_and_loads_into_unet() -> None:
    class BackboneNode:
        block_type = "sdxl/unet"

        def __init__(self, unet: object) -> None:
            self._unet = unet

    unet = MagicMock()
    unet.encoder_hid_proj = None
    unet.parameters = lambda: iter([])  # type: ignore[assignment]
    unet._load_ip_adapter_weights = MagicMock()

    g = Hypergraph(graph_id="t")
    g.add_node("unet", BackboneNode(unet))

    b = DiffusionGraphBuilder(g)

    with patch(
        "yggdrasill.integrations.diffusers.builder._load_ip_adapter_state_dict",
        return_value={"image_proj": {}, "ip_adapter": {}},
    ) as load_sd:
        b.add_component(
            "IPAdapter",
            "sdxl.ipadapter_faceid",
            pretrained="h94/IP-Adapter-FaceID",
        )

    assert load_sd.called
    args, kwargs = load_sd.call_args
    assert args[0] == "h94/IP-Adapter-FaceID"
    assert kwargs.get("subfolder") is None
    assert kwargs.get("weight_name") == "ip-adapter-faceid_sdxl.bin"

    node = g.get_node("IPAdapter")
    assert getattr(node, "_image_encoder", None) is None
    assert getattr(node, "_feature_extractor", None) is None

    assert unet._load_ip_adapter_weights.call_count == 1


def test_builder_sd15_ipadapter_faceid_defaults_weight_name() -> None:
    class BackboneNode:
        block_type = "sd15/unet"

        def __init__(self, unet: object) -> None:
            self._unet = unet

    unet = MagicMock()
    unet.encoder_hid_proj = None
    unet.parameters = lambda: iter([])  # type: ignore[assignment]
    unet._load_ip_adapter_weights = MagicMock()

    g = Hypergraph(graph_id="t")
    g.add_node("unet", BackboneNode(unet))

    b = DiffusionGraphBuilder(g)

    with patch(
        "yggdrasill.integrations.diffusers.builder._load_ip_adapter_state_dict",
        return_value={"image_proj": {}, "ip_adapter": {}},
    ) as load_sd:
        b.add_component(
            "IPAdapter",
            "sd15.ipadapter_faceid",
            pretrained="h94/IP-Adapter-FaceID",
        )

    assert load_sd.called
    args, kwargs = load_sd.call_args
    assert args[0] == "h94/IP-Adapter-FaceID"
    assert kwargs.get("subfolder") is None
    assert kwargs.get("weight_name") == "ip-adapter-faceid_sd15.bin"

