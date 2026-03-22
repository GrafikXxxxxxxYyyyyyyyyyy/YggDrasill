"""Tests for skipping ControlNet / IP-Adapter when no conditioning image is provided for a run."""
from __future__ import annotations

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.run import (
    _diffusion_skip_inactive_adapters,
    _merged_provides_input_for_node_port,
)


class _Node:
    def __init__(self, block_type: str) -> None:
        self.block_type = block_type


class _MockGraph:
    def __init__(
        self,
        nodes: dict[str, _Node],
        *,
        input_spec: list[dict] | None = None,
    ) -> None:
        self.node_ids = tuple(nodes.keys())
        self._nodes_map = nodes
        self._input_spec = input_spec or []

    def get_node(self, nid: str) -> _Node | None:
        return self._nodes_map.get(nid)

    def get_input_spec(self) -> list[dict]:
        return list(self._input_spec)


def test_merged_provides_port_respects_exposed_name() -> None:
    spec = [
        {
            "node_id": "CN",
            "port_name": C.PORT_CONTROL_IMAGE,
            "name": "control_image",
        },
    ]
    assert _merged_provides_input_for_node_port(
        {"control_image": object()}, "CN", C.PORT_CONTROL_IMAGE, spec,
    )
    assert not _merged_provides_input_for_node_port(
        {}, "CN", C.PORT_CONTROL_IMAGE, spec,
    )


def test_skip_inactive_adapters_empty_merged() -> None:
    g = _MockGraph(
        {
            "a": _Node("adapter/controlnet"),
            "b": _Node("adapter/ip_adapter"),
            "c": _Node("sdxl/unet"),
        },
        input_spec=[
            {"node_id": "a", "port_name": C.PORT_CONTROL_IMAGE, "name": "cn_img"},
            {"node_id": "b", "port_name": C.PORT_IP_ADAPTER_IMAGE, "name": "ip_img"},
        ],
    )
    # IP-Adapter always runs (inactive zeros when no image) so multi-IP CONCAT stays aligned.
    assert _diffusion_skip_inactive_adapters(g, {}) == {"a"}


def test_skip_inactive_adapters_ip_not_skipped_when_embeds_only() -> None:
    g = _MockGraph(
        {"ip": _Node("adapter/ip_adapter")},
        input_spec=[
            {"node_id": "ip", "port_name": C.PORT_IP_ADAPTER_IMAGE, "name": "ip_img"},
        ],
    )
    merged = {f"ip:{C.PORT_IP_ADAPTER_IMAGE_EMBEDS}": object()}
    assert _diffusion_skip_inactive_adapters(g, merged) == set()


def test_skip_inactive_adapters_partial_images() -> None:
    g = _MockGraph(
        {
            "cn1": _Node("adapter/controlnet"),
            "cn2": _Node("adapter/controlnet"),
            "ip": _Node("adapter/ip_adapter"),
        },
        input_spec=[
            {"node_id": "cn1", "port_name": C.PORT_CONTROL_IMAGE, "name": "c1"},
            {"node_id": "cn2", "port_name": C.PORT_CONTROL_IMAGE, "name": "c2"},
            {"node_id": "ip", "port_name": C.PORT_IP_ADAPTER_IMAGE, "name": "ip_img"},
        ],
    )
    merged = {f"cn1:{C.PORT_CONTROL_IMAGE}": object()}
    assert _diffusion_skip_inactive_adapters(g, merged) == {"cn2"}


def test_skip_inactive_adapters_never_skips_ip_even_with_two_ip_nodes() -> None:
    g = _MockGraph(
        {
            "style_ip": _Node("adapter/ip_adapter"),
            "face_ip": _Node("adapter/ip_adapter"),
        },
        input_spec=[
            {"node_id": "style_ip", "port_name": C.PORT_IP_ADAPTER_IMAGE, "name": "s"},
            {"node_id": "face_ip", "port_name": C.PORT_IP_ADAPTER_IMAGE, "name": "f"},
        ],
    )
    merged = {f"style_ip:{C.PORT_IP_ADAPTER_IMAGE}": object()}
    assert _diffusion_skip_inactive_adapters(g, merged) == set()


def test_prepare_merges_skip_into_run_kw() -> None:
    from yggdrasill.integrations.diffusers.run import _prepare_diffusion_run

    g = _MockGraph(
        {"cn": _Node("adapter/controlnet")},
        input_spec=[
            {"node_id": "cn", "port_name": C.PORT_CONTROL_IMAGE, "name": "x"},
        ],
    )
    run_kw: dict = {"skip_node_ids": {"other"}}
    _prepare_diffusion_run(g, run_kw, merged_inputs={})
    assert run_kw["skip_node_ids"] == {"other", "cn"}
