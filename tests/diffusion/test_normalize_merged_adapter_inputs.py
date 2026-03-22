"""Adapter payloads on the positional ``inputs`` dict must expand like kwargs."""
from __future__ import annotations

from typing import Any, List

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.run import normalize_merged_adapter_inputs


class _Graph:
    """Minimal graph with input_spec only (matches how routing uses get_input_spec)."""

    def get_input_spec(self) -> List[dict[str, Any]]:
        return [
            {
                "node_id": "IPAdapter",
                "port_name": C.PORT_IP_ADAPTER_IMAGE,
                "name": f"IPAdapter:{C.PORT_IP_ADAPTER_IMAGE}",
            },
        ]


def test_normalize_expands_ip_adapter_image_dict_from_inputs() -> None:
    merged = {C.PORT_IP_ADAPTER_IMAGE: {"IPAdapter": "ref.png"}}
    normalize_merged_adapter_inputs(merged, _Graph())
    assert merged == {f"IPAdapter:{C.PORT_IP_ADAPTER_IMAGE}": "ref.png"}
    assert C.PORT_IP_ADAPTER_IMAGE not in merged


def test_normalize_scalar_ip_image_assigns_single_exposed() -> None:
    merged = {C.PORT_IP_ADAPTER_IMAGE: "only.png"}
    normalize_merged_adapter_inputs(merged, _Graph())
    assert merged == {f"IPAdapter:{C.PORT_IP_ADAPTER_IMAGE}": "only.png"}
