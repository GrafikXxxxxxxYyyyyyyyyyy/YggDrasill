from __future__ import annotations

from enum import Enum
from typing import Optional, Set, Tuple


class Role(Enum):
    BACKBONE = "backbone"
    INJECTOR = "injector"
    CONJECTOR = "conjector"
    INNER_MODULE = "inner_module"
    OUTER_MODULE = "outer_module"
    HELPER = "helper"
    CONVERTER = "converter"


BACKBONE = Role.BACKBONE.value
INJECTOR = Role.INJECTOR.value
CONJECTOR = Role.CONJECTOR.value
INNER_MODULE = Role.INNER_MODULE.value
OUTER_MODULE = Role.OUTER_MODULE.value
HELPER = Role.HELPER.value
CONVERTER = Role.CONVERTER.value

KNOWN_ROLES: Set[str] = {
    BACKBONE, INJECTOR, CONJECTOR, INNER_MODULE,
    OUTER_MODULE, HELPER, CONVERTER,
}

ALL_ROLES: Tuple[Role, ...] = tuple(Role)


def role_from_block_type(block_type: str) -> Optional[str]:
    """Extract the role string from a *block_type*.

    Recognises exact match (``backbone``), slash-separated subtypes
    (``backbone/identity``), and underscore-separated subtypes
    (``backbone_unet2d``).  For multi-word roles like ``inner_module``,
    longer prefixes are tried first.

    ControlNet (adapter/controlnet, adapter/controlnet_flux) is always
    Inner Module per canon (02 §7, DIFFUSION_MODELS §3.3).

    IP-Adapter (adapter/ip_adapter) is a **Conjector**: an external conditioning encoder,
    not weight-level backbone adaptation (see :class:`~yggdrasill.task_nodes.abstract.AbstractInjector`).

    Returns the canonical role **string** (e.g. ``"backbone"``), not the
    ``Role`` enum, per spec PHASE_4 §6.2.
    """
    bt = block_type.strip().lower()
    if "controlnet" in bt and "adapter" in bt:
        return INNER_MODULE
    if "ip_adapter" in bt and "adapter" in bt:
        return CONJECTOR
    for role_str in sorted(KNOWN_ROLES, key=len, reverse=True):
        if bt == role_str:
            return role_str
        if bt.startswith(role_str + "/") or bt.startswith(role_str + "_"):
            return role_str
    return None
