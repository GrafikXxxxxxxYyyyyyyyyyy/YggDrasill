"""SD1.5 alias for :class:`InpaintLatentBlendNode` (backward-compatible block_type)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.integrations.diffusers.common.inpaint_latent_blend import InpaintLatentBlendNode


class SD15InpaintFourChannelBlendNode(InpaintLatentBlendNode):
    """Same behaviour as ``common/inpaint_latent_blend``; legacy graph block id."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)

    @property
    def block_type(self) -> str:
        return "sd15/inpaint_four_channel_blend"
