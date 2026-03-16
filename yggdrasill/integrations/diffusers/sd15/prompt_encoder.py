"""SD1.5 text encoder node: AbstractConjector — token_ids → embeddings.

Canon: Conjector receives token_ids from Converter (tokenizer). No tokenization.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConjector


class SD15PromptEncoderNode(AbstractConjector):
    """CLIP text encoder for SD1.5. Input: token_ids from Converter. Output: embeddings."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        text_encoder: Any = None,
    ) -> None:
        cfg = dict(config or {})
        self._text_encoder = text_encoder or cfg.pop("text_encoder", None)
        super().__init__(node_id=node_id, block_id=block_id, config=cfg)

    @property
    def block_type(self) -> str:
        return "sd15/prompt_encoder"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_INPUT_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_INPUT_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def _encode_from_ids(self, input_ids: Any, clip_skip: Optional[int] = None) -> Any:
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._text_encoder = resolve_if_lazy(self._text_encoder)
        input_ids = input_ids.to(self._text_encoder.device)
        if clip_skip is not None and clip_skip > 0:
            output = self._text_encoder(input_ids, output_hidden_states=True)
            embeds = output.hidden_states[-(clip_skip + 1)]
            embeds = self._text_encoder.text_model.final_layer_norm(embeds)
        else:
            embeds = self._text_encoder(input_ids)[0]
        return embeds

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        clip_skip = self._config.get("clip_skip")
        input_ids = inputs[C.PORT_INPUT_IDS]
        negative_input_ids = inputs.get(C.PORT_NEGATIVE_INPUT_IDS)

        prompt_embeds = self._encode_from_ids(input_ids, clip_skip)
        neg_embeds = (
            self._encode_from_ids(negative_input_ids, clip_skip)
            if negative_input_ids is not None
            else torch.zeros_like(prompt_embeds)
        )

        return {
            C.PORT_PROMPT_EMBEDS: prompt_embeds,
            C.PORT_NEGATIVE_PROMPT_EMBEDS: neg_embeds,
        }

    def get_sub_blocks(self) -> Dict[str, Any]:
        return {}

    def to(self, device: Any) -> "SD15PromptEncoderNode":
        from yggdrasill.integrations.diffusers.lazy_component import resolve_if_lazy
        self._text_encoder = resolve_if_lazy(self._text_encoder)
        if self._text_encoder is not None and hasattr(self._text_encoder, "to"):
            self._text_encoder.to(device)
        return self
