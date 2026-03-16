"""SDXL dual text encoder node: AbstractConjector — token_ids → embeddings.

Canon: Conjector receives token_ids from Converter (tokenizer). No tokenization.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConjector


class SDXLPromptEncoderNode(AbstractConjector):
    """Dual CLIP text encoder for SDXL. Input: token_ids from Converter. Output: embeddings."""

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        text_encoder: Any = None,
        text_encoder_2: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._text_encoder = text_encoder
        self._text_encoder_2 = text_encoder_2

    @property
    def block_type(self) -> str:
        return "sdxl/prompt_encoder"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_INPUT_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_INPUT_IDS_2, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_INPUT_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_NEGATIVE_INPUT_IDS_2, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_POOLED_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def _encode_from_ids_single(
        self, input_ids: Any, encoder: Any, clip_skip: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        input_ids = input_ids.to(encoder.device)
        output = encoder(input_ids, output_hidden_states=True)
        if clip_skip is not None and clip_skip > 0:
            hidden = output.hidden_states[-(clip_skip + 1)]
        else:
            hidden = output.hidden_states[-2]
        pooled = output[0]
        return hidden, pooled

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        clip_skip = self._config.get("clip_skip")
        input_ids = inputs[C.PORT_INPUT_IDS]
        input_ids_2 = inputs[C.PORT_INPUT_IDS_2]
        neg_ids = inputs.get(C.PORT_NEGATIVE_INPUT_IDS)
        neg_ids_2 = inputs.get(C.PORT_NEGATIVE_INPUT_IDS_2)

        hidden_1, _ = self._encode_from_ids_single(
            input_ids, self._text_encoder, clip_skip
        )
        hidden_2, pooled = self._encode_from_ids_single(
            input_ids_2, self._text_encoder_2, clip_skip
        )
        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)

        if neg_ids is not None and neg_ids_2 is not None:
            n1, _ = self._encode_from_ids_single(
                neg_ids, self._text_encoder, clip_skip
            )
            n2, neg_pooled = self._encode_from_ids_single(
                neg_ids_2, self._text_encoder_2, clip_skip
            )
            neg_embeds = torch.cat([n1, n2], dim=-1)
        else:
            neg_embeds = torch.zeros_like(prompt_embeds)
            neg_pooled = torch.zeros_like(pooled)

        return {
            C.PORT_PROMPT_EMBEDS: prompt_embeds,
            C.PORT_NEGATIVE_PROMPT_EMBEDS: neg_embeds,
            C.PORT_POOLED_PROMPT_EMBEDS: pooled,
            C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS: neg_pooled,
        }

    def to(self, device: Any) -> "SDXLPromptEncoderNode":
        for enc in (self._text_encoder, self._text_encoder_2):
            if enc is not None and hasattr(enc, "to"):
                enc.to(device)
        return self
