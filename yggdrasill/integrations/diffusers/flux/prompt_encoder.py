"""FLUX text encoder node: AbstractConjector — token_ids → embeddings.

Canon: Conjector receives token_ids from Converter (tokenizer). No tokenization.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConjector


class FluxPromptEncoderNode(AbstractConjector):
    """CLIP + T5 text encoder for FLUX. Input: token_ids from Converter. Output: embeddings."""

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
        return "flux/prompt_encoder"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_INPUT_IDS, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_INPUT_IDS_2, PortDirection.IN, PortType.TENSOR),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_POOLED_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_TXT_IDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def _encode_clip_from_ids(self, input_ids: Any) -> Any:
        input_ids = input_ids.to(self._text_encoder.device)
        output = self._text_encoder(input_ids, output_hidden_states=False)
        return output.pooler_output

    def _encode_t5_from_ids(self, input_ids: Any, max_sequence_length: int) -> Any:
        import torch

        input_ids = input_ids.to(self._text_encoder_2.device)
        output = self._text_encoder_2(input_ids)
        prompt_embeds = output[0]
        seq_len = prompt_embeds.shape[1]
        text_ids = torch.zeros(seq_len, 3, device=prompt_embeds.device)
        return prompt_embeds, text_ids

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        max_seq_len = self._config.get("max_sequence_length", 512)
        input_ids = inputs[C.PORT_INPUT_IDS]
        input_ids_2 = inputs[C.PORT_INPUT_IDS_2]

        pooled_prompt_embeds = self._encode_clip_from_ids(input_ids)
        prompt_embeds, text_ids = self._encode_t5_from_ids(input_ids_2, max_seq_len)

        return {
            C.PORT_PROMPT_EMBEDS: prompt_embeds,
            C.PORT_POOLED_PROMPT_EMBEDS: pooled_prompt_embeds,
            C.PORT_TXT_IDS: text_ids,
        }

    def to(self, device: Any) -> "FluxPromptEncoderNode":
        for enc in (self._text_encoder, self._text_encoder_2):
            if enc is not None and hasattr(enc, "to"):
                enc.to(device)
        return self
