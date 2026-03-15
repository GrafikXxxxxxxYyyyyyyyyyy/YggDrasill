"""SDXL dual prompt encoding: tokenizer + tokenizer_2, text_encoder + text_encoder_2."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConjector


class SDXLPromptEncoderNode(AbstractConjector):
    """Encodes prompts through both CLIP encoders for SDXL.

    Produces concatenated hidden states and pooled embeddings.
    Supports prompt/prompt_2 pairs for dual text encoder path.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Any = None,
        tokenizer_2: Any = None,
        text_encoder: Any = None,
        text_encoder_2: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._tokenizer = tokenizer
        self._tokenizer_2 = tokenizer_2
        self._text_encoder = text_encoder
        self._text_encoder_2 = text_encoder_2

    @property
    def block_type(self) -> str:
        return "sdxl/prompt_encoder"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_PROMPT_2, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_NEGATIVE_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_NEGATIVE_PROMPT_2, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_INPUT_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_INPUT_IDS_2, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_NEGATIVE_INPUT_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_NEGATIVE_INPUT_IDS_2, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_POOLED_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_POOLED_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def _encode_single(
        self, text: str, tokenizer: Any, encoder: Any, clip_skip: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        """Encode with one tokenizer/encoder pair. Returns (hidden_states, pooled)."""
        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(encoder.device)
        return self._encode_from_ids_single(input_ids, encoder, clip_skip)

    def _encode_from_ids_single(
        self, input_ids: Any, encoder: Any, clip_skip: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        """Encode from token_ids (from Converter). Returns (hidden_states, pooled)."""
        input_ids = input_ids.to(encoder.device)
        output = encoder(input_ids, output_hidden_states=True)
        if clip_skip is not None and clip_skip > 0:
            hidden = output.hidden_states[-(clip_skip + 1)]
        else:
            hidden = output.hidden_states[-2]
        pooled = output[0]
        return hidden, pooled

    def _encode_prompt_pair(self, prompt: str, prompt_2: Optional[str]) -> Tuple[Any, Any]:
        """Encode through both text encoders and concatenate."""
        import torch

        p2 = prompt_2 if prompt_2 is not None else prompt
        clip_skip = self._config.get("clip_skip")

        hidden_1, _ = self._encode_single(prompt, self._tokenizer, self._text_encoder, clip_skip)
        hidden_2, pooled_2 = self._encode_single(p2, self._tokenizer_2, self._text_encoder_2, clip_skip)

        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)
        return prompt_embeds, pooled_2

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        clip_skip = self._config.get("clip_skip")
        input_ids = inputs.get(C.PORT_INPUT_IDS)
        input_ids_2 = inputs.get(C.PORT_INPUT_IDS_2)
        neg_ids = inputs.get(C.PORT_NEGATIVE_INPUT_IDS)
        neg_ids_2 = inputs.get(C.PORT_NEGATIVE_INPUT_IDS_2)

        if input_ids is not None and input_ids_2 is not None:
            # Canonical path: token_ids from Converter node
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
        else:
            # Backward compat: tokenize from prompt
            prompt = inputs.get(C.PORT_PROMPT, "")
            prompt_2 = inputs.get(C.PORT_PROMPT_2)
            neg_prompt = inputs.get(C.PORT_NEGATIVE_PROMPT, "")
            neg_prompt_2 = inputs.get(C.PORT_NEGATIVE_PROMPT_2)
            prompt_embeds, pooled = self._encode_prompt_pair(prompt, prompt_2)
            if neg_prompt or neg_prompt_2:
                neg_embeds, neg_pooled = self._encode_prompt_pair(
                    neg_prompt or "", neg_prompt_2
                )
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
