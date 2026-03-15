"""SD1.5 prompt encoding node: CLIPTokenizer + CLIPTextModel."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConjector


class SD15PromptEncoderNode(AbstractConjector):
    """Encodes text prompts into CLIP embeddings for SD1.5.

    Accepts positive and negative text prompts, produces prompt_embeds tensors.
    Supports clip_skip via config.
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Any = None,
        text_encoder: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder

    @property
    def block_type(self) -> str:
        return "sd15/prompt_encoder"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_NEGATIVE_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_INPUT_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_NEGATIVE_INPUT_IDS, PortDirection.IN, PortType.TENSOR, optional=True),
            Port(C.PORT_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_PROMPT_EMBEDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def _encode_prompt(self, prompt: str, clip_skip: Optional[int] = None) -> Any:
        text_inputs = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self._text_encoder.device)

        if clip_skip is not None and clip_skip > 0:
            output = self._text_encoder(input_ids, output_hidden_states=True)
            embeds = output.hidden_states[-(clip_skip + 1)]
            embeds = self._text_encoder.text_model.final_layer_norm(embeds)
        else:
            embeds = self._text_encoder(input_ids)[0]

        return embeds

    def _encode_from_ids(self, input_ids: Any, clip_skip: Optional[int] = None) -> Any:
        """Encode from token_ids (from Converter); no tokenization."""
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
        input_ids = inputs.get(C.PORT_INPUT_IDS)
        negative_input_ids = inputs.get(C.PORT_NEGATIVE_INPUT_IDS)

        if input_ids is not None:
            # Canonical path: token_ids from Converter node
            prompt_embeds = self._encode_from_ids(input_ids, clip_skip)
            neg_embeds = (
                self._encode_from_ids(negative_input_ids, clip_skip)
                if negative_input_ids is not None
                else torch.zeros_like(prompt_embeds)
            )
        else:
            # Backward compat: tokenize from prompt (or from upstream tokenizer not wired)
            prompt = inputs.get(C.PORT_PROMPT, "")
            negative_prompt = inputs.get(C.PORT_NEGATIVE_PROMPT, "")
            if isinstance(prompt, list):
                prompt_embeds = torch.cat(
                    [self._encode_prompt(p, clip_skip) for p in prompt], dim=0
                )
            else:
                prompt_embeds = self._encode_prompt(prompt or "", clip_skip)
            if not negative_prompt:
                negative_prompt = ""
            if isinstance(negative_prompt, list):
                neg_embeds = torch.cat(
                    [self._encode_prompt(p, clip_skip) for p in negative_prompt], dim=0
                )
            else:
                neg_embeds = self._encode_prompt(negative_prompt, clip_skip)

        return {
            C.PORT_PROMPT_EMBEDS: prompt_embeds,
            C.PORT_NEGATIVE_PROMPT_EMBEDS: neg_embeds,
        }

    def get_sub_blocks(self) -> Dict[str, Any]:
        return {}

    def to(self, device: Any) -> "SD15PromptEncoderNode":
        if self._text_encoder is not None and hasattr(self._text_encoder, "to"):
            self._text_encoder.to(device)
        return self
