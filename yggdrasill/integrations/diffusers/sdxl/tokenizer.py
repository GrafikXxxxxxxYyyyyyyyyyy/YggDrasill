"""SDXL tokenizer node: Converter — text → token_ids for Conjector (canon: converter/tokenizer)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConverter


class SDXLTokenizerNode(AbstractConverter):
    """Converts prompt text to input_ids for SDXL dual CLIP (Conjector).

    Canonical role: Converter. Outputs feed into prompt_encoder (Conjector).
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Any = None,
        tokenizer_2: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._tokenizer = tokenizer
        self._tokenizer_2 = tokenizer_2

    @property
    def block_type(self) -> str:
        return "sdxl/tokenizer"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_PROMPT_2, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_NEGATIVE_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_NEGATIVE_PROMPT_2, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_INPUT_IDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_INPUT_IDS_2, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_INPUT_IDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_INPUT_IDS_2, PortDirection.OUT, PortType.TENSOR),
        ]

    def _tokenize(self, text: str, tok: Any) -> Any:
        if not tok:
            raise RuntimeError("SDXLTokenizerNode requires tokenizer; none provided")
        out = tok(
            text or "",
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return out.input_ids

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get(C.PORT_PROMPT, "")
        prompt_2 = inputs.get(C.PORT_PROMPT_2)
        neg = inputs.get(C.PORT_NEGATIVE_PROMPT, "")
        neg_2 = inputs.get(C.PORT_NEGATIVE_PROMPT_2)

        p2 = prompt_2 if prompt_2 is not None else prompt
        n2 = neg_2 if neg_2 is not None else neg

        input_ids = self._tokenize(prompt, self._tokenizer)
        input_ids_2 = self._tokenize(p2, self._tokenizer_2)
        negative_input_ids = self._tokenize(neg or "", self._tokenizer)
        negative_input_ids_2 = self._tokenize(n2 or "", self._tokenizer_2)

        return {
            C.PORT_INPUT_IDS: input_ids,
            C.PORT_INPUT_IDS_2: input_ids_2,
            C.PORT_NEGATIVE_INPUT_IDS: negative_input_ids,
            C.PORT_NEGATIVE_INPUT_IDS_2: negative_input_ids_2,
        }
