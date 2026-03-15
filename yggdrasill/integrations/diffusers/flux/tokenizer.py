"""FLUX tokenizer node: Converter — text → token_ids for Conjector (canon: converter/tokenizer)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConverter


class FluxTokenizerNode(AbstractConverter):
    """Converts prompt text to input_ids for FLUX CLIP + T5 (Conjector).

    Canonical role: Converter. Outputs feed into prompt_encoder (Conjector).
    FLUX uses no negative prompt (no CFG).
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
        return "flux/tokenizer"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_PROMPT_2, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_INPUT_IDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_INPUT_IDS_2, PortDirection.OUT, PortType.TENSOR),
        ]

    def _tokenize(self, text: str, tok: Any, max_length: Optional[int] = None) -> Any:
        if not tok:
            raise RuntimeError("FluxTokenizerNode requires tokenizer; none provided")
        max_len = max_length or getattr(tok, "model_max_length", 77)
        out = tok(
            text or "",
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        return out.input_ids

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get(C.PORT_PROMPT, "")
        prompt_2 = inputs.get(C.PORT_PROMPT_2) or prompt
        max_seq = self._config.get("max_sequence_length", 512)

        input_ids = self._tokenize(prompt, self._tokenizer)
        input_ids_2 = self._tokenize(prompt_2, self._tokenizer_2, max_length=max_seq)

        return {
            C.PORT_INPUT_IDS: input_ids,
            C.PORT_INPUT_IDS_2: input_ids_2,
        }
