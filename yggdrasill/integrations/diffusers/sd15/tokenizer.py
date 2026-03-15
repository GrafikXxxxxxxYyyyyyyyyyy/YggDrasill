"""SD1.5 tokenizer node: Converter — text → token_ids for Conjector (canon: converter/tokenizer)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from yggdrasill.diffusion import contracts as C
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.task_nodes.abstract import AbstractConverter


class SD15TokenizerNode(AbstractConverter):
    """Converts prompt text to input_ids for SD1.5 CLIP (Conjector).

    Canonical role: Converter. Output feeds into prompt_encoder (Conjector).
    """

    def __init__(
        self,
        node_id: str,
        block_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._tokenizer = tokenizer

    @property
    def block_type(self) -> str:
        return "sd15/tokenizer"

    def declare_ports(self) -> List[Port]:
        return [
            Port(C.PORT_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_NEGATIVE_PROMPT, PortDirection.IN, PortType.TEXT, optional=True),
            Port(C.PORT_INPUT_IDS, PortDirection.OUT, PortType.TENSOR),
            Port(C.PORT_NEGATIVE_INPUT_IDS, PortDirection.OUT, PortType.TENSOR),
        ]

    def _tokenize(self, text: str) -> Any:
        if not self._tokenizer:
            raise RuntimeError("SD15TokenizerNode requires tokenizer; none provided")
        out = self._tokenizer(
            text or "",
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return out.input_ids

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get(C.PORT_PROMPT, "")
        negative_prompt = inputs.get(C.PORT_NEGATIVE_PROMPT, "")

        if isinstance(prompt, list):
            import torch
            input_ids = torch.cat([self._tokenize(p or "") for p in prompt], dim=0)
        else:
            input_ids = self._tokenize(prompt)

        if isinstance(negative_prompt, list):
            import torch
            neg_ids = torch.cat([self._tokenize(p or "") for p in negative_prompt], dim=0)
        else:
            neg_ids = self._tokenize(negative_prompt or "")

        return {
            C.PORT_INPUT_IDS: input_ids,
            C.PORT_NEGATIVE_INPUT_IDS: neg_ids,
        }
