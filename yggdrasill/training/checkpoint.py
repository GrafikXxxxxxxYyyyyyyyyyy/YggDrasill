"""Training-graph sidecar metadata alongside diffusion ``trainer_state.pt`` checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def read_training_graph_sidecar(checkpoint_dir: str | Path) -> Optional[dict]:
    """Return parsed ``training_graph.json`` if present."""
    path = Path(checkpoint_dir) / "training_graph.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_training_graph_sidecar(
    checkpoint_dir: str | Path,
    *,
    global_step: int,
    training_plan_signature: str,
) -> Path:
    """Write ``training_graph.json`` next to ``trainer_state.pt``."""
    path = Path(checkpoint_dir) / "training_graph.json"
    path.write_text(
        json.dumps(
            {
                "kind": "yggdrasill_training_graph",
                "global_step": global_step,
                "training_plan_signature": training_plan_signature,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def assert_checkpoint_matches_plan(checkpoint_dir: str | Path, structure: Any) -> None:
    """Raise ``ValueError`` if sidecar signature disagrees with *structure* (optional guard)."""
    from yggdrasill.engine.planner import training_plan_signature

    side = read_training_graph_sidecar(checkpoint_dir)
    if side is None:
        return
    expected = training_plan_signature(structure)
    got = side.get("training_plan_signature")
    if got is not None and got != expected:
        raise ValueError(
            f"Checkpoint training_graph.json signature mismatch: {got!r} != {expected!r}"
        )
