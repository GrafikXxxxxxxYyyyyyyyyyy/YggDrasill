"""Config + Checkpoint serialization for blocks and hypergraphs.

Each level is saved as:
  <dir>/config.json    -- structure/metadata (JSON)
  <dir>/checkpoint.pkl -- dynamic state/weights (pickle)

block_id deduplication: when saving a hypergraph, blocks with the same
block_id share a single checkpoint entry.

.. warning::
   Checkpoint files use Python pickle. **Never load checkpoints from
   untrusted sources** — pickle can execute arbitrary code. For
   production use with untrusted data, consider safetensors or other
   safe formats.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.block import AbstractBaseBlock


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_config(path: Path) -> Dict[str, Any]:
    """Read a JSON or YAML config file."""
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(f"PyYAML required to load YAML config: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_checkpoint(path: Path) -> Dict[str, Any]:
    """Load checkpoint from pickle file.

    Warning: ``pickle.load`` can execute arbitrary code. Only load
    checkpoints from trusted sources.
    """
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


# ---------------------------------------------------------------------------
# Public config / checkpoint helpers (used by both block and hypergraph)
# ---------------------------------------------------------------------------

def save_config(config: Dict[str, Any], path: str | Path) -> None:
    _write_json(config, Path(path))


def load_config(path: str | Path) -> Dict[str, Any]:
    return _read_config(Path(path))


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    _write_checkpoint(state, Path(path))


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return _read_checkpoint(Path(path))


# ---------------------------------------------------------------------------
# Block-level save / load
# ---------------------------------------------------------------------------

def save_block(
    block: AbstractBaseBlock,
    directory: str | Path,
    *,
    config_filename: str = "config.json",
    checkpoint_filename: str = "checkpoint.pkl",
) -> Path:
    """Save a block to <directory>/config.json + checkpoint.pkl."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    config = block.get_config()
    config["schema_version"] = SCHEMA_VERSION
    _write_json(config, directory / config_filename)

    state = block.state_dict()
    if state:
        _write_checkpoint(state, directory / checkpoint_filename)

    return directory


def load_block(
    directory: str | Path,
    *,
    registry: Optional[Any] = None,
    config_filename: str = "config.json",
    checkpoint_filename: str = "checkpoint.pkl",
) -> AbstractBaseBlock:
    """Load a block from <directory>/config.json + checkpoint.pkl."""
    from yggdrasill.foundation.registry import BlockRegistry

    directory = Path(directory)
    config = _read_config(directory / config_filename)

    sv = config.get("schema_version")
    if sv is not None and sv != SCHEMA_VERSION:
        warnings.warn(
            f"Block config schema_version='{sv}' differs from expected '{SCHEMA_VERSION}'",
            stacklevel=2,
        )

    reg = registry or BlockRegistry.global_registry()
    block = reg.build(config)

    ckpt_path = directory / checkpoint_filename
    if ckpt_path.exists():
        state = _read_checkpoint(ckpt_path)
        block.load_state_dict(state, strict=False)

    return block


# ---------------------------------------------------------------------------
# Hypergraph-level save / load
# ---------------------------------------------------------------------------

def save_hypergraph(
    hypergraph: Hypergraph,
    directory: str | Path,
    *,
    config_filename: str = "config.json",
    checkpoint_filename: str = "checkpoint.pkl",
) -> Path:
    """Save <directory>/config.json + checkpoint.pkl (with block_id dedup)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    cfg = hypergraph.to_config()
    cfg["schema_version"] = SCHEMA_VERSION
    _write_json(cfg, directory / config_filename)

    full_state = hypergraph.state_dict()
    deduped = _deduplicate_state(hypergraph, full_state)
    _write_checkpoint(deduped, directory / checkpoint_filename)

    return directory


def load_hypergraph(
    directory: str | Path,
    *,
    registry: Optional[Any] = None,
    validate: bool = False,
    config_filename: str = "config.json",
    checkpoint_filename: str = "checkpoint.pkl",
    load_checkpoint_flag: bool = True,
) -> Hypergraph:
    """Load <directory>/config.json + checkpoint.pkl, rebuild via registry."""
    from yggdrasill.foundation.registry import BlockRegistry

    directory = Path(directory)
    config = _read_config(directory / config_filename)

    sv = config.get("schema_version")
    if sv is not None and sv != SCHEMA_VERSION:
        warnings.warn(
            f"Hypergraph config schema_version='{sv}' differs from expected '{SCHEMA_VERSION}'",
            stacklevel=2,
        )

    reg = registry or BlockRegistry.global_registry()
    h = Hypergraph.from_config(config, registry=reg, validate=validate)

    if load_checkpoint_flag:
        ckpt_path = directory / checkpoint_filename
        if ckpt_path.exists():
            state = _read_checkpoint(ckpt_path)
            expanded = _expand_deduped_state(h, state)
            h.load_state_dict(expanded, strict=False)

    return h


# ---------------------------------------------------------------------------
# block_id deduplication
# ---------------------------------------------------------------------------

def _deduplicate_state(
    hypergraph: Hypergraph, state: Dict[str, Any],
) -> Dict[str, Any]:
    """Collapse nodes with the same block_id into one checkpoint entry."""
    seen_block_ids: Dict[str, str] = {}
    aliases: Dict[str, str] = {}
    deduped: Dict[str, Any] = {}

    for nid in sorted(state.keys()):
        if nid == "_aliases":
            continue
        node = hypergraph.get_node(nid)
        block_id = getattr(node, "block_id", nid) if node is not None else nid
        if block_id in seen_block_ids:
            aliases[nid] = seen_block_ids[block_id]
        else:
            seen_block_ids[block_id] = nid
            deduped[nid] = state[nid]

    if aliases:
        deduped["_aliases"] = aliases

    return deduped


def _expand_deduped_state(
    hypergraph: Hypergraph, state: Dict[str, Any],
) -> Dict[str, Any]:
    """Expand aliases back to per-node state dicts.

    IMPORTANT: operates on a *copy* so the original checkpoint dict is not
    mutated (the ``_aliases`` key is not popped from ``state``).
    """
    aliases: Dict[str, str] = state.get("_aliases", {})
    expanded = {k: v for k, v in state.items() if k != "_aliases"}

    for nid, alias_target in aliases.items():
        if alias_target in expanded:
            expanded[nid] = expanded[alias_target]

    return expanded
