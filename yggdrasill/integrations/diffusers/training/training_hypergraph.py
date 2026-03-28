"""Assemble a minimal hypergraph for diffusion LoRA training (loss + optim helpers)."""
from __future__ import annotations

from typing import Any, List, Tuple

from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.hypergraph.structure import Hypergraph
from yggdrasill.integrations.diffusers.training.diffusion_loss_node import DiffusionLoRALoss
from yggdrasill.integrations.diffusers.training.types import TrainingObjective

# Ensure training helpers are registered
import yggdrasill.training.blocks  # noqa: F401


def build_diffusion_lora_training_hypergraph(
    *,
    objective: TrainingObjective,
    post_backward_node_ids: Tuple[str, ...] = ("optim", "sched"),
    include_checkpoint_hook: bool = False,
) -> Hypergraph:
    """Build a DAG: single loss converter + registered helper nodes (no inter-node edges)."""
    g = Hypergraph("diffusion_lora_training")
    g.graph_kind = "training"
    g.metadata["graph_kind"] = "training"
    g.metadata["num_loop_steps"] = 1

    loss_node = DiffusionLoRALoss(node_id="loss", objective=objective)
    g.add_node("loss", loss_node)
    g.expose_input("loss", "batch", name="batch")

    reg = BlockRegistry.global_registry()
    post_ids: List[str] = list(post_backward_node_ids)
    if include_checkpoint_hook and "ckpt" not in post_ids:
        post_ids.append("ckpt")

    for nid in post_ids:
        if nid == "optim":
            g.add_node(
                "optim",
                reg.build({"block_type": "training/optim_step", "node_id": "optim"}),
            )
        elif nid == "sched":
            g.add_node(
                "sched",
                reg.build({"block_type": "training/lr_scheduler_step", "node_id": "sched"}),
            )
        elif nid == "ckpt":
            g.add_node(
                "ckpt",
                reg.build({"block_type": "training/checkpoint_hook", "node_id": "ckpt"}),
            )
        else:
            raise ValueError(f"Unknown post_backward token: {nid!r}")

    g.metadata["training"] = {
        "loss_node_id": "loss",
        "loss_port": "loss",
        "forward_node_ids": ["loss"],
        "post_backward_node_ids": post_ids,
    }
    return g
