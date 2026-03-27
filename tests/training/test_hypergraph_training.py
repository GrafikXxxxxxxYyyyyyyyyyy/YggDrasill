"""Tests for hypergraph-native training plan and executor."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yggdrasill.engine.edge import Edge
from yggdrasill.engine.validator import validate
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.hypergraph.structure import Hypergraph
from yggdrasill.integrations.diffusers.training.diffusion_loss_node import DiffusionLoRALossConverter
from yggdrasill.integrations.diffusers.training.training_hypergraph import (
    build_diffusion_lora_training_hypergraph,
)
from yggdrasill.training.context import TrainingStepContext
from yggdrasill.training.executor import run_training_step
from yggdrasill.training.plan import build_training_plan, training_plan_signature

import yggdrasill.training.blocks  # noqa: F401


def _toy_training_graph(dim: int = 3) -> Hypergraph:
    g = Hypergraph("toy_train")
    g.graph_kind = "training"
    g.metadata["graph_kind"] = "training"
    g.metadata["num_loop_steps"] = 1
    reg = BlockRegistry.global_registry()
    toy = reg.build({"block_type": "training/toy_linear", "node_id": "toy", "config": {"dim": dim}})
    mse = reg.build({"block_type": "training/mse_loss", "node_id": "mse"})
    g.add_node("toy", toy)
    g.add_node("mse", mse)
    g.add_node("optim", reg.build({"block_type": "training/optim_step", "node_id": "optim"}))
    g.add_node("sched", reg.build({"block_type": "training/lr_scheduler_step", "node_id": "sched"}))
    g.add_edge(Edge("toy", "pred", "mse", "pred"))
    g.add_edge(Edge("toy", "target", "mse", "target"))
    g.expose_input("toy", "x", name="x")
    g.metadata["training"] = {
        "loss_node_id": "mse",
        "loss_port": "loss",
        "post_backward_node_ids": ["optim", "sched"],
    }
    return g


def test_build_training_plan_toy() -> None:
    g = _toy_training_graph()
    plan = build_training_plan(g)
    assert plan.loss_node_id == "mse"
    assert plan.forward_node_ids == ("toy", "mse")
    assert plan.post_backward_node_ids == ("optim", "sched")
    sig = training_plan_signature(g)
    assert "toy_train" in sig
    assert "mse" in sig


def test_training_graph_validates() -> None:
    g = _toy_training_graph()
    r = validate(g)
    assert r.valid, r.errors


def test_run_training_step_toy_reduces_loss() -> None:
    g = _toy_training_graph(dim=4)
    toy = g.get_node("toy")
    assert toy is not None
    toy.to(torch.device("cpu"))
    opt = torch.optim.AdamW(toy.trainable_parameters(), lr=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    ctx = TrainingStepContext(
        optimizer=opt,
        lr_scheduler=sched,
        trainable_parameters=list(toy.trainable_parameters()),
        grad_accumulation_steps=1,
        max_grad_norm=None,
        scaler=None,
    )
    x = torch.randn(2, 4)
    losses = []
    for i in range(15):
        out = run_training_step(g, {"x": x}, ctx, validate_before=(i == 0))
        losses.append(out.loss)
    assert losses[-1] < losses[0]


def test_gradient_accumulation_optimizer_not_every_step() -> None:
    g = _toy_training_graph()
    toy = g.get_node("toy")
    assert toy is not None
    toy.to(torch.device("cpu"))
    opt = torch.optim.AdamW(toy.trainable_parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    ctx = TrainingStepContext(
        optimizer=opt,
        lr_scheduler=sched,
        trainable_parameters=list(toy.trainable_parameters()),
        grad_accumulation_steps=3,
        max_grad_norm=None,
        scaler=None,
        micro_step=0,
        global_step=0,
    )
    x = torch.randn(1, 3)
    gs = []
    for _ in range(6):
        out = run_training_step(g, {"x": x}, ctx, validate_before=False)
        gs.append(out.global_step)
    assert gs == [0, 0, 1, 1, 1, 2]


def test_build_diffusion_lora_training_hypergraph_smoke() -> None:
    class _Obj:
        def compute_loss(self, batch):
            t = batch["pixel_values"]
            return (t ** 2).mean()

    g = build_diffusion_lora_training_hypergraph(objective=_Obj())
    r = validate(g)
    assert r.valid, r.errors
    loss_node = g.get_node("loss")
    assert isinstance(loss_node, DiffusionLoRALossConverter)
