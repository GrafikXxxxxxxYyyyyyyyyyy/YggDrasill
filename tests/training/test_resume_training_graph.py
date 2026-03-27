"""Resume step counters + optimizer state via :func:`resume_training`."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yggdrasill.integrations.diffusers.training.checkpointing import save_training_state
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.training.context import TrainingStepContext
from yggdrasill.training.executor import resume_training

import yggdrasill.training.blocks  # noqa: F401


def test_resume_training_restores_global_step_and_micro_step(tmp_path):
    """Optimizer state round-trip; micro_step aligned with grad accumulation."""
    from yggdrasill.hypergraph.structure import Hypergraph

    # ``resume_training`` only forwards *structure* for future consistency checks.
    g = Hypergraph("resume_test_dummy")

    p = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    opt = torch.optim.AdamW([p], lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    cfg = TrainingConfig(
        pretrained_model_name_or_path="stub",
        data_dir=str(tmp_path),
        family="sd15",
        task="text2img",
        output_dir=str(tmp_path / "out"),
    )
    ckpt_dir = tmp_path / "ckpt"
    save_training_state(
        ckpt_dir,
        global_step=7,
        optimizer=opt,
        lr_scheduler=sched,
        scaler=None,
        config=cfg,
        training_plan_signature="test:sig",
    )

    opt2 = torch.optim.AdamW([p], lr=0.05)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=lambda _: 1.0)
    ctx = TrainingStepContext(
        optimizer=opt2,
        lr_scheduler=sched2,
        trainable_parameters=[p],
        grad_accumulation_steps=4,
    )
    resume_training(g, ckpt_dir, ctx, map_location="cpu")
    assert ctx.global_step == 7
    assert ctx.micro_step == 7 * 4
