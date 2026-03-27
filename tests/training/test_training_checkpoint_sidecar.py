"""Training graph sidecar next to diffusion checkpoints."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from yggdrasill.integrations.diffusers.training.checkpointing import save_training_state
from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.training.checkpoint import read_training_graph_sidecar


def test_save_training_state_writes_training_graph_json(tmp_path: Path) -> None:
    import torch

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    cfg = TrainingConfig(
        pretrained_model_name_or_path="stub",
        data_dir=str(tmp_path),
        family="sd15",
        task="text2img",
        output_dir=str(tmp_path / "out"),
    )
    save_training_state(
        tmp_path / "ckpt",
        global_step=5,
        optimizer=opt,
        lr_scheduler=sched,
        scaler=None,
        config=cfg,
        training_plan_signature="diffusion_lora_training:loss:loss",
    )
    side = read_training_graph_sidecar(tmp_path / "ckpt")
    assert side is not None
    assert side["global_step"] == 5
    assert "loss" in side["training_plan_signature"]
