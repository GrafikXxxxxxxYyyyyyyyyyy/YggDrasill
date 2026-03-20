"""DiffusionGraphBuilder: Scheduler role and scheduler_type swap."""
from __future__ import annotations

import pytest

from yggdrasill.integrations.diffusers.presets.sd15 import build_sd15_text2img_graph

from tests.diffusion.conftest import (
    FakeScheduler,
    FakeTextEncoder,
    FakeTokenizer,
    FakeUNet,
    FakeVAE,
    requires_torch,
)


@pytest.fixture
def sd15_fake_graph():
    return build_sd15_text2img_graph(
        tokenizer=FakeTokenizer(),
        text_encoder=FakeTextEncoder(),
        unet=FakeUNet(),
        vae=FakeVAE(),
        scheduler=FakeScheduler(),
        config={"device": "cpu"},
    )


def test_resolve_scheduler_role_to_sched_base(sd15_fake_graph):
    from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder

    b = DiffusionGraphBuilder(sd15_fake_graph)
    assert b._resolve_scheduler_base_id() == "sched"
    assert b._resolve_role_to_node_id("Scheduler") == "sched"
    assert b._resolve_role_to_node_id("scheduler") == "sched"


@requires_torch
def test_replace_scheduler_with_euler_type():
    pytest.importorskip("diffusers")
    from diffusers import DDIMScheduler, EulerDiscreteScheduler

    from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder

    sched = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    g = build_sd15_text2img_graph(
        tokenizer=FakeTokenizer(),
        text_encoder=FakeTextEncoder(),
        unet=FakeUNet(),
        vae=FakeVAE(),
        scheduler=sched,
        config={"device": "cpu"},
    )
    b = DiffusionGraphBuilder(g)
    b.replace_component("Scheduler", "sd15.scheduler", scheduler_type="euler")

    setup = g.get_node("sched_setup")
    step = g.get_node("sched_step")
    assert setup is not None and step is not None
    assert isinstance(setup._scheduler, EulerDiscreteScheduler)
    assert setup._scheduler is step._scheduler
