"""Execute one training step: forward (through loss), backward, post-backward helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from yggdrasill.engine.buffers import EdgeBuffers
from yggdrasill.engine.executor import ValidationError, _gather_node_inputs, _run_node_outputs, validate
from yggdrasill.engine.planner import build_training_plan
from yggdrasill.training.context import TrainingStepContext, TrainingStepOutcome


def run_training_step(
    structure: Any,
    inputs: Dict[str, Any],
    ctx: TrainingStepContext,
    *,
    validate_before: bool = True,
    dry_run: bool = False,
) -> TrainingStepOutcome:
    """Run forward through *loss* node, backward, then post-backward helper chain.

    *structure* must define ``metadata['training']`` (see :func:`build_training_plan`).
    *inputs* are merged like :func:`yggdrasill.engine.executor.run` (exposed inputs + buffer keys).
    """
    if validate_before:
        result = validate(structure)
        if not result.valid:
            raise ValidationError(result.errors)

    plan = build_training_plan(structure)
    input_spec = structure.get_input_spec()
    buf = EdgeBuffers.init_from_inputs(input_spec, inputs)

    for nid in plan.forward_node_ids:
        _execute_training_forward_node(structure, nid, buf, input_spec, dry_run, ctx)

    loss_t = buf.read(plan.loss_node_id, plan.loss_port)
    if loss_t is None:
        raise RuntimeError(
            f"Loss tensor missing at buffer[{plan.loss_node_id!r}][{plan.loss_port!r}]"
        )

    grad_accum = max(1, int(ctx.grad_accumulation_steps))
    loss_scaled = loss_t / float(grad_accum)

    if not torch.isfinite(loss_scaled.detach()).all():
        raise RuntimeError("Non-finite training loss before backward().")

    if ctx.scaler is not None:
        ctx.scaler.scale(loss_scaled).backward()
    else:
        loss_scaled.backward()

    ctx.last_loss = float(loss_t.detach().item())

    for nid in plan.post_backward_node_ids:
        _execute_training_post_node(structure, nid, buf, input_spec, dry_run, ctx)

    optimizer_ran = bool(ctx.last_optimizer_ran)
    ctx.micro_step += 1
    if optimizer_ran:
        ctx.global_step += 1

    buf.scratch["last_training_loss"] = ctx.last_loss
    buf.scratch["last_optimizer_ran"] = optimizer_ran

    return TrainingStepOutcome(
        loss=float(ctx.last_loss or 0.0),
        optimizer_ran=optimizer_ran,
        global_step=ctx.global_step,
        micro_step=ctx.micro_step,
        should_stop=False,
    )


def _execute_training_forward_node(
    structure: Any,
    node_id: str,
    buf: EdgeBuffers,
    input_spec: Any,
    dry_run: bool,
    ctx: TrainingStepContext,
) -> None:
    node = structure.get_node(node_id)
    if node is None:
        return
    node_inputs = _gather_node_inputs(structure, node_id, buf, input_spec)
    acm = ctx.autocast_cm
    if acm is not None:
        with acm():
            outputs = _run_node_outputs(node, node_inputs, dry_run=dry_run, training=True)
    else:
        outputs = _run_node_outputs(node, node_inputs, dry_run=dry_run, training=True)
    for port_name, value in outputs.items():
        buf.write(node_id, port_name, value)


def _execute_training_post_node(
    structure: Any,
    node_id: str,
    buf: EdgeBuffers,
    input_spec: Any,
    dry_run: bool,
    ctx: TrainingStepContext,
) -> None:
    node = structure.get_node(node_id)
    if node is None:
        return
    node_inputs = _gather_node_inputs(structure, node_id, buf, input_spec)
    node_inputs["training_step"] = ctx
    outputs = _run_node_outputs(node, node_inputs, dry_run=dry_run, training=True)
    for port_name, value in outputs.items():
        buf.write(node_id, port_name, value)


def fit_training_graph(
    structure: Any,
    batch_iter,
    ctx: TrainingStepContext,
    *,
    max_train_steps: Optional[int] = None,
    validate_before: bool = True,
) -> TrainingStepContext:
    """Iterate *batch_iter* via :func:`yggdrasill.engine.executor.run` (``run_mode='train'``)."""
    from yggdrasill.engine.executor import run

    payload: Dict[str, Any] = {
        "training_step_context": ctx,
        "training_batch_iter": batch_iter,
    }
    if max_train_steps is not None:
        payload["max_train_steps"] = max_train_steps
    run(structure, payload, run_mode="train", validate_before=validate_before)
    return ctx


def resume_training(
    structure: Any,
    checkpoint_dir: Union[str, Path],
    ctx: TrainingStepContext,
    *,
    map_location: str = "cpu",
) -> TrainingStepContext:
    """Load optimizer/scheduler/scaler state and restore step counters from a trainer checkpoint."""
    from yggdrasill.integrations.diffusers.training.checkpointing import load_training_state

    ckpt = Path(checkpoint_dir)
    gs = load_training_state(
        ckpt,
        optimizer=ctx.optimizer,
        lr_scheduler=ctx.lr_scheduler,
        scaler=ctx.scaler,
        map_location=map_location,
    )
    ctx.global_step = int(gs)
    grad = max(1, int(ctx.grad_accumulation_steps))
    ctx.micro_step = ctx.global_step * grad if ctx.global_step > 0 else 0
    _ = structure  # reserved for future training-plan / checkpoint consistency checks
    return ctx
