"""Diffusion run wrapper: prepares diffusion-specific kwargs and wraps output in DiffusionOutput."""
from __future__ import annotations

from typing import Any, Dict, Optional

from yggdrasill.integrations.diffusers.output import DiffusionOutput


def run(
    graph: Any,
    inputs: Optional[Dict[str, Any]] = None,
    *,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[Any] = None,
    wrap_output: bool = True,
    **kwargs: Any,
) -> DiffusionOutput | Dict[str, Any]:
    """Run a diffusion graph with optional DiffusionOutput wrapping.

    Args:
        graph: Hypergraph or Workflow to run.
        inputs: Input dict (e.g. prompt, negative_prompt).
        num_inference_steps: Override for denoising steps.
        seed: Random seed for latent init.
        device: Target device.
        wrap_output: If True, return DiffusionOutput; otherwise raw dict.
        **kwargs: Passed through to graph.run().

    Returns:
        DiffusionOutput when wrap_output=True, else raw executor dict.
    """
    run_kw = dict(kwargs)
    if num_inference_steps is not None:
        run_kw["num_inference_steps"] = num_inference_steps
    if seed is not None:
        run_kw["seed"] = seed
    if device is not None:
        run_kw["device"] = device

    _prepare_diffusion_run(graph, run_kw)
    raw = graph.run(inputs, **run_kw)

    if wrap_output:
        return DiffusionOutput.from_executor_output(raw)
    return raw


def _prepare_diffusion_run(graph: Any, run_kwargs: Dict[str, Any]) -> None:
    """In-place preparation for diffusion run (device, node config overrides).

    Call before graph.run() when you need to inject num_inference_steps,
    seed, or device into specific node configs. The engine's run() already
    routes num_inference_steps→num_loop_steps and seed to the executor;
    this hook is for any extra diffusion-specific setup.
    """
    device = run_kwargs.get("device")
    if device is not None and hasattr(graph, "to") and callable(getattr(graph, "to")):
        graph.to(device)
