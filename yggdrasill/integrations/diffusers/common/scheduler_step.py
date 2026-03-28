"""Scheduler.step() timestep conventions (int vs float tensor) across Diffusers classes."""
from __future__ import annotations

from typing import Any, FrozenSet

# Schedulers that reject int/LongTensor in step() and expect values from ``.timesteps`` (often float).
_FLOAT_TIMESTEP_STEP_SCHEDULERS: FrozenSet[str] = frozenset({
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "HeunDiscreteScheduler",
    "UniPCMultistepScheduler",
    "DEISMultistepScheduler",
    "DPMSolverSDEScheduler",
    "EDMEulerScheduler",
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchHeunDiscreteScheduler",
    "FlowMatchLCMScheduler",
    "ConsistencyDecoderScheduler",
})


def scheduler_uses_float_timestep_in_step(scheduler: Any) -> bool:
    """True if ``scheduler.step(..., timestep=...)`` must not receive int/long (Euler-family, flow-match, …)."""
    if scheduler is None:
        return False
    return type(scheduler).__name__ in _FLOAT_TIMESTEP_STEP_SCHEDULERS


def scheduler_reference_timestep_dtype(scheduler: Any) -> Any:
    """Dtype for *timestep* tensors passed to ``scale_model_input`` / ``step`` for float-timestep schedulers.

    Must match ``scheduler.timesteps.dtype`` (almost always float32). Casting to the UNet's fp16 dtype
    breaks ``EulerDiscreteScheduler.index_for_timestep`` (no exact match → empty indices → IndexError).
    """
    import torch

    ts = getattr(scheduler, "timesteps", None)
    if ts is not None and hasattr(ts, "dtype") and hasattr(ts, "numel") and ts.numel() > 0:
        return ts.dtype
    return torch.float32


def coerce_timestep_for_scheduler_step(
    timestep: Any,
    latents: Any,
    scheduler: Any,
) -> Any:
    """Return *timestep* suitable for ``scheduler.step`` (float tensor vs int)."""
    import torch

    if not scheduler_uses_float_timestep_in_step(scheduler):
        return timestep

    device = latents.device if hasattr(latents, "device") else "cpu"
    ts_ref = getattr(scheduler, "timesteps", None)
    target_dtype = (
        ts_ref.dtype
        if ts_ref is not None and hasattr(ts_ref, "dtype")
        else torch.float32
    )

    if isinstance(timestep, torch.Tensor):
        t = timestep.to(device=device)
        if t.ndim > 0:
            t = t.reshape(-1)[0]
        return t.to(dtype=target_dtype)

    if hasattr(timestep, "item"):
        try:
            v = timestep.item()
        except Exception:
            v = float(timestep)
    else:
        v = float(timestep)
    return torch.tensor(v, device=device, dtype=target_dtype)


def coerce_timestep_for_add_noise(
    timestep_value: Any,
    *,
    scheduler: Any,
    device: Any,
) -> Any:
    """1-D tensor ``(1,)`` for ``scheduler.add_noise(..., timestep=...)``.

    Diffusers inpaint (4-ch UNet) uses ``torch.tensor([timesteps[i + 1]])``; dtype must match
    ``scheduler.timesteps`` (float for Euler-family, long for DDIM-style), not forced ``long``.
    """
    import torch

    ts = getattr(scheduler, "timesteps", None)
    if ts is not None and isinstance(ts, torch.Tensor) and ts.numel() > 0:
        target_dtype = ts.dtype
    elif scheduler_uses_float_timestep_in_step(scheduler):
        target_dtype = torch.float32
    else:
        target_dtype = torch.long

    if isinstance(timestep_value, torch.Tensor):
        v = timestep_value.reshape(-1)[0].to(device=device, dtype=target_dtype)
    else:
        v = torch.tensor(timestep_value, device=device, dtype=target_dtype)
    return v.reshape(1)
