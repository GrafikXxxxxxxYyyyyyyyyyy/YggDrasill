"""Shared helpers for diffusion training objectives."""
from __future__ import annotations

from typing import Any, Iterable

from yggdrasill.integrations.diffusers.training.types import ConditioningBundle, LatentRepresentation

def module_device_dtype(module: Any) -> tuple[Any, Any]:
    import torch

    parameter = next(iter(module.parameters()), None) if hasattr(module, "parameters") else None
    if parameter is not None:
        return parameter.device, parameter.dtype
    device = getattr(module, "device", torch.device("cpu"))
    dtype = getattr(module, "dtype", torch.float32)
    return device, dtype


def _autocast_disabled_if_needed(*, device: Any, disable: bool) -> Any:
    import contextlib
    import torch

    if not disable:
        return contextlib.nullcontext()
    if getattr(device, "type", None) != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", enabled=False)


def vae_encode_latents(*, vae: Any, pixel_values: Any, device: Any, dtype: Any) -> Any:
    import torch

    vae_device, vae_dtype = module_device_dtype(vae)
    parameter = next(iter(vae.parameters()), None) if hasattr(vae, "parameters") else None
    force_upcast = bool(getattr(getattr(vae, "config", None), "force_upcast", False))
    prep_dtype = torch.float32 if force_upcast else vae_dtype
    pixels = pixel_values.to(device=vae_device, dtype=prep_dtype)
    original_vae_dtype = vae_dtype
    did_fp32_encode = False

    if force_upcast and parameter is not None and vae_dtype == torch.float16:
        vae.to(dtype=torch.float32)
        pixels = pixels.float()
        did_fp32_encode = True
    elif force_upcast:
        pixels = pixels.float()

    with torch.no_grad():
        latents = vae.encode(pixels).latent_dist.sample()
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        if scaling_factor is None:
            scaling_factor = 0.18215
        shift_factor = getattr(vae.config, "shift_factor", 0.0)
        if shift_factor is None:
            shift_factor = 0.0
        latents = (latents - shift_factor) * scaling_factor

    if did_fp32_encode and original_vae_dtype is not None:
        vae.to(dtype=original_vae_dtype)

    # Always return float32 for training stability; autocast handles model
    # forward passes, but scheduler math (add_noise, get_velocity) must not
    # run in fp16 to avoid overflow on large timesteps.
    return latents.to(device=device, dtype=torch.float32)


def encode_sd15_prompt(*, tokenizer: Any, text_encoder: Any, captions: list[str], train_text_encoder: bool, device: Any, dtype: Any) -> Any:
    import torch

    tokens = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    with _autocast_disabled_if_needed(device=device, disable=train_text_encoder):
        if train_text_encoder:
            return text_encoder(input_ids)[0].to(device=device).float()
        with torch.no_grad():
            return text_encoder(input_ids)[0].to(device=device).float()


def encode_sdxl_prompt(
    *,
    tokenizer: Any,
    tokenizer_2: Any,
    text_encoder: Any,
    text_encoder_2: Any,
    captions: list[str],
    captions_2: list[str],
    train_text_encoder: bool,
    train_text_encoder_2: bool,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    import torch

    def _tokenize(tok: Any, texts: list[str]) -> Any:
        return tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=tok.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)

    ids_1 = _tokenize(tokenizer, captions)
    ids_2 = _tokenize(tokenizer_2, captions_2)

    def _encode_one(encoder: Any, input_ids: Any, trainable: bool) -> tuple[Any, Any]:
        with _autocast_disabled_if_needed(device=device, disable=trainable):
            if trainable:
                output = encoder(input_ids, output_hidden_states=True)
            else:
                with torch.no_grad():
                    output = encoder(input_ids, output_hidden_states=True)
        hidden = output.hidden_states[-2]
        pooled = getattr(output, "text_embeds", None)
        if pooled is None:
            pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            base = output[0]
            pooled = base if getattr(base, "ndim", 0) == 2 else None
        if pooled is None:
            raise RuntimeError("SDXL text encoder did not produce pooled prompt embeddings")
        return hidden.to(device=device).float(), pooled.to(device=device).float()

    hidden_1, _ = _encode_one(text_encoder, ids_1, train_text_encoder)
    hidden_2, pooled = _encode_one(text_encoder_2, ids_2, train_text_encoder_2)
    return torch.cat([hidden_1, hidden_2], dim=-1), pooled


def sample_noise_schedule(*, scheduler: Any, latents: Any) -> tuple[Any, Any, Any]:
    import torch

    latents = latents.float()
    noise = torch.randn_like(latents)
    num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    timesteps = torch.randint(
        0,
        num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
        dtype=torch.long,
    )
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    return noise, timesteps, noisy_latents


def resolve_diffusion_target(*, scheduler: Any, latents: Any, noise: Any, timesteps: Any) -> Any:
    latents = latents.float()
    noise = noise.float()
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        if not hasattr(scheduler, "get_velocity"):
            raise RuntimeError("Scheduler uses v_prediction but does not provide get_velocity()")
        return scheduler.get_velocity(latents, noise, timesteps)
    if prediction_type == "sample":
        return latents
    raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type!r}")


def build_sdxl_time_ids(
    *,
    batch_size: int,
    original_size: tuple[int, int],
    target_size: tuple[int, int],
    crops_coords_top_left: tuple[int, int],
    device: Any,
    dtype: Any,
    requires_aesthetics_score: bool = False,
    aesthetic_score: float = 6.0,
    aesthetic_scores: Any = None,
) -> Any:
    import torch

    f32 = torch.float32
    if requires_aesthetics_score:
        if aesthetic_scores is not None:
            if hasattr(aesthetic_scores, "to"):
                scores = aesthetic_scores.to(device=device, dtype=f32).reshape(batch_size, 1)
            else:
                scores = torch.tensor(aesthetic_scores, device=device, dtype=f32).reshape(batch_size, 1)
            prefix = torch.tensor(
                [*original_size, *crops_coords_top_left],
                device=device,
                dtype=f32,
            ).unsqueeze(0).expand(batch_size, -1)
            return torch.cat([prefix, scores], dim=1)
        values: Iterable[float] = [*original_size, *crops_coords_top_left, aesthetic_score]
    else:
        values = [*original_size, *crops_coords_top_left, *target_size]
    time_ids = torch.tensor(list(values), device=device, dtype=f32)
    return time_ids.unsqueeze(0).expand(batch_size, -1)


def resize_mask_to_latents(mask_values: Any, latents: Any) -> Any:
    import torch.nn.functional as F

    if mask_values.shape[-2:] == latents.shape[-2:]:
        return mask_values[:, :1]
    return F.interpolate(mask_values[:, :1], size=latents.shape[-2:], mode="nearest")


def encode_flux_prompt(
    *,
    tokenizer: Any,
    tokenizer_2: Any,
    text_encoder: Any,
    text_encoder_2: Any,
    captions: list[str],
    captions_2: list[str],
    train_text_encoder: bool,
    train_text_encoder_2: bool,
    device: Any,
    dtype: Any,
    max_sequence_length: int = 512,
) -> ConditioningBundle:
    import torch

    clip_tokens = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    t5_tokens = tokenizer_2(
        captions_2,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    clip_ids = clip_tokens.input_ids.to(device)
    t5_ids = t5_tokens.input_ids.to(device)

    if train_text_encoder:
        clip_output = text_encoder(clip_ids, output_hidden_states=False)
    else:
        with torch.no_grad():
            clip_output = text_encoder(clip_ids, output_hidden_states=False)
    pooled = getattr(clip_output, "pooler_output", None)
    if pooled is None:
        pooled = getattr(clip_output, "text_embeds", None)
    if pooled is None:
        pooled = clip_output[0]

    if train_text_encoder_2:
        t5_output = text_encoder_2(t5_ids)
    else:
        with torch.no_grad():
            t5_output = text_encoder_2(t5_ids)
    prompt_embeds = t5_output[0]
    seq_len = prompt_embeds.shape[1]
    txt_ids = torch.zeros((seq_len, 3), device=device, dtype=torch.float32)

    return ConditioningBundle(
        encoder_hidden_states=prompt_embeds.to(device=device).float(),
        pooled_prompt_embeds=pooled.to(device=device).float(),
        extra={"txt_ids": txt_ids},
    )


def pack_flux_latents(latents: Any) -> Any:
    """Pack [B, C, H, W] -> [B, (H/2)*(W/2), C*4] via 2x2 patches."""
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def build_flux_img_ids(*, latents: Any, device: Any, dtype: Any) -> Any:
    import torch

    f32 = torch.float32
    h = latents.shape[-2] // 2
    w = latents.shape[-1] // 2
    img_ids = torch.zeros(h, w, 3, device=device, dtype=f32)
    img_ids[..., 1] = torch.arange(h, device=device, dtype=f32)[:, None]
    img_ids[..., 2] = torch.arange(w, device=device, dtype=f32)[None, :]
    return img_ids.reshape(h * w, 3)


def sample_flux_latent_representation(*, scheduler: Any, latents: Any) -> LatentRepresentation:
    import torch

    latents = latents.float()
    noise = torch.randn_like(latents)
    if hasattr(scheduler, "add_noise"):
        num_train_timesteps = int(getattr(getattr(scheduler, "config", None), "num_train_timesteps", 1000))
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    elif hasattr(scheduler, "scale_noise"):
        timesteps = torch.rand((latents.shape[0],), device=latents.device, dtype=latents.dtype)
        noisy_latents = scheduler.scale_noise(latents, timesteps, noise)
    else:
        timesteps = torch.rand((latents.shape[0],), device=latents.device, dtype=latents.dtype)
        noisy_latents = latents + noise
    return LatentRepresentation(
        clean_latents=latents,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        noise=noise,
        packed_latents=pack_flux_latents(noisy_latents),
        extra={"packed_clean_latents": pack_flux_latents(latents)},
    )
