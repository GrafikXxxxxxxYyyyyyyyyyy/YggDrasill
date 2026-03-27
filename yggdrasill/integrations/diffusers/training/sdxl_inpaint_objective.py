"""Single-step SDXL inpaint LoRA training objective."""
from __future__ import annotations

from typing import Any, Dict

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.objective_utils import (
    build_sdxl_time_ids,
    encode_sdxl_prompt,
    module_device_dtype,
    resize_mask_to_latents,
    resolve_diffusion_target,
    sample_noise_schedule,
    vae_encode_latents,
)
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


class SDXLInpaintLoRAObjective:
    """SDXL inpaint objective supporting both 4ch and 9ch UNets."""

    def __init__(
        self,
        *,
        components: TrainingComponents,
        targets: TrainingTargetSetup,
        config: TrainingConfig,
        device: Any,
    ) -> None:
        self._tokenizer = components.tokenizer
        self._tokenizer_2 = components.tokenizer_2
        self._text_encoder = targets.text_encoder
        self._text_encoder_2 = targets.text_encoder_2 or components.text_encoder_2
        self._unet = targets.unet
        self._vae = components.vae
        self._scheduler = components.scheduler
        self._config = config
        self._device = device

    def compute_loss(self, batch: Dict[str, Any]) -> Any:
        import torch
        import torch.nn.functional as F

        unet_device, unet_dtype = module_device_dtype(self._unet)
        latents = vae_encode_latents(
            vae=self._vae,
            pixel_values=batch["pixel_values"],
            device=unet_device,
            dtype=unet_dtype,
        )
        masked_pixels = batch.get("masked_pixel_values", batch["init_pixel_values"])
        masked_latents = vae_encode_latents(
            vae=self._vae,
            pixel_values=masked_pixels,
            device=unet_device,
            dtype=unet_dtype,
        )
        mask = resize_mask_to_latents(batch["mask_values"].to(device=unet_device, dtype=unet_dtype), latents)
        prompt_embeds, pooled_embeds = encode_sdxl_prompt(
            tokenizer=self._tokenizer,
            tokenizer_2=self._tokenizer_2,
            text_encoder=self._text_encoder,
            text_encoder_2=self._text_encoder_2,
            captions=list(batch["caption"]),
            captions_2=list(batch.get("prompt_2", batch["caption"])),
            train_text_encoder=self._config.train_text_encoder,
            train_text_encoder_2=self._config.train_text_encoder_2,
            device=unet_device,
            dtype=unet_dtype,
        )
        noise, timesteps, noisy_latents = sample_noise_schedule(scheduler=self._scheduler, latents=latents)
        time_ids = build_sdxl_time_ids(
            batch_size=latents.shape[0],
            original_size=self._config.original_size or (self._config.resolution, self._config.resolution),
            target_size=self._config.target_size or (self._config.resolution, self._config.resolution),
            crops_coords_top_left=self._config.crops_coords_top_left,
            device=unet_device,
            dtype=unet_dtype,
            requires_aesthetics_score=self._config.requires_aesthetics_score,
            aesthetic_score=self._config.aesthetic_score,
            aesthetic_scores=batch.get("aesthetic_score"),
        )
        unet_input = noisy_latents
        in_channels = int(getattr(getattr(self._unet, "config", None), "in_channels", 4))
        if in_channels == 9:
            unet_input = torch.cat([noisy_latents[:, :4], mask[:, :1], masked_latents[:, :4]], dim=1)
        noise_pred = self._unet(
            unet_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
        ).sample
        target = resolve_diffusion_target(
            scheduler=self._scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
