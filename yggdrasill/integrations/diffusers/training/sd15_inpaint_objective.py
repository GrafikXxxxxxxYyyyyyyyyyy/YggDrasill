"""Single-step SD1.5 inpaint LoRA training objective."""
from __future__ import annotations

from typing import Any, Dict

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.objective_utils import (
    encode_sd15_prompt,
    module_device_dtype,
    resize_mask_to_latents,
    resolve_diffusion_target,
    sample_noise_schedule,
    vae_encode_latents,
)
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


class SD15InpaintLoRAObjective:
    """SD1.5 inpaint training objective for both 4ch and 9ch UNets."""

    def __init__(
        self,
        *,
        components: TrainingComponents,
        targets: TrainingTargetSetup,
        config: TrainingConfig,
        device: Any,
    ) -> None:
        self._tokenizer = components.tokenizer
        self._text_encoder = targets.text_encoder
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
        prompt_embeds = encode_sd15_prompt(
            tokenizer=self._tokenizer,
            text_encoder=self._text_encoder,
            captions=list(batch["caption"]),
            train_text_encoder=self._config.train_text_encoder,
            device=unet_device,
            dtype=unet_dtype,
        )
        noise, timesteps, noisy_latents = sample_noise_schedule(scheduler=self._scheduler, latents=latents)
        unet_input = noisy_latents
        in_channels = int(getattr(getattr(self._unet, "config", None), "in_channels", 4))
        kwargs = {"encoder_hidden_states": prompt_embeds}
        if in_channels == 9:
            unet_input = torch.cat([noisy_latents[:, :4], mask[:, :1], masked_latents[:, :4]], dim=1)
        noise_pred = self._unet(unet_input, timesteps, **kwargs).sample
        target = resolve_diffusion_target(
            scheduler=self._scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
