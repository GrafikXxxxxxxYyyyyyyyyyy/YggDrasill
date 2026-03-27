"""Single-step SD1.5 img2img LoRA training objective."""
from __future__ import annotations

from typing import Any, Dict

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.objective_utils import (
    encode_sd15_prompt,
    module_device_dtype,
    resolve_diffusion_target,
    sample_noise_schedule,
    vae_encode_latents,
)
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


class SD15Img2ImgLoRAObjective:
    """SD1.5 img2img training step driven by init-image latents."""

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
        import torch.nn.functional as F

        unet_device, unet_dtype = module_device_dtype(self._unet)
        latents = vae_encode_latents(
            vae=self._vae,
            pixel_values=batch["init_pixel_values"],
            device=unet_device,
            dtype=unet_dtype,
        )
        prompt_embeds = encode_sd15_prompt(
            tokenizer=self._tokenizer,
            text_encoder=self._text_encoder,
            captions=list(batch["caption"]),
            train_text_encoder=self._config.train_text_encoder,
            device=unet_device,
            dtype=unet_dtype,
        )
        noise, timesteps, noisy_latents = sample_noise_schedule(scheduler=self._scheduler, latents=latents)
        noise_pred = self._unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample
        target = resolve_diffusion_target(
            scheduler=self._scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
