"""Single-step SD1.5 LoRA training objective."""
from __future__ import annotations

from typing import Any, Dict

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


class SD15LoRAObjective:
    """Minimal diffusion training step: noisy latents -> UNet -> MSE loss."""

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

    @staticmethod
    def _module_device_dtype(module: Any) -> tuple[Any, Any]:
        import torch

        parameter = next(iter(module.parameters()), None) if hasattr(module, "parameters") else None
        if parameter is not None:
            return parameter.device, parameter.dtype
        device = getattr(module, "device", torch.device("cpu"))
        dtype = getattr(module, "dtype", torch.float32)
        return device, dtype

    def _encode_prompts(self, captions: list[str]) -> Any:
        unet_device, unet_dtype = self._module_device_dtype(self._unet)
        tokens = self._tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self._tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(unet_device)
        if self._config.train_text_encoder:
            return self._text_encoder(input_ids)[0].to(device=unet_device, dtype=unet_dtype)
        import torch
        with torch.no_grad():
            return self._text_encoder(input_ids)[0].to(device=unet_device, dtype=unet_dtype)

    def compute_loss(self, batch: Dict[str, Any]) -> Any:
        import torch
        import torch.nn.functional as F

        vae_device, vae_dtype = self._module_device_dtype(self._vae)
        unet_device, unet_dtype = self._module_device_dtype(self._unet)
        pixel_values = batch["pixel_values"].to(device=vae_device, dtype=vae_dtype)
        captions = list(batch["caption"])

        with torch.no_grad():
            latents = self._vae.encode(pixel_values).latent_dist.sample()
            scaling_factor = getattr(self._vae.config, "scaling_factor", 0.18215)
            latents = latents * scaling_factor
            latents = latents.to(device=unet_device, dtype=unet_dtype)

        prompt_embeds = self._encode_prompts(captions)
        noise = torch.randn_like(latents)

        num_train_timesteps = int(getattr(self._scheduler.config, "num_train_timesteps", 1000))
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = self._scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self._unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample

        prediction_type = getattr(self._scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            if not hasattr(self._scheduler, "get_velocity"):
                raise RuntimeError("Scheduler uses v_prediction but does not provide get_velocity()")
            target = self._scheduler.get_velocity(latents, noise, timesteps)
        elif prediction_type == "sample":
            target = latents
        else:
            raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type!r}")

        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
