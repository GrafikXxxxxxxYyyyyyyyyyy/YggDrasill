"""Single-step FLUX text2img LoRA training objective."""
from __future__ import annotations

from typing import Any, Dict

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.objective_utils import (
    build_flux_img_ids,
    encode_flux_prompt,
    module_device_dtype,
    resolve_diffusion_target,
    sample_flux_latent_representation,
    vae_encode_latents,
)
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


class FluxText2ImgLoRAObjective:
    """FLUX text2img objective over packed latents and transformer conditioning."""

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
        self._backbone = targets.backbone
        self._vae = components.vae
        self._scheduler = components.scheduler
        self._config = config
        self._device = device

    def compute_loss(self, batch: Dict[str, Any]) -> Any:
        import torch.nn.functional as F

        backbone_device, backbone_dtype = module_device_dtype(self._backbone)
        latents = vae_encode_latents(
            vae=self._vae,
            pixel_values=batch["pixel_values"],
            device=backbone_device,
            dtype=backbone_dtype,
        )
        conditioning = encode_flux_prompt(
            tokenizer=self._tokenizer,
            tokenizer_2=self._tokenizer_2,
            text_encoder=self._text_encoder,
            text_encoder_2=self._text_encoder_2,
            captions=list(batch["caption"]),
            captions_2=list(batch.get("prompt_2", batch["caption"])),
            train_text_encoder=self._config.train_text_encoder,
            train_text_encoder_2=self._config.train_text_encoder_2,
            device=backbone_device,
            dtype=backbone_dtype,
        )
        latent_repr = sample_flux_latent_representation(scheduler=self._scheduler, latents=latents)
        img_ids = build_flux_img_ids(latents=latents, device=backbone_device, dtype=backbone_dtype)

        kwargs: Dict[str, Any] = {
            "hidden_states": latent_repr.packed_latents.to(device=backbone_device, dtype=backbone_dtype),
            "timestep": latent_repr.timesteps.to(device=backbone_device, dtype=backbone_dtype) / 1000,
            "encoder_hidden_states": conditioning.encoder_hidden_states,
            "pooled_projections": conditioning.pooled_prompt_embeds,
            "img_ids": img_ids,
            "txt_ids": conditioning.extra["txt_ids"],
            "return_dict": False,
        }
        if self._config.guidance is not None and hasattr(getattr(self._backbone, "config", None), "guidance_embeds"):
            kwargs["guidance"] = latent_repr.timesteps.new_full(
                (latent_repr.timesteps.shape[0],),
                float(self._config.guidance),
            )

        output = self._backbone(**kwargs)
        noise_pred = output[0] if isinstance(output, (tuple, list)) else getattr(output, "sample", output)
        target = resolve_diffusion_target(
            scheduler=self._scheduler,
            latents=latent_repr.extra["packed_clean_latents"],
            noise=latent_repr.noise.to(device=backbone_device, dtype=backbone_dtype),
            timesteps=latent_repr.timesteps,
        )
        target = target.reshape_as(noise_pred).to(device=backbone_device, dtype=backbone_dtype)
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
