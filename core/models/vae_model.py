import torch

from typing import Optional, Tuple
from diffusers import AutoencoderKL



class VaeModel:
    vae: AutoencoderKL

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        # Инициализируем модель, можно добавить выбор различных VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_path, 
            subfolder="vae",
            torch_dtype=dtype,
            variant='fp16', 
            use_safetensors=True
        )
        self.vae.to(device=device, dtype=dtype)

        self.model_path = model_path
        self.model_type = model_type or "sd15"
        print(f"VAE model has successfully loaded from '{model_path}' checkpoint!")

    @property
    def config(self) -> int:
        return self.vae.config
    
    @property
    def dtype(self) -> type:
        return self.vae.dtype

    @property
    def device(self) -> torch.device:
        return self.vae.device

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** (len(self.config.block_out_channels) - 1)
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # ################################################################################################################ #
    # Основной функционал модели VAE
    # ################################################################################################################ #
    def retrieve_latents(
        self,
        encoder_output: torch.Tensor, 
        generator: Optional[torch.Generator] = None, 
        sample_mode: str = "sample",
    ) -> torch.Tensor:
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")


    def encode(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:  
        """
        По сути просто обёртка над методом .encode() оригинального энкодера, 
        которая делает upcast vae при необходимости
        """
        _dtype = image.dtype

        if self.config.force_upcast:
            image = image.float()
            self.to(dtype=torch.float32)

        latents = self.retrieve_latents(
            self.vae.encode(image), 
            generator=generator
        )
        latents = latents.to(_dtype)
        latents = self.config.scaling_factor * latents

        if self.config.force_upcast:
            self.to(dtype=_dtype)

        return latents

    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:  
        # unscale/denormalize the latents denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.config, "latents_mean") and self.config.latents_mean
        has_latents_std = hasattr(self.config, "latents_std") and self.config.latents_std

        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.config.latents_mean).view(1, 4, 1, 1).to(
                latents.device, latents.dtype
            )
            latents_std = torch.tensor(self.config.latents_std).view(1, 4, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_std / self.config.scaling_factor + latents_mean
        else:
            latents = latents / self.config.scaling_factor

        images = self.vae.decode(latents, return_dict=False)[0]

        return images


    def get_processed_latents_or_images(
        self,
        images: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[
        Optional[torch.FloatTensor],
        Optional[torch.FloatTensor],
    ]:
        encoded_images = images
        if images is not None:
            images = images.to(device=self.device, dtype=self.dtype)
            encoded_images = self.encode(images, generator)

        decoded_images = latents
        if latents is not None:
            decoded_images = self.decode(latents)
        
        return encoded_images, decoded_images
    # ################################################################################################################ #



    # ================================================================================================================ #
    def __call__(
        self,
        images: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[
        Optional[torch.FloatTensor],
        Optional[torch.FloatTensor],
    ]:
    # ================================================================================================================ #
        """
        Получает на вход изображение и/или латенты 
        В зависимости от входа кодирует и/или декодирует данные
        """
        return self.get_processed_latents_or_images(
            images=images,
            latents=latents,
            generator=generator,
            **kwargs,
        )
    # ================================================================================================================ #







