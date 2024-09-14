import torch

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from .models.vae_model import VaeModel
from .models.noise_predictor import ModelKey, NoisePredictor
from .pipelines.backward_diffusion import BackwardDiffusion, Conditions


@dataclass
class DiffusionModelKey(ModelKey):
    is_latent_model: bool = True
    scheduler_name: str = "euler"


class DiffusionModel:
    key: DiffusionModelKey
    predictor: NoisePredictor
    do_cfg: bool = False
    guidance_scale: float = 5.0
    vae: Optional[VaeModel] = None
    conditions: Optional[Conditions] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        is_latent_model: bool = False,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):     
        self.key = DiffusionModelKey(
            scheduler_name=scheduler_name,
            is_latent_model=is_latent_model,
            **kwargs,
        )
        self.predictor = NoisePredictor(**kwargs)
        self.vae = VaeModel(**kwargs) if is_latent_model else None

    @property
    def dtype(self):
        return self.predictor.dtype

    @property
    def device(self):
        return self.predictor.device

    @property
    def sample_size(self):
        return (
            self.predictor.config.sample_size * self.vae.scale_factor
            if self.vae is not None else
            self.predictor.config.sample_size    
        )
    
    @property
    def num_channels(self):
        return (
            self.vae.config.latent_channels
            if self.vae is not None else
            self.predictor.config.in_channels
        )
    
    def switch_to_refiner(self):
        pass
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # ================================================================================================================ #
    def __call__(
        self,
        mask_image: Optional[torch.FloatTensor] = None,
        masked_image: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> BackwardDiffusion:
    # ================================================================================================================ #
        """
        Вызов данной модели работает как фабрика, выплёвывающая нужную реализацию
        пайплайна BackwardDiffusion
        """
        print("DiffusionModel --->")

        bacward_step_pipeline = BackwardDiffusion(**self.key)

        # Устанавливает в пайплайн метку cfg
        bacward_step_pipeline.do_cfg = self.do_cfg
        # Правильно инитит маску и маскированную картинку в пайпе
        bacward_step_pipeline.mask_sample = (
            torch.cat([mask_image] * 2)
            if self.do_cfg and mask_image is not None else
            mask_image
        )
        bacward_step_pipeline.masked_sample = (
            torch.cat([masked_image] * 2)
            if self.do_cfg and masked_image is not None else
            masked_image
        )
        # Устанавливает там же guidance_scale
        bacward_step_pipeline.guidance_scale = self.guidance_scale

        # Если ещё и переданы условия, то разворачивает их в словарь в пайплайне
        if self.conditions is not None:
            bacward_step_pipeline.conditions = self.conditions
        
        # Возвращает собранный экземпляр пайплайна
        return bacward_step_pipeline
    # ================================================================================================================ #