import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Dict, Any

from ..models.noise_scheduler import NoiseScheduler
from ..diffusion_model import DiffusionModelKey, Conditions, NoisePredictor


@dataclass
class BackwardDiffusionInput(BaseOutput):
    timestep: int
    noisy_sample: torch.FloatTensor
    conditions: Optional[Conditions] = None



class BackwardDiffusion(NoiseScheduler):
    predictor: Optional[NoisePredictor] = None

    def __init__(
        self,
        do_cfg: bool = False,
        guidance_scale: float = 5.0,
        model_key: Optional[DiffusionModelKey] = None,
        mask_sample: Optional[torch.FloatTensor] = None,
        masked_sample: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
        if model_key is not None:
            super().__init__(**model_key)
            self.predictor = NoisePredictor(**model_key)

        self.do_cfg = do_cfg
        self.guidance_scale = guidance_scale
        self.mask_sample = mask_sample
        self.masked_sample = masked_sample


    def backward_step(
        self,
        timestep: int, 
        noisy_sample: torch.FloatTensor,
        conditions: Optional[Conditions] = None,
        **kwargs,
    ) -> BackwardDiffusionInput:
        """
        """
        # Учитываем CFG
        model_input = (
            torch.cat([noisy_sample] * 2)
            if self.do_cfg else
            noisy_sample
        )   

        # Скейлит входы модели
        model_input = self.scheduler.scale_model_input(
            timestep=timestep,
            sample=model_input,
        )

        # Конкатит маску и маскированную картинку для inpaint модели
        if (
            self.predictor.is_inpainting_model
            and self.mask_sample is not None
            and self.masked_sample is not None
        ):
            model_input = torch.cat([model_input, self.mask_sample, self.masked_sample], dim=1)   
        
        print(f"Step: {timestep}")
        # Получаем предсказание шума
        noise_predict = self.predictor(
            timestep=timestep,
            noisy_sample=model_input,
            conditions=conditions,
        )
        print(f"Back step: {timestep}")

        # Учитываем CFG
        if self.do_cfg:
            negative_noise_pred, noise_pred = noise_predict.chunk(2)
            noise_predict = self.guidance_scale * (noise_pred - negative_noise_pred) + negative_noise_pred

        # Делаем шаг расшумления изображения 
        less_noisy_sample = self.scheduler.step(
            timestep=timestep,
            sample=noisy_sample,
            model_output=noise_predict,
        )

        return BackwardDiffusionInput(
            timestep=timestep,
            conditions=conditions,
            noisy_sample=less_noisy_sample,
        )

    
    
    def __call__(
        self,
        input: BackwardDiffusionInput,
        predictor: Optional[NoisePredictor] = None,
        **kwargs,
    ) -> BackwardDiffusionInput:
        """
        Данный пайплайн выполняет один полный шаг снятия шума в диффузионном процессе
        """
        print("BackwardDiffusion --->")

        if predictor is not None:
            self.predictor = predictor

        return self.backward_step(**input)

