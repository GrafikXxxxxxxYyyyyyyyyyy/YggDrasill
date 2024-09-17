import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Dict, Any

from ..models.noise_scheduler import NoiseScheduler
from ..models.noise_predictor import Conditions, NoisePredictor


@dataclass
class BackwardDiffusionInput(BaseOutput):
    timestep: int
    noisy_sample: torch.FloatTensor
    conditions: Optional[Conditions] = None



class BackwardDiffusion(NoiseScheduler):
    # Собственные аргументы инитятся там, где они последний раз используются 
    do_cfg: bool = False
    guidance_scale: float = 5.0
    mask_sample: Optional[torch.FloatTensor] = None
    masked_sample: Optional[torch.FloatTensor] = None


    def backward_step(
        self,
        predictor: NoisePredictor,
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
            predictor.is_inpainting_model
            and self.mask_sample is not None
            and self.masked_sample is not None
        ):
            model_input = torch.cat([model_input, self.mask_sample, self.masked_sample], dim=1)   
        
        print(f"Step: {timestep}")
        # Получаем предсказание шума
        noise_predict = predictor(
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
        predictor: NoisePredictor,
        input: BackwardDiffusionInput,
        **kwargs,
    ) -> BackwardDiffusionInput:
        """
        Данный пайплайн выполняет один полный шаг снятия шума в диффузионном процессе
        """
        print("BackwardDiffusion --->")

        return self.backward_step(
            predictor, 
            **input,
        )

