import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Dict, Any

from ..models.noise_scheduler import NoiseScheduler
from ..diffusion_model import Conditions, NoisePredictor, ModelKey


@dataclass
class BackwardDiffusionInput(BaseOutput):
    timestep: int
    noisy_sample: torch.FloatTensor
    conditions: Optional[Conditions] = None



class BackwardDiffusion(NoiseScheduler):
    model: Optional[NoisePredictor] = None

    do_cfg: bool = False
    guidance_scale: float = 5.0
    mask_sample: Optional[torch.FloatTensor] = None
    masked_sample: Optional[torch.FloatTensor] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        do_cfg: bool = False,
        guidance_scale: float = 5.0,
        model_key: Optional[ModelKey] = None,
        scheduler_name: Optional[str] = None,
        mask_sample: Optional[torch.FloatTensor] = None,
        masked_sample: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            super().__init__(
                scheduler_name=scheduler_name, 
                **model_key
            )
            self.model = NoisePredictor(**model_key)

        self.do_cfg = do_cfg
        self.guidance_scale = guidance_scale
        self.mask_sample = mask_sample
        self.masked_sample = masked_sample
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



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
            self.model.is_inpainting_model
            and self.mask_sample is not None
            and self.masked_sample is not None
        ):
            model_input = torch.cat([model_input, self.mask_sample, self.masked_sample], dim=1)   
        
        # Получаем предсказание шума
        noise_predict = self.model.get_noise_predict(
            timestep=timestep,
            noisy_sample=model_input,
            conditions=conditions,
        )

        # Учитываем CFG
        if self.do_cfg:
            negative_noise_pred, noise_pred = noise_predict.chunk(2)
            noise_predict = self.guidance_scale * (noise_pred - negative_noise_pred) + negative_noise_pred

        # Делает шаг расшумления изображения 
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

    
    
    # ================================================================================================================ #
    def __call__(
        self,
        input: BackwardDiffusionInput,
        predictor: Optional[NoisePredictor] = None,
        **kwargs,
    ) -> BackwardDiffusionInput:
    # ================================================================================================================ #
        """
        Данный пайплайн выполняет один полный шаг снятия шума в диффузионном процессе
        """
        print("BackwardDiffusion --->")

        # ВАЖНО! Backward не следит за тем, какой планировщик используется
        # подразумевается что если происходит запуск из DiffusionPipeline и выше
        # то автоматически настраивается верный self.scheduler
        # А если запуск происходит в виде отдельной блочной структуры, то 
        # как и для ForwardDiffusion планировщик инициализируется свой для 
        # каждого из пайплайнов
        if (    
            predictor is not None
            and isinstance(predictor, NoisePredictor)    
        ):
            self.model = predictor

        return self.backward_step(**input)
    # ================================================================================================================ #

