import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Dict, Any

from ..diffusion_model import Conditions, DiffusionModel



@dataclass
class BackwardDiffusionInput(BaseOutput):
    timestep: int
    noisy_sample: torch.FloatTensor
    conditions: Optional[Conditions] = None



class BackwardDiffusion:
    do_cfg: bool = False
    guidance_scale: float = 5.0
    mask_sample: Optional[torch.FloatTensor] = None
    masked_sample: Optional[torch.FloatTensor] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        do_cfg: bool = False,
        guidance_scale: float = 5.0,
        mask_sample: Optional[torch.FloatTensor] = None,
        masked_sample: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        self.do_cfg = do_cfg
        self.guidance_scale = guidance_scale
        self.mask_sample = mask_sample
        self.masked_sample = masked_sample
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

    
    
    # ================================================================================================================ #
    def __call__(
        self,
        timestep: int, 
        noisy_sample: torch.FloatTensor,
        diffuser: DiffusionModel,
        conditions: Optional[Conditions] = None,
        **kwargs,
    ) -> BackwardDiffusionInput:
    # ================================================================================================================ #
        """
        """
        # Учитываем CFG
        model_input = (
            torch.cat([noisy_sample] * 2)
            if self.do_cfg else
            noisy_sample
        )   

        # Скейлит входы модели
        model_input = diffuser.scheduler.scale_model_input(
            timestep=timestep,
            sample=model_input,
        )

        # Конкатит маску и маскированную картинку для inpaint модели
        if (
            diffuser.is_inpainting_model
            and self.mask_sample is not None
            and self.masked_sample is not None
        ):
            model_input = torch.cat([model_input, self.mask_sample, self.masked_sample], dim=1)   
        
        # Получаем предсказание шума
        noise_predict = diffuser.get_noise_predict(
            timestep=timestep,
            noisy_sample=model_input,
            conditions=conditions,
        )

        # Учитываем CFG
        if self.do_cfg:
            negative_noise_pred, noise_pred = noise_predict.chunk(2)
            noise_predict = self.guidance_scale * (noise_pred - negative_noise_pred) + negative_noise_pred

        # Делает шаг расшумления изображения 
        less_noisy_sample = diffuser.scheduler.step(
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

