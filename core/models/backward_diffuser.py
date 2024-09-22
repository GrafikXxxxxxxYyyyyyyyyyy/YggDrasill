import torch

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Optional, Union, Dict, Any

from .models.noise_scheduler import NoiseScheduler
from .models.noise_predictor import ModelKey, Conditions, NoisePredictor





@dataclass
class BackwardDiffuserKey(ModelKey):
    scheduler_name: str = "euler"





@dataclass
class BackwardDiffuserConditions(BaseOutput):
    do_cfg: bool = False
    guidance_scale: float = 5.0
    conditions: Optional[Conditions] = None
    mask_sample: Optional[torch.FloatTensor] = None
    masked_sample: Optional[torch.FloatTensor] = None





# МОДЕЛЬ НАСЛЕДУЕТСЯ ОТ ПЛАНИРОВЩИКА
class BackwardDiffuser(NoiseScheduler):
    predictor: NoisePredictor
    
    use_refiner: bool = False

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self, 
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #            
        # Инициализируем модель планировщика
        super().__init__(
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
            scheduler_name=scheduler_name
        )

        # Инициализируется внутренняя модель предиктора
        self.predictor = NoisePredictor(
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
        )

        # Инициализируем константы
        self.dtype = dtype
        self.model_path = model_path
        self.model_type = model_type or "sd15"
        self.device = torch.device(device)

        print("\t<<<BackwardDiffuser ready!>>>\t")

        return


    def maybe_switch_to_refiner(self, use_refiner: bool):
        if use_refiner:
            self.predictor = NoisePredictor(
                model_path="REFINER_PATH",
                **self.key,
            )
            self.use_refiner = True


    def maybe_switch_to_base(self, use_refiner: bool):
        if not use_refiner:
            self.predictor = NoisePredictor(**self.key)
            self.use_refiner = False
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    def backward_step(
        self,
        timestep: int, 
        noisy_sample: torch.FloatTensor,
        do_cfg: bool = False,
        guidance_scale: float = 5.0,
        conditions: Optional[Conditions] = None,
        mask_sample: Optional[torch.FloatTensor] = None,
        masked_sample: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Делает один шаг снятия шума
        """
        # Используем функционал self.scheduler
        noisy_sample = self.scheduler.scale_model_input(
            sample=noisy_sample,
            timestep=timestep,
        )

        # Учитываем CFG
        if do_cfg:
            noisy_sample = torch.cat([noisy_sample] * 2)

        # Вызываем NoisePredictor
        noise_predict = self.predictor(
            timestep=timestep,
            noisy_sample=noisy_sample,
            **conditions,
        )

        # Учитываем CFG
        if do_cfg:
            negative_noise_pred, noise_pred = noise_predict.chunk(2)
            noise_predict = guidance_scale * (noise_pred - negative_noise_pred) + negative_noise_pred

        # Используем функционал self.scheduler
        less_noisy_sample = self.scheduler.step(
            timestep=timestep,
            sample=noisy_sample,
            model_output=noise_predict,
        )

        return less_noisy_sample



    # ================================================================================================================ #
    def __call__(
        self,
        **kwargs,
    ) -> torch.FloatTensor:
    # ================================================================================================================ #
        return self.backward_step(**kwargs)
    # ================================================================================================================ #    



