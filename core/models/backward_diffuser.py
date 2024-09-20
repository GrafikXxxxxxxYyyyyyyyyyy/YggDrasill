import torch

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Optional, Union, Dict, Any

from .models.noise_predictor import NoisePredictor
from .models.noise_scheduler import NoiseScheduler






# Пускай теперь тут же инициализируется ключ для модели
@dataclass
class ModelKey(BaseOutput):
    """
    Базовый класс для инициализации всех 
    моделей которые используются в проекте
    """
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    model_type: str = "sdxl"
    scheduler_name: str = "euler"
    model_path: str = "GrafikXxxxxxxYyyyyyyyyyy/sdxl_Juggernaut"






# Аналогично пусть тут  инициализируется набор условий на вход самой модели
@dataclass
class Conditions(BaseOutput):
    """
    Общий класс всех дополнительных условий для всех
    моделей которые используются в проекте
    """
    # UNet2DModel
    class_labels: Optional[torch.Tensor] = None
    # UNet2DConditionModel
    prompt_embeds: Optional[torch.Tensor] = None
    timestep_cond: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None






# Пускай это будет первая модель в наследной иерархии диффузионных моделей 
# Поскольку это первая модель, она сама ни от кого не наследуется, но хранит 
# все необходимые компоненты как свои собственные части
class BackwardDiffuser(ModelKey):
    predictor: NoisePredictor
    scheduler: NoiseScheduler

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

        # Инициализируем внутреннюю модель планировщика
        self.scheduler = NoiseScheduler(
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
            scheduler_name=scheduler_name
        )

        # Аналогично инициализируется внутренняя модель предиктора
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







    # # ================================================================================================================ #
    # def __call__(
    #     self,
    # ):
    # # ================================================================================================================ #
    #     """
    #     Вызов данной модели по сути произовдит просто процедуру одного обратного
    #     шага диффузионного процесса
    #     """

    #     pass
    # # ================================================================================================================ #






    # ================================================================================================================ #
    def __call__(
        self,
        timestep: int, 
        noisy_sample: torch.FloatTensor,
            # diffuser: DiffusionModel,
        do_cfg: bool = False,
        guidance_scale: float = 5.0,
        conditions: Optional[Conditions] = None,
        **kwargs,
    ) -> torch.FloatTensor:
    # ================================================================================================================ #
        """
        Делает один шаг снятия шума
        """
        # Учитываем CFG
        model_input = (
            torch.cat([noisy_sample] * 2)
            if do_cfg else
            noisy_sample
        )   

        # Скейлит входы модели
        model_input = self.scheduler.scale_model_input(
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

        return less_noisy_sample



