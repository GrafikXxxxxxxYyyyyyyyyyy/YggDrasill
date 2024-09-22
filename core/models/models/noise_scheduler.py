# 1) Планировщик - единственная модель, которая не имеет своего собственного метода .call
# 2) Связующая модель, от которой наследуются и модели диффузии и пайплайны
# 3) Сам он в свою очередь наследуется от класса-ключа для всех моделей


import torch

from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from typing import Optional, Union, List, Tuple



class NoiseScheduler:
    scheduler: Union[
        DDIMScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
        PNDMScheduler,
        UniPCMultistepScheduler,
    ]
    scheduler_name: str = "euler"

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
        scheduler_name = scheduler_name or "euler"

        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder='scheduler'
        )
        if scheduler_name == "DDIM":
            self.scheduler = DDIMScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "euler":
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "euler_a":
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "DPM++ 2M":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "DPM++ 2M Karras":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_path,
                subfolder='scheduler',
                use_karras_sigmas=True,
            )
        elif scheduler_name == "DPM++ 2M SDE Karras":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_path,
                subfolder='scheduler',
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
        elif scheduler_name == "PNDM":
            self.scheduler = PNDMScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        elif scheduler_name == "uni_pc":
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                model_path,
                subfolder='scheduler'
            )
        else:
            raise ValueError(f'Unknown scheduler name: {scheduler_name}')
        
        self.scheduler_name = scheduler_name
        print(f"Scheduler has successfully changed to '{scheduler_name}'")
    
    @property
    def order(self):
        return self.scheduler.order
    
    @property
    def num_train_timesteps(self):
        return self.scheduler.config.num_train_timesteps
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    def retrieve_timesteps(
        self, 
        num_inference_steps: int, 
        strength: float = 1.0, 
        device: Union[str, torch.device] = None,
    ) -> Tuple[List[int], int]:
        """
        Возвращает расписание временных шагов с учётом их количества и силы зашумления
        """
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        # Устанавливаются шаги и возвращаются с учтенной силой
        self.scheduler.set_timesteps(num_inference_steps, device=device) 
        timesteps = self.scheduler.timesteps[t_start * self.order :]

        return timesteps, len(timesteps)



    def add_noise(
        self,
        noise: torch.FloatTensor,
        is_strength_max: bool = True,
        sample: Optional[torch.FloatTensor] = None,
        initial_timesteps: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Накладывает шум на оригинальные изображения
        """

        noisy_sample = (
            self.scheduler.add_noise(sample, noise, initial_timesteps)
            if (
                sample is not None 
                and not is_strength_max
            ) else
            # scale the initial noise by the standard deviation required by the scheduler
            noise * self.scheduler.init_noise_sigma
        )

        return noisy_sample
        







