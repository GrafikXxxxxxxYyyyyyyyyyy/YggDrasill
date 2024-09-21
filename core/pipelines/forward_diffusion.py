import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Tuple
from diffusers.utils.torch_utils import randn_tensor

from ..models.backward_diffuser import ModelKey
from ..models.models.noise_scheduler import NoiseScheduler






@dataclass
class ForwardDiffusionInput(BaseOutput):
    device: str = "cuda"
    strength: float = 1.0
    num_inference_steps: int = 30
    dtype: torch.dtype = torch.float16
    timesteps: Optional[List[int]] = None
    denoising_end: Optional[float] = None
    denoising_start: Optional[float] = None
    sample: Optional[torch.FloatTensor] = None
    generator: Optional[torch.Generator] = None
    noisy_sample: Optional[torch.FloatTensor] = None






@dataclass
class ForwardDiffusionOutput(BaseOutput):
    timesteps: List[int]
    noisy_sample: torch.Tensor





# НИ ОТ ЧЕГО НЕ НАСЛЕДУЕТСЯ + ХРАНИТ СВОЙ Scheduler
class ForwardDiffusion:
    # Теперь пайплайны хранят свой собственный планировщик шума
    pipe_scheduler: Optional[NoiseScheduler] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_key: Optional[ModelKey] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            self.pipe_scheduler = NoiseScheduler(**model_key)
    # ////////////////// ////////////////////////////////////////////////////////////////////////////////////////////// #



    def forward_pass(
        self,
        shape: Tuple[int, int, int, int],
        device: str = "cuda",
        strength: float = 1.0, 
        num_inference_steps: int = 30, 
        dtype: torch.dtype = torch.float16,
        timesteps: Optional[List[int]] = None,
        denoising_end: Optional[float] = None,
        denoising_start: Optional[float] = None,
        sample: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        noisy_sample: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
        if "1. Формируем временные шаги, если те не переданы":
            if timesteps is None:
                timesteps, num_inference_steps = self.pipe_scheduler.retrieve_timesteps(
                    device=device,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                )

        

        if "2. Применяем логику рефайнера, если переданы параметры start/end":
            if (
                denoising_start is not None
                and isinstance(denoising_start, float)
                and denoising_start > 0.0
                and denoising_start < 1.0
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.pipe_scheduler.num_train_timesteps 
                        - (denoising_start * self.pipe_scheduler.num_train_timesteps)
                    )
                )

                num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
                if self.pipe_scheduler.order == 2 and num_inference_steps % 2 == 0:
                    num_inference_steps = num_inference_steps + 1

                # because t_n+1 >= t_n, we slice the timesteps starting from the end
                timesteps = timesteps[-num_inference_steps:]


            if (
                denoising_end is not None
                and isinstance(denoising_end, float)
                and denoising_end > 0.0
                and denoising_end < 1.0
                and denoising_start is not None
                and isinstance(denoising_start, float)
                and denoising_start > 0.0
                and denoising_start < 1.0
                and denoising_start >= denoising_end
            ):
                raise ValueError("'denoising_start' cannot be larger than or equal to 'denoising_end'")
            
            elif (
                denoising_end is not None
                and isinstance(denoising_end, float)
                and denoising_end > 0.0
                and denoising_end < 1.0
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.pipe_scheduler.num_train_timesteps 
                        - (denoising_end * self.pipe_scheduler.num_train_timesteps)
                    )
                )

                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]


        
        if "3. Формируем зашумлённый вход, если тот не передан":
            if noisy_sample is None:
                # Сэмплируем чистый шум
                clear_noise = randn_tensor(
                    shape=shape,
                    generator=generator, 
                    device=device, 
                    dtype=dtype,
                )

                # Накладываем шум на входные изображения
                noisy_sample = self.pipe_scheduler.add_noise(
                    clear_noise,
                    sample=sample,
                    is_strength_max=strength == 1.0,
                    initial_timesteps=timesteps[:1].repeat(shape[0])
                )


        return ForwardDiffusionOutput(
            timesteps=timesteps,
            noisy_sample=noisy_sample,
        )



    # ================================================================================================================ #
    def __call__(
        self,
        shape: Tuple[int, int, int, int],
        input: Optional[ForwardDiffusionInput] = None,
        noise_scheduler: Optional[NoiseScheduler] = None,
        **kwargs,
    ):  
    # ================================================================================================================ #
        pass
    # ================================================================================================================ #