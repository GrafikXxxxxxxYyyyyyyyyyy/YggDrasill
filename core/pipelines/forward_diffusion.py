import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Tuple
from diffusers.utils.torch_utils import randn_tensor

from ..models.noise_predictor import ModelKey
from ..models.noise_scheduler import NoiseScheduler


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



class ForwardDiffusion(NoiseScheduler):
    """
    Данный пайплайн выполняет процедуру прямого диффузионного процесса 
    """
    denoising_end: Optional[float] = None,
    denoising_start: Optional[float] = None,

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_key: Optional[ModelKey] = None,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            super().__init__(
                scheduler_name=scheduler_name, 
                **model_key
            )
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # TODO: Разбить это на 3 отдельные функции для понятности
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
        # Опционально формируем временные шаги, если те не переданы
        if timesteps is None:
            # get the original timestep using init_timestep
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            # Устанавливаются шаги и возвращаются с учтенной силой
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            num_inference_steps = len(timesteps)

        
        if "Применяем логику рефайнера, если переданы параметры start/end":
            if (
                denoising_start is not None
                and isinstance(denoising_start, float)
                and denoising_start > 0.0
                and denoising_start < 1.0
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.num_train_timesteps 
                        - (denoising_start * self.num_train_timesteps)
                    )
                )
                num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
                if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
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
                raise ValueError(
                    f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                    + f" {denoising_end} when using type float."
                )
            elif (
                denoising_end is not None
                and isinstance(denoising_end, float)
                and denoising_end > 0.0
                and denoising_end < 1.0
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.num_train_timesteps 
                        - (denoising_end * self.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]


            # Опционально, если не передан шумный семпл, формируем его
            if noisy_sample is None:
                is_strength_max = strength == 1.0
                initial_timestep = timesteps[:1].repeat(shape[0])

                # Сэмплируем чистый шум
                noise = randn_tensor(
                    shape=shape,
                    generator=generator, 
                    device=device, 
                    dtype=dtype,
                )

                # Добавляем шум к входным данным
                noisy_sample = (
                    self.scheduler.add_noise(sample, noise, initial_timestep)
                    if sample is not None and not is_strength_max else
                    # scale the initial noise by the standard deviation required by the scheduler
                    noise * self.scheduler.init_noise_sigma
                )

            return ForwardDiffusionOutput(
                timesteps=timesteps,
                noisy_sample=noisy_sample,
            )



    # ================================================================================================================ #
    def __call__(
        self,
        shape,
        input: Optional[ForwardDiffusionInput] = None,
        **kwargs,
    ):  
    # ================================================================================================================ #
        print("ForwardDiffusion --->")

        # Если на вход не пришло инпута, создаём дефолтный
        if input is None: 
            input = ForwardDiffusionInput()

        return self.forward_pass(
            shape=shape,
            **input,
        )
    # ================================================================================================================ #