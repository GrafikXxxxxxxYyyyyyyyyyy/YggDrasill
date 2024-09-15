import torch 

from typing import List, Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from ..models.noise_scheduler import NoiseScheduler


@dataclass
class ForwardDiffusionInput(BaseOutput):
    strength: float = 1.0
    num_inference_steps: int = 30
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
    def __call__(
        self,
        shape,
        strength: float = 1.0, 
        num_inference_steps: int = 30, 
        timesteps: Optional[List[int]] = None,
        denoising_end: Optional[float] = None,
        denoising_start: Optional[float] = None,
        sample: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        noisy_sample: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
        print("ForwardDiffusion --->")


        # Опционально формируем временные шаги, если те не переданы
        if timesteps is None:
            # get the original timestep using init_timestep
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            # Устанавливаются шаги и возвращаются с учтенной силой
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
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
                        self.num_train_timesteps - (denoising_start * self.num_train_timesteps)
                    )
                )
                num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
                if self.order == 2 and num_inference_steps % 2 == 0:
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
                        self.num_train_timesteps - (denoising_end * self.num_train_timesteps)
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
                    device=self.device, 
                    dtype=self.dtype,
                )

                # Добавляем шум к входным данным
                noisy_sample = (
                    self.scheduler.add_noise(sample, noise, initial_timestep)
                    if sample is not None and not is_strength_max else
                    # scale the initial noise by the standard deviation required by the scheduler
                    noise * self.scale_factor
                )

        
        return ForwardDiffusionOutput(
            timesteps=timesteps,
            noisy_sample=noisy_sample,
        )