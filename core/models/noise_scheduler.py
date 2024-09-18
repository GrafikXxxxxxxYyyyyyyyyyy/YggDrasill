import torch

from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from typing import Optional, Union



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

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):     
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
    def num_train_timesteps(self):
        return self.scheduler.config.num_train_timesteps
    

