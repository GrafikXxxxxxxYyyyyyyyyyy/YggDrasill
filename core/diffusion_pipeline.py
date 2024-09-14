import torch 

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput

from .pipelines.vae_pipeline import VaePipeline
from .diffusion_model import DiffusionModel
from .pipelines.forward_diffusion import (
    ForwardDiffusion,
    ForwardDiffusionInput,
)
from .pipelines.backward_diffusion import BackwardDiffusionInput


@dataclass
class DiffusionPipelineInput(BaseOutput):
    forward_input: ForwardDiffusionInput
    width: Optional[int] = None
    height: Optional[int] = None
    image: Optional[PipelineImageInput] = None
    generator: Optional[torch.Generator] = None
    mask_image: Optional[PipelineImageInput] = None


@dataclass
class DiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



class DiffusionPipeline:
    """
    Данный класс служит для того, чтобы выполнять полностью проход
    прямого и обратного диффузионного процессов и учитывать использование VAE
    """
    def __call__(
        self,
        diffuser: DiffusionModel,
        width: Optional[int] = None,
        height: Optional[int] = None,
        image: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        forward_input: Optional[ForwardDiffusionInput] = None,
        **kwargs,
    ):  
        print("DiffusionPipeline --->")


        # Инитим форвард пайплайн из ключа модели
        FORWARD = ForwardDiffusion(**diffuser.key)
        
        # Препроцессим входные изображения
        IMAGE_PROCESSOR = VaePipeline()
        processor_output = IMAGE_PROCESSOR(
            vae=diffuser.vae,
            width=width,
            height=height,
            image=image,
            generator=generator,
            mask_image=mask_image,
        )
        image = processor_output.image_latents
        if image is not None:
            width, height = image.shape[2:]
        else:
            width = width or diffuser.sample_size
            height = height or diffuser.sample_size
        
        # Получаем пайп для шага обратного процесса из самой модели
        BACKWARD = diffuser(
            mask_image=processor_output.mask_latents,
            masked_image=processor_output.masked_image_latents,
        )
        
        if forward_input is None:
            forward_input = ForwardDiffusionInput(

            )
        forward_input.sample = image
        forward_input.generator = generator
        forward_input.num_channels = diffuser.num_channels
        forward_output = FORWARD(
            width=width,
            height=height,
            **forward_input
        )   


        backward_input = BackwardDiffusionInput(
            timestep=-1,
            noisy_sample=forward_output.noisy_sample, 
        )
        for i, t in enumerate(forward_output.timesteps):
            backward_input.timestep = t
            backward_input = BACKWARD(
                diffuser.predictor,
                **backward_input
            )
            
            # TODO: Добавить обработку маски через image
            # в случае если модель не для inpainting


        vae_output = IMAGE_PROCESSOR(
            vae=diffuser.vae,
            latents=backward_input.noisy_sample
        )


        return DiffusionPipelineOutput(
            images=vae_output.images,
        )