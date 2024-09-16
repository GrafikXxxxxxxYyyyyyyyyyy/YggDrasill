import torch 

from tqdm import tqdm
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
from .pipelines.backward_diffusion import (
    Conditions,
    BackwardDiffusionInput
)


@dataclass
class DiffusionPipelineInput(BaseOutput):
    width: Optional[int] = None
    height: Optional[int] = None
    conditions: Optional[Conditions] = None
    image: Optional[PipelineImageInput] = None
    generator: Optional[torch.Generator] = None
    mask_image: Optional[PipelineImageInput] = None
    forward_input: Optional[ForwardDiffusionInput] = None
    


@dataclass
class DiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



class DiffusionPipeline:
    # Попробуем сделать модель собственным членом класса
    diffuser: Optional[DiffusionModel] = None

    """
    Данный класс служит для того, чтобы выполнять полностью проход
    прямого и обратного диффузионного процессов и учитывать использование VAE
    """
    def __call__(
        self,
        diffuser: Optional[DiffusionModel] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        conditions: Optional[Conditions] = None,
        image: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        forward_input: Optional[ForwardDiffusionInput] = None,
        **kwargs,
    ):  
        print("DiffusionPipeline --->")
        if (
            diffuser is not None
            and isinstance(diffuser, DiffusionModel)
        ):
            self.diffuser = diffuser

        
        # Препроцессим входные изображения
        IMAGE_PROCESSOR = VaePipeline()
        processor_output = IMAGE_PROCESSOR(
            vae=self.diffuser.vae,
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
        BACKWARD, conditions = self.diffuser(
            width=width,
            height=height,
            conditions=conditions,
            mask_image=processor_output.mask_latents,
            masked_image=processor_output.masked_image_latents,
        )
        


        # Инитим форвард пайплайн из ключа модели
        FORWARD = ForwardDiffusion(**(self.diffuser.key))
        forward_input.sample = image
        forward_input.generator = generator
        forward_output = FORWARD(
            shape=(
                self.diffuser.batch_size,
                self.diffuser.num_channels,
                width,
                height,
            ),
            **forward_input
        )   


        backward_output = BackwardDiffusionInput(
            timestep=-1,
            noisy_sample=forward_output.noisy_sample, 
        )
        for i, t in tqdm(enumerate(forward_output.timesteps)):
            # TODO: Добавить расширение условий за счёт ControlNet
            # <...>

            backward_output.timestep = t
            backward_output = BACKWARD(
                self.diffuser.predictor,
                conditions=conditions,
                **backward_output
            )
            
            # TODO: Добавить обработку маски через image
            # в случае если модель не для inpainting


        vae_output = IMAGE_PROCESSOR(
            vae=self.diffuser.vae,
            latents=backward_output.noisy_sample
        )


        return DiffusionPipelineOutput(
            images=vae_output.images,
        )