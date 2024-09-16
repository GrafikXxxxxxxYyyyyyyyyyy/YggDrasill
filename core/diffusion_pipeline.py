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
    NoisePredictor,
    BackwardDiffusion,
    BackwardDiffusionInput
)


@dataclass
class DiffusionPipelineInput(BaseOutput):
    batch_size: int = 1
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



class DiffusionPipeline(
    VaePipeline, 
    ForwardDiffusion, 
    BackwardDiffusion
):
    """
    Данный класс служит для того, чтобы выполнять полностью проход
    прямого и обратного диффузионного процессов и учитывать использование VAE
    """
    def diffusion_process(
        self,
        diffuser: DiffusionModel,
        batch_size: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        conditions: Optional[Conditions] = None,
        image: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        forward_input: Optional[ForwardDiffusionInput] = None,
        **kwargs,
    ):  
        (
            _, 
            mask_latents, 
            image_latents, 
            masked_image_latents,
        ) = self.process_images(
            vae=diffuser.vae,
            width=width,
            height=height,
            image=image,
            generator=generator,
            mask_image=mask_image,
        )

        initial_image = image_latents


        # ###################################################################### #
        # TODO: вынести на сторону модели
        # ###################################################################### #
        if initial_image is not None:
            width, height = image.shape[2:]
        else:
            width = width or diffuser.sample_size
            height = height or diffuser.sample_size


        if mask_latents is not None:
            mask_latents = mask_latents.repeat(
                batch_size // mask_latents.shape[0], 1, 1, 1
            )
            mask_latents = (
                torch.cat([mask_latents] * 2)
                if self.do_cfg else
                mask_latents
            )
        if masked_image_latents is not None:
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )
            masked_image_latents = (
                torch.cat([masked_image_latents] * 2)
                if self.do_cfg else
                masked_image_latents
            )
        # ###################################################################### #
        self.mask_sample = mask_latents
        self.masked_sample = masked_image_latents


        # Инитим форвард пайплайн из ключа модели
        forward_input.sample = image
        forward_input.generator = generator
        timesteps, noisy_sample = self.forward_pass(
            shape=(
                batch_size,
                diffuser.num_channels,
                width,
                height,
            ),
            **forward_input
        )   


        backward_input = BackwardDiffusionInput(
            timestep=-1,
            noisy_sample=noisy_sample, 
        )
        for i, t in tqdm(enumerate(timesteps)):
            # TODO: Добавить расширение условий за счёт ControlNet
            # <...>

            backward_input.timestep = t
            _, less_noisy_sample = self.backward_step(
                predictor=diffuser.predictor,
                conditions=conditions,
                **backward_input
            )
            backward_input.noisy_sample = less_noisy_sample
            
            # TODO: Добавить обработку маски через image
            # в случае если модель не для inpainting


        images, _ = self.process_images(
            vae=diffuser.vae,
            latents=backward_input.noisy_sample,
        )

        return images
    


    def __call__(
        self,
        diffuser: DiffusionModel,
        input: DiffusionPipelineInput,
        **kwargs,
    ):  
        print("DiffusionPipeline --->")

        images = self.diffusion_process(
            diffuser=diffuser,
            **input,
        )


        return DiffusionPipelineOutput(
            images=images,
        )