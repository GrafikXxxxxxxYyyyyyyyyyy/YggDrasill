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
        # Препроцессим выходные данные
        (
            _, 
            mask_latents, 
            image_latents, 
            masked_image_latents,
        ) = self.pre_post_process(
            vae = diffuser.vae,
            width = width,
            height = height,
            image = image,
            generator = generator,
            mask_image = mask_image,
        )

        initial_image = image_latents


        # Учитываем возможные пользовательские размеры изображений
        if initial_image is not None:
            width, height = initial_image.shape[2:]
        else:
            width = width or diffuser.sample_size
            height = height or diffuser.sample_size

        
        # Учитываем CFG для масок и картинок
        if mask_latents is not None:
            mask_latents = mask_latents.repeat(
                batch_size // mask_latents.shape[0], 1, 1, 1
            )
            self.mask_sample = (
                torch.cat([mask_latents] * 2)
                if self.do_cfg else
                mask_latents
            )
        if masked_image_latents is not None:
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )
            self.masked_sample = (
                torch.cat([masked_image_latents] * 2)
                if self.do_cfg else
                masked_image_latents
            )

        
        # Дополняем входы Forward пайпа и запускаем 
        forward_input.sample = initial_image
        forward_input.generator = generator
        # forward_input.denoising_end = 
        # forward_input.denoising_start = 
        forward_output = self.forward_pass(
            shape=(
                batch_size,
                diffuser.num_channels,
                width,
                height,
            ),
            **forward_input
        )   


        print(f"ConditionsDP: {conditions}")
        conditions = diffuser(
            do_cfg=self.do_cfg,
            width=width,
            height=height,
            conditions=conditions,
        )
        print(f"ConditionsDP after model: {conditions}")


        #  Аналогично для Backward но в цикле
        backward_input = BackwardDiffusionInput(
            timestep=-1,
            noisy_sample=forward_output.noisy_sample, 
        )
        backward_input.conditions = conditions
        for i, t in tqdm(enumerate(forward_output.timesteps)):
            # TODO: Добавить расширение условий за счёт ControlNet
            # <...>

            backward_input.timestep = t
            backward_input = self.backward_step(
                diffuser.predictor,
                **backward_input
            )
            
            # TODO: Добавить обработку маски через image
            # в случае если модель не для inpainting

        
        images, _ = self.pre_post_process(
            vae=diffuser.vae,
            latents=backward_input.noisy_sample,
        )


        return DiffusionPipelineOutput(
            images=images,
        )
    


    def __call__(
        self,
        diffuser: DiffusionModel,
        input: DiffusionPipelineInput,
        **kwargs,
    ):  
        print("DiffusionPipeline --->")

        return self.diffusion_process(
            diffuser=diffuser,
            **input,
        )