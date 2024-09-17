import torch 

from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput

from .pipelines.vae_pipeline import VaePipeline
from .pipelines.forward_diffusion import (
    ForwardDiffusion,
    ForwardDiffusionInput,
)
from .pipelines.backward_diffusion import (
    BackwardDiffusion,
    BackwardDiffusionInput
)
from .diffusion_model import Conditions, DiffusionModel, DiffusionModelKey


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
    diffuser: DiffusionModel

    def __init__(
        self,
        model_key: Optional[DiffusionModelKey] = None,
        **kwargs,
    ):
        if model_key is not None:
            self.diffuser = DiffusionModel(**model_key)
            self.vae = self.diffuser.vae
            self.predictor = self.diffuser.predictor
            self.scheduler = self.diffuser.scheduler


    def diffusion_process(
        self,
        # diffuser: DiffusionModel,
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
            width = width or self.diffuser.sample_size
            height = height or self.diffuser.sample_size

        
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
                self.diffuser.num_channels,
                width,
                height,
            ),
            **forward_input
        )   



        conditions = self.diffuser(
            batch_size=batch_size,
            do_cfg=self.do_cfg,
            width=width,
            height=height,
            conditions=conditions,
        )


        #  Аналогично для Backward но в цикле
        backward_input = BackwardDiffusionInput(
            timestep=-1,
            noisy_sample=forward_output.noisy_sample, 
            # conditions=Conditions(**conditions),
            conditions=conditions,
        )
        for i, t in tqdm(enumerate(forward_output.timesteps)):
            # TODO: Добавить расширение условий за счёт ControlNet
            # <...>
            print(f"Step: {t}")
            backward_input.timestep = t
            backward_input = self.backward_step(
                # diffuser.predictor,
                **backward_input
            )
            print(f"Back step: {t}")
            
            # TODO: Добавить обработку маски через image
            # в случае если модель не для inpainting

        
        images, _ = self.pre_post_process(
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

        self.diffuser = diffuser

        return self.diffusion_process(
            # diffuser=diffuser,
            **input,
        )