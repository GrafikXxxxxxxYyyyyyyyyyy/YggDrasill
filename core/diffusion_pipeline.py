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
    do_cfg: bool = False
    guidance_scale: float = 5.0
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
    """
    Данный класс служит для того, чтобы выполнять полностью проход
    прямого и обратного диффузионного процессов и учитывать использование VAE
    """
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        pass
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    


    # ================================================================================================================ #
    def __call__(
        self,
        diffuser: DiffusionModel,
        batch_size: int = 1,
        do_cfg: bool = False,
        guidance_scale: float = 5.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        conditions: Optional[Conditions] = None,
        image: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        forward_input: Optional[ForwardDiffusionInput] = None,
        **kwargs,
    ):  
    # ================================================================================================================ #
        # Препроцессим выходные данные
        processor_pipeline = VaePipeline()
        processor_output = processor_pipeline(
            width = width,
            height = height,
            image = image,
            generator = generator,
            mask_image = mask_image,
            vae = diffuser if diffuser.is_latent_model else None,
        )

        initial_image = processor_output.image_latents


        # Учитываем возможные пользовательские размеры изображений
        # И пришедший на вход параметр do_cfg
        if initial_image is not None:
            width, height = initial_image.shape[2:]
        else:
            width = width or diffuser.sample_size
            height = height or diffuser.sample_size

        
        # Учитываем CFG для масок и картинок
        if processor_output.mask_latents is not None:
            mask_latents = processor_output.mask_latents.repeat(
                batch_size // mask_latents.shape[0], 1, 1, 1
            )
            processor_output.mask_latents = (
                torch.cat([mask_latents] * 2)
                if do_cfg else
                mask_latents
            )
        if processor_output.masked_image_latents is not None:
            masked_image_latents = processor_output.masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )
            processor_output.masked_image_latents = (
                torch.cat([masked_image_latents] * 2)
                if do_cfg else
                masked_image_latents
            )


        if "Выполняется ForwardDiffusion":
            forward_input.generator = generator
            forward_input.sample = initial_image
            forward_input.dtype = diffuser.dtype
            forward_input.device = diffuser.device
            # forward_input.denoising_end = 
            # forward_input.denoising_start = 
            forward_pipeline = ForwardDiffusion()
            forward_output = forward_pipeline(
                shape=(
                    batch_size,
                    diffuser.num_channels,
                    width,
                    height,
                ),
                noise_scheduler=diffuser,
                **forward_input
            )   


        # Инитим пайп обратного шага
        backward_pipeline = BackwardDiffusion(
            do_cfg=do_cfg,
            guidance_scale=guidance_scale,
            mask_sample=processor_output.mask_latents,
            masked_sample=processor_output.masked_image_latents,
        )

        if "Возможно если будет использован контролнет, надо будет убрать это в цикл":
            extended_conditions = diffuser.get_extended_conditions(
                batch_size=batch_size,
                do_cfg=do_cfg,
                width=width,
                height=height,
                conditions=conditions,
            )

            #  Аналогично для Backward но в цикле
            backward_input = BackwardDiffusionInput(
                timestep=-1,
                noisy_sample=forward_output.noisy_sample, 
                conditions=extended_conditions,
            )

        for i, t in tqdm(enumerate(forward_output.timesteps)):
            backward_input.timestep = t
            backward_input = backward_pipeline(
                diffuser=diffuser,
                **backward_input
            )
            
            # TODO: Добавить обработку маски через image
            # в случае если модель не для inpainting

        
        images, _ = processor_pipeline(
            latents=backward_input.noisy_sample,
        )


        return DiffusionPipelineOutput(
            images=images,
        )
    # ================================================================================================================ #