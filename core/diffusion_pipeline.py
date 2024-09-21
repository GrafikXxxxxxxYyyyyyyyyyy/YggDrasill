import torch 

from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput

from .diffusion_model import Conditions, DiffusionModel
from .pipelines.vae_pipeline import VaePipeline
from .pipelines.forward_diffusion import ForwardDiffusion, ForwardDiffusionInput






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






class DiffusionPipeline(
    VaePipeline,
    ForwardDiffusion,
):
    """
    Данный класс служит для того, чтобы выполнять полностью проход
    прямого и обратного диффузионного процессов и учитывать использование VAE
    """
    model: Optional[DiffusionModel] = None

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



    def diffusion_process(
        self,
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
        """
        Выполняет полностью ПРЯМОЙ + ОБРАТНЫЙ диффузионные процессы из заданных условий
        """

        if "1. Возможно предобрабатываем входные данные":
            # Должен быть self.vae: VaeModel
            processor_output = self.vae_pipeline_call(
                width = width,
                height = height,
                image = image,
                generator = generator,
                mask_image = mask_image,
            )

            initial_image = processor_output.image_latents

            # Учитываем возможные пользовательские размеры изображений
            if initial_image is not None:
                width, height = initial_image.shape[2:]
            else:
                width = width or self.model.sample_size
                height = height or self.model.sample_size

            # Учитываем batch_size если он был изменен
            if processor_output.mask_latents is not None:
                processor_output.mask_latents = processor_output.mask_latents.repeat(
                    batch_size // processor_output.mask_latents.shape[0], 1, 1, 1
                )
            if processor_output.masked_image_latents is not None:
                processor_output.masked_image_latents = processor_output.masked_image_latents.repeat(
                    batch_size // processor_output.masked_image_latents.shape[0], 1, 1, 1
                )

            print(processor_output)



        if "Выполняется ForwardDiffusion":
            forward_input.generator = generator
            forward_input.sample = initial_image
            forward_input.dtype = self.model.dtype
            forward_input.device = self.model.device
            # forward_input.denoising_end = 
            # forward_input.denoising_start = 

            
            forward_output = self.forward_pass(
                shape=(
                    batch_size,
                    self.model.num_channels,
                    width,
                    height,
                ),
                **forward_input,
            )

            print(forward_output)



        # Учитываем CFG для масок и картинок
        if do_cfg:
            processor_output.mask_latents = torch.cat([processor_output.mask_latents] * 2)
            processor_output.masked_image_latents = torch.cat([processor_output.masked_image_latents] * 2)


        noisy_sample = forward_output.noisy_sample
        for i, t in tqdm(enumerate(forward_output.timesteps)):
            # Учитываем что может быть inpaint модель
            if self.model.predictor.is_inpainting_model:
                noisy_sample = torch.cat([
                    noisy_sample, processor_output.mask_latents, processor_output.masked_image_latents
                ], dim=1)   
                
            noisy_sample = self.model.backward_step(
                timestep=t,
                noisy_sample=noisy_sample,
                do_cfg=do_cfg,
                guidance_scale=guidance_scale,
                conditions=conditions,
            )
            
            # TODO: Добавить обработку маски через image
            # в случае если модель не для inpainting

        
        # images, _ = processor_pipeline(
        #     latents=backward_input.noisy_sample,
        # )


        # return DiffusionPipelineOutput(
        #     images=images,
        # )
    


    # ================================================================================================================ #
    def __call__(
        self,
        # diffuser: DiffusionModel,
        # batch_size: int = 1,
        # do_cfg: bool = False,
        # guidance_scale: float = 5.0,
        # width: Optional[int] = None,
        # height: Optional[int] = None,
        # conditions: Optional[Conditions] = None,
        # image: Optional[torch.FloatTensor] = None,
        # generator: Optional[torch.Generator] = None,
        # mask_image: Optional[torch.FloatTensor] = None,
        # forward_input: Optional[ForwardDiffusionInput] = None,
        **kwargs,
    ):  
    # ================================================================================================================ #
        pass
    # ================================================================================================================ #