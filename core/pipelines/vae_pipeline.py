import torch

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput

from ..models.vae_model import VaeModel
from ..diffusion_model import DiffusionModelKey


@dataclass
class VaePipelineInput(BaseOutput):
    width: Optional[int] = None
    height: Optional[int] = None
    image: Optional[PipelineImageInput] = None
    latents: Optional[torch.FloatTensor] = None
    generator: Optional[torch.Generator] = None
    mask_image: Optional[PipelineImageInput] = None


@dataclass
class VaePipelineOutput(BaseOutput):
    images: Optional[torch.FloatTensor] = None
    mask_latents: Optional[torch.FloatTensor] = None
    image_latents: Optional[torch.FloatTensor] = None
    masked_image_latents: Optional[torch.FloatTensor] = None



class VaePipeline:
    model: Optional[VaeModel] = None

    def __init__(
        self,
        model_key: Optional[DiffusionModelKey] = None,
        **kwargs,
    ):
        if model_key is not None:
            self.vae = VaeModel(**model_key)


    def pre_post_process(
        self, 
        width: Optional[int] = None,
        height: Optional[int] = None,
        image: Optional[PipelineImageInput] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[PipelineImageInput] = None,
        latents: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):  
        """
        1) Препроцессит изображения
        2) Кодирует картинку (и маски) в латентное представление
        3) Декодирует пришедшие на вход латентные представления
        """
        use_vae = (
            True 
            if hasattr(self.model, "vae") and self.model.vae is not None else
            False
        )

        # Инициализируем необходимые классы
        image_processor = VaeImageProcessor(
            vae_scale_factor=(
                self.model.vae_scale_factor
                if use_vae else
                1
            )
        )
        mask_processor = VaeImageProcessor(
            vae_scale_factor=(
                self.model.vae_scale_factor
                if use_vae else
                1
            ), 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True,
        )
        output = VaePipelineOutput()


        images: Optional[torch.FloatTensor] = None
        mask_latents: Optional[torch.FloatTensor] = None
        image_latents: Optional[torch.FloatTensor] = None
        masked_image_latents: Optional[torch.FloatTensor] = None
        # Предобрабатываем пришедшие на вход изображения (и их маски)
        if image is not None:
            image = image_processor.preprocess(image)    
            if height is not None and width is not None:
                image = torch.nn.functional.interpolate(
                    image, 
                    size=(height, width)
                )
                
                # Возвращает либо латентное представление картинки
                # либо запроцешенную картинку
                image_latents = (
                    self.model.get_processed_latents_or_images(
                        images=image,
                        generator=generator,
                    )[0]
                    if use_vae else
                    image
                )

            if mask_image is not None:
                mask_image = mask_processor.preprocess(mask_image)        
                if height is not None and width is not None:
                    mask_image = torch.nn.functional.interpolate(
                        mask_image, 
                        size=(height, width)
                    )
                masked_image = image * (mask_image < 0.5)
                mask_latents = (
                    torch.nn.functional.interpolate(
                        mask_image, 
                        size=(
                            height // self.vae.scale_factor, 
                            width // self.vae.scale_factor
                        )
                    )
                    if use_vae else
                    mask_image
                )

                masked_image_latents = (
                    self.model.get_processed_latents_or_images(
                        images=masked_image,
                        generator=generator,
                    )[0]
                    if use_vae else
                    masked_image
                )

        if latents is not None:
            images = (
                self.model.get_processed_latents_or_images(
                    latents=latents
                )[1]
                if use_vae else
                latents
            )
            output.images = image_processor.postprocess(images.detach())

        
        return VaePipelineOutput(
            images=images,
            mask_latents=mask_latents,
            image_latents=image_latents,
            masked_image_latents=masked_image_latents
        )
    


    def __call__(
        self, 
        input: VaePipelineInput,
        vae: Optional[VaeModel] = None,
        **kwargs,
    ) -> VaePipelineOutput:  
        print("VaePipeline --->")

        if vae is not None:
            self.vae = vae
    
        return self.pre_post_process(**input)



