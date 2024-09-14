import torch

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput

from ..models.vae_model import VaeModel


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
    def __call__(
        self, 
        width: Optional[int] = None,
        height: Optional[int] = None,
        vae: Optional[VaeModel] = None,
        image: Optional[PipelineImageInput] = None,
        generator: Optional[torch.Generator] = None,
        mask_image: Optional[PipelineImageInput] = None,
        latents: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> VaePipelineOutput:  
        """
        1) Препроцессит изображения
        2) Кодирует картинку (и маски) в латентное представление
        3) Декодирует пришедшие на вход латентные представления
        """
        use_vae = True if vae is not None else False

        # Инициализируем необходимые классы
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=(
                vae.scale_factor
                if use_vae else
                1
            )
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=(
                vae.scale_factor
                if use_vae else
                1
            ), 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True,
        )
        output = VaePipelineOutput()


        # Предобрабатываем пришедшие на вход изображения (и их маски)
        if image is not None:
            image = self.image_processor.preprocess(image)    
            if height is not None and width is not None:
                image = torch.nn.functional.interpolate(
                    image, 
                    size=(height, width)
                )
                
                # Возвращает либо латентное представление картинки
                # либо запроцешенную картинку
                output.image_latents = (
                    vae(
                        images=image,
                        generator=generator,
                    )[0]
                    if use_vae else
                    image
                )

            if mask_image is not None:
                mask_image = self.mask_processor.preprocess(mask_image)        
                if height is not None and width is not None:
                    mask_image = torch.nn.functional.interpolate(
                        mask_image, 
                        size=(height, width)
                    )
                masked_image = image * (mask_image < 0.5)
                output.mask_latents = mask_image

                output.masked_image_latents = (
                    vae(
                        images=masked_image,
                        generator=generator,
                    )[0]
                    if use_vae else
                    masked_image
                )

        if latents is not None:
            images = (
                vae(latents=latents)[1]
                if use_vae else
                latents
            )
            output.images = self.image_processor.postprocess(images.detach())


        return output


