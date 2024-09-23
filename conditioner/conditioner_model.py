import torch

from typing import Optional

from ..core.diffusion_model import DiffusionModelKey
from .models.text_encoder_model import TextEncoderModel



class ConditionerModel(TextEncoderModel):
        # image_encoder: Optional[ImageEncoderModel] = None

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        use_image_encoder: bool = False,
        **kwargs,
    ):  
        super().__init__(
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
        )

        # if use_image_encoder:
        #     self.image_encoder = ImageEncoderModel(**kwargs)

        print("\t<<<StableDiffusionModel ready!>>>\t")



    def get_external_conditions(
        self,
        clip_embeds_1: torch.FloatTensor,
        clip_embeds_2: Optional[torch.FloatTensor] = None,
        pooled_clip_embeds: Optional[torch.FloatTensor] = None,
        # transformer_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """
        По идее эта функция должна возвращать предобработанные условия на вход модели

        БЛЯДЬ ПРОСТО ТО ЖЕ САМОЕ ЧТО И БЫВШИЙ МЕТОД СД
        """

        if self.model_type == "sd15":
            output = (clip_embeds_1, None, None)

        elif self.model_type == "sdxl":
            output = (
                torch.concat([clip_embeds_1, clip_embeds_2], dim=-1),
                pooled_clip_embeds,
                # clip_embeds_2 if use_refiner else None,
                None
            )

        elif self.model_type == "sd3":
            pass

        elif self.model_type == "flux":
            pass


        return output
    


    def __call__(
        self,
        **kwargs,
    ):
       return 