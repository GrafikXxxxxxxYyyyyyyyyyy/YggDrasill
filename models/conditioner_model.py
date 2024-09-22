import torch

from typing import Optional
from dataclasses import dataclass

from .models.text_encoder_model import TextEncoderModel



class ConditionerModel(TextEncoderModel):
        # image_encoder: Optional[ImageEncoderModel] = None

    def __init__(
        self,
        use_image_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # if use_image_encoder:
        #     self.image_encoder = ImageEncoderModel(**kwargs)



    def __call__(
        self,

    ):
        pass