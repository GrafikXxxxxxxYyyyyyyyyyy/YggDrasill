import torch

from typing import Optional, Dict

from .models.clip_te_model import CLIPTextEncoderModel



class TextEncoderModel(CLIPTextEncoderModel):
        # transformer_encoder: 

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
        super().__init__(
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
        )

        self.model_path = model_path
        self.model_type = model_type or "sd15"



    def __call__(
        self,
    ):
        """
        По идее должен делать что-то типа подготовки нужных аргументов для модели 
        UPD: Нихуя! Он должен когерентные состояния высчитывать вот че он должен!
        UPD2: Ну не прям нихуя, ну ты понял 
        """

        return
