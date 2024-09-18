import torch

from typing import Optional, Dict

from .models.clip_te_model import CLIPTextEncoderModel



class TextEncoderModel(
    CLIPTextEncoderModel
):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:  
        # В любом случае инитим clip энкодер
        CLIPTextEncoderModel.__init__(
            self, 
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
        )

        # self.transformer_encoder = 

        self.model_path = model_path
        self.model_type = model_type or "sd15"


    # def __call__ 
    
