import torch

from typing import Optional, Dict

from .models.clip_te_model import CLIPTextEncoderModel



class TextEncoderModel:
    clip_encoder: CLIPTextEncoderModel
        # transformer_encoder: 

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
        self.clip_encoder = CLIPTextEncoderModel(
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
        )

        self.model_path = model_path
        self.model_type = model_type or "sd15"


    # def __call__(
        
    # )
    
