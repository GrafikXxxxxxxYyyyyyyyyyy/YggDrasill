import torch

from typing import Optional, Dict

from .models.clip_te_model import CLIPTextEncoderModel



class TextEncoderModel:
    clip_encoder: CLIPTextEncoderModel
    # t5_encoder: Optional[TransformerModel] = None

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:  
        # Инитим модель CLIP
        self.clip_encoder = CLIPTextEncoderModel(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
        )

        # self.transformer_encoder = 

        self.model_path = model_path
        self.model_type = model_type or "sd15"


    
