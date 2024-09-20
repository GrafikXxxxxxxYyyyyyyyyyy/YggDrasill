import torch

from diffusers import (
    UNet2DModel, 
    UNet2DConditionModel,
    SD3Transformer2DModel,
    FluxTransformer2DModel,
)
from typing import Optional, Union



class NoisePredictor:
    predictor: Union[
        UNet2DModel, 
        UNet2DConditionModel,
        SD3Transformer2DModel,
        FluxTransformer2DModel,
    ]
    
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        self.predictor = UNet2DConditionModel.from_pretrained(
            model_path, 
            subfolder='unet', 
            torch_dtype=dtype,
            variant='fp16',
            use_safetensors=True
        )
        self.predictor.to(device=device, dtype=dtype)

        # Инитим константы
        self.model_path = model_path
        self.model_type = model_type or "sd15"
        print(f"NoisePredictor model has successfully loaded from '{model_path}' checkpoint!")
        
    @property
    def config(self):
        return self.predictor.config
    
    @property
    def dtype(self):
        return self.predictor.dtype
    
    @property
    def device(self):
        return self.predictor.device
    
    @property
    def is_latent_model(self):
        return self.predictor.config.in_channels == 4

    # ALL RIGHT!
    @property
    def is_inpainting_model(self):
        return (
            self.predictor.config.in_channels == 9 
            or self.predictor.config.in_channels == 7
        )
    
    @property
    def add_embed_dim(self):
        return self.predictor.add_embedding.linear_1.in_features
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # ================================================================================================================ #
    def __call__(
        self,
        timestep: int,
        noisy_sample: torch.FloatTensor,
        # ПО ИДЕЕ УСЛОВИЯ ТУТ НЕ ДОЛЖНЫ ПЕРЕДАВАТЬСЯ, А ДОЛЖНЫ ПАДАТЬ В КВАРГАХ
        # СО СЛОЯ ВЫШЕ
        # conditions: Optional[Conditions] = None,
        **kwargs,
    ) -> torch.FloatTensor:          
    # ================================================================================================================ #
        extra_kwargs = {}

        # Пересобираем пришедшие аргументы под нужную архитектуру(!), если те переданы
        if isinstance(self.predictor, UNet2DModel):
            extra_kwargs["class_labels"] = conditions.class_labels

        elif isinstance(self.predictor, UNet2DConditionModel):
            extra_kwargs["class_labels"] = conditions.class_labels
            extra_kwargs["timestep_cond"] = conditions.timestep_cond
            extra_kwargs["added_cond_kwargs"] = conditions.added_cond_kwargs
            extra_kwargs["encoder_hidden_states"] = conditions.prompt_embeds
            extra_kwargs["cross_attention_kwargs"] = conditions.cross_attention_kwargs

        elif isinstance(self.predictor, SD3Transformer2DModel):
            pass

        elif isinstance(self.predictor, FluxTransformer2DModel):
            pass


        print(f"Step: {timestep}")
        # Предсказывает шум моделью + собранными параметрами
        predicted_noise = self.predictor(
            timestep=timestep,
            sample=noisy_sample,
            **extra_kwargs,
        )
        print(f"Back step: {timestep}")


        return predicted_noise
    # ================================================================================================================ #









# def forward(
#     self,
#     sample: torch.Tensor,
#     timestep: Union[torch.Tensor, float, int],
#     class_labels: Optional[torch.Tensor] = None,
#     return_dict: bool = True,
# ) -> Union[UNet2DOutput, Tuple]:




# def forward(
#         self,
#         sample: torch.Tensor,
#         timestep: Union[torch.Tensor, float, int],
#         encoder_hidden_states: torch.Tensor,
#         class_labels: Optional[torch.Tensor] = None,
#         timestep_cond: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#         added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
#         down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
#         mid_block_additional_residual: Optional[torch.Tensor] = None,
#         down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
#         encoder_attention_mask: Optional[torch.Tensor] = None,
#         return_dict: bool = True,
#     ) -> Union[UNet2DConditionOutput, Tuple]:
#         r"""
#         The [`UNet2DConditionModel`] forward method.




# def forward(
#         self,
#         hidden_states: torch.FloatTensor,
#         encoder_hidden_states: torch.FloatTensor = None,
#         pooled_projections: torch.FloatTensor = None,
#         timestep: torch.LongTensor = None,
#         block_controlnet_hidden_states: List = None,
#         joint_attention_kwargs: Optional[Dict[str, Any]] = None,
#         return_dict: bool = True,
#     ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
#         """
#         The [`SD3Transformer2DModel`] forward method.




# def forward(
#         self,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: torch.Tensor = None,
#         pooled_projections: torch.Tensor = None,
#         timestep: torch.LongTensor = None,
#         img_ids: torch.Tensor = None,
#         txt_ids: torch.Tensor = None,
#         guidance: torch.Tensor = None,
#         joint_attention_kwargs: Optional[Dict[str, Any]] = None,
#         return_dict: bool = True,
#     ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
#         """
#         The [`FluxTransformer2DModel`] forward method.

