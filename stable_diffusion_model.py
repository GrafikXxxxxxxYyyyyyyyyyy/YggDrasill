import torch

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from .core.diffusion_model import Conditions, DiffusionModel, DiffusionModelKey
from .pipelines.text_encoder_pipeline import TextEncoderModel, TextEncoderPipelineOutput


@dataclass
class StableDiffusionModelKey(DiffusionModelKey):
    pass



class StableDiffusionModel:
    diffuser: DiffusionModel
    text_encoder: Optional[TextEncoderModel] = None
    # image_encoder: Optional[ImageEncoderModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        is_latent_model: bool = True,
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        self.diffuser = DiffusionModel(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
            scheduler_name=scheduler_name,
            is_latent_model=is_latent_model,
        )

        self.text_encoder = TextEncoderModel(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
        )

        # self.image_encoder = 
        self.model_path = model_path
        self.model_type = model_type or "sd15"
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
    


    # ================================================================================================================ #
    def __call__(
        self,
        use_refiner: bool = False,
        guidance_scale: float = 5.0,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        te_output: Optional[TextEncoderPipelineOutput] = None,
        # ie_output: Optional[ImageEncoderPipelineOutput] = None,
        **kwargs,
    ) -> Conditions:
    # ================================================================================================================ #
        """
        Подготавливает нужную последовательность входных аргументов
        и обуславливающих значений, соответсвующих заданной модели диффузии
        Также перенастраивает саму модельку
        """
        print("StableDiffusionModel --->")

        self.diffuser.use_refiner = use_refiner
        self.diffuser.aesthetic_score = aesthetic_score
        self.diffuser.negative_aesthetic_score = negative_aesthetic_score
        
        conditions = Conditions()
        
        if te_output is not None:
            conditions.cross_attention_kwargs = te_output.cross_attention_kwargs

            # ПЕРЕНАСТРОЙКА МОДЕЛИ
            self.diffuser.do_cfg = te_output.do_cfg
            self.diffuser.guidance_scale = guidance_scale
            self.diffuser.batch_size = te_output.batch_size

            if self.model_type == "sd15":
                conditions.prompt_embeds = te_output.clip_embeds_1

            elif self.model_type == "sdxl":
                pooled_prompt_embeds = te_output.pooled_clip_embeds
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                }

                conditions.added_cond_kwargs = added_cond_kwargs
                conditions.prompt_embeds = (
                    te_output.clip_embeds_2
                    if use_refiner else
                    torch.concat([te_output.clip_embeds_1, te_output.clip_embeds_2], dim=-1)       
                )

                # ПЕРЕНАСТРОЙКА МОДЕЛИ
                self.diffuser.text_encoder_projection_dim = self.text_encoder.clip_encoder.text_encoder_projection_dim
            
            elif self.model_type == "sd3":
                pass

            elif self.model_type == "flux":
                pass
        
        # Возвращаем собранные условия в формате специального класса-обёртки
        return conditions
    # ================================================================================================================ #
        