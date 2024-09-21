import torch

from typing import Optional
from dataclasses import dataclass

from .models.text_encoder_model import TextEncoderModel
from .pipelines.text_encoder_pipeline import TextEncoderPipelineOutput
from .core.diffusion_model import DiffusionModelKey, DiffusionConditions, DiffusionModel



@dataclass
class StableDiffusionModelKey(DiffusionModelKey):
    use_ip_adapter: bool = False
    use_text_encoder: bool = True



@dataclass
class StableDiffusionConditions(DiffusionConditions):
    pass



class StableDiffusionModel(DiffusionModel):  
    text_encoder: Optional[TextEncoderModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_text_encoder: bool = True,
        is_latent_model: bool = False,
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ): 
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
        # Инитим диффузионную модель
        super().__init__(
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
            scheduler_name=scheduler_name,
            is_latent_model=is_latent_model,
        )
        
        # Опционально инитим текстовый энкодер
        self.text_encoder = (
            TextEncoderModel(
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
            )
            if use_text_encoder else
            None
        )

        print("\t<<<StableDiffusionModel ready!>>>\t")
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    



    # TODO: Тут что-то должна делать модель self.text_encoder если она есть
    def retrieve_conditions(
        self,
        use_refiner: bool = False,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        te_output: Optional[TextEncoderPipelineOutput] = None,
        **kwargs,
    ) -> StableDiffusionConditions:
        """
        Подготавливает условные аргументы с энкодеров и самой модели
        """
        # Собираем текстовые и картиночные условия генерации
        conditions = StableDiffusionConditions()

        if te_output is not None:
            conditions.cross_attention_kwargs = te_output.cross_attention_kwargs

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
            
            elif self.model_type == "sd3":
                pass

            elif self.model_type == "flux":
                pass
        
        
        return conditions
        



    # ================================================================================================================ #
    def __call__(self, **kwargs):
    # ================================================================================================================ #
        print("DiffusionModel --->")

        return self.retrieve_conditions(**kwargs)
    # ================================================================================================================ #
        