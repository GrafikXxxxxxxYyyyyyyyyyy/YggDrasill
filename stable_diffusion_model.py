import torch

from typing import Optional
from dataclasses import dataclass

from .models.text_encoder_model import TextEncoderModel
from .core.diffusion_model import Conditions, DiffusionModel, DiffusionModelKey

# TODO: Убрать этот аргумент и заменить его прямыми выходами пайплайна на вход call
from .pipelines.text_encoder_pipeline import TextEncoderPipelineOutput



@dataclass
class StableDiffusionModelKey(DiffusionModelKey):
    use_ip_adapter: bool = False
    use_text_encoder: bool = True



class StableDiffusionModel(
    DiffusionModel,
    TextEncoderModel,
    # ImageEncoderModel,
    StableDiffusionModelKey
):  
    # Я пока хз делать ли это внутренними полями класса или нет
    use_ip_adapter: bool = False
    use_text_encoder: bool = True

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
    def __init__(
        self,
        use_ip_adapter: bool = False,
        use_text_encoder: bool = True,
        **kwargs,
    ): 
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
        # В любом случае инитим диффузионную модель
        DiffusionModel.__init__(self, **kwargs)
        
        # Опционально инитим картиночный и текстовый энкодеры
        if use_ip_adapter:
            # ImageEncoderModel.__init__(self, **kwargs)
            pass
        
        self.use_text_encoder = use_text_encoder
        if use_text_encoder:
            TextEncoderModel.__init__(self, **kwargs)

        print("\t<<<StableDiffusionModel ready!>>>\t")
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    


    
    def get_conditions(
        self,
        use_refiner: bool = False,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        te_output: Optional[TextEncoderPipelineOutput] = None,
        # ie_output: Optional[ImageEncoderPipelineOutput] = None,
        **kwargs,
    ) -> Conditions:
        # Устанавливаем собственные аргументы модели
        # TODO: Вот эту хуйню надо переделать
        self.maybe_switch_to_refiner(use_refiner)
        self.aesthetic_score = aesthetic_score
        self.negative_aesthetic_score = negative_aesthetic_score
        # Понадобилось так сделать, поскольку диффузионная модель
        # и текстовая модель лежат на разных ветках 
        self.text_encoder_projection_dim = self.projection_dim

        
        # Собираем текстовые и картиночные условия генерации
        conditions = Conditions()

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
    def __call__(
        self,
        use_refiner: bool = False,
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

        return self.get_conditions()
    # ================================================================================================================ #
        