import torch

from typing import Optional
from dataclasses import dataclass

from .models.text_encoder_model import TextEncoderModel
from .pipelines.text_encoder_pipeline import TextEncoderPipelineOutput
from .core.diffusion_model import DiffusionModel, DiffusionModelKey, DiffusionConditions






@dataclass
class StableDiffusionModelKey(DiffusionModelKey):
    use_ip_adapter: bool = False
    use_text_encoder: bool = True






@dataclass
class StableDiffusionConditions(DiffusionConditions):
    pass






class StableDiffusionModel(
    DiffusionModel,
    # TODO: Сделать ключ внутренним полем классов-моделей
    StableDiffusionModelKey
):  
    text_encoder: Optional[TextEncoderModel] = None
    # image_encoder:

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_ip_adapter: bool = False,
        use_text_encoder: bool = True,
        is_latent_model: bool = False,
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ): 
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
        # Инитит класс ключа, просто чтобы сохранить параметры модели

            # TODO: Просто сохранять внутри модели поле ключа-класса
            # StableDiffusionModelKey.__init__(
            #     self, 
            #     dtype=dtype,
            #     device=device,
            #     model_path=model_path,
            #     model_type=model_type,
            #     scheduler_name=scheduler_name,
            #     is_latent_model=is_latent_model,
            #     use_ip_adapter=use_ip_adapter,
            #     use_text_encoder=use_text_encoder,
            # )

        # Инитим диффузионную модель
        DiffusionModel.__init__(
            self, 
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
            scheduler_name=scheduler_name,
            is_latent_model=is_latent_model,
        )
        
        # Опционально инитим текстовый энкодер
        if use_text_encoder:
            self.text_encoder = TextEncoderModel(
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
            )

        print("\t<<<StableDiffusionModel ready!>>>\t")
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    



    def retrieve_conditions(
        self,
        # REFINER!!! #
        use_refiner: bool = False,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        te_output: Optional[TextEncoderPipelineOutput] = None,
        # ie_output: Optional[ImageEncoderPipelineOutput] = None,
        **kwargs,
    ) -> StableDiffusionConditions:
        """
        """
            # # Устанавливаем собственные аргументы модели
            # self.maybe_switch_to_refiner(use_refiner)
            # # Переделать хуйню ниже, это ваще бред 
            # self.text_encoder_projection_dim = self.projection_dim

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
        



    # # ================================================================================================================ #
    # def __call__(
    #     self,
    #     use_refiner: bool = False,
    #     aesthetic_score: float = 6.0,
    #     negative_aesthetic_score: float = 2.5,
    #     # te_output: Optional[TextEncoderPipelineOutput] = None,
    #     # ie_output: Optional[ImageEncoderPipelineOutput] = None,
    #     **kwargs,
    # ) -> StableDiffusionConditions:
    # # ================================================================================================================ #
    #     """
    #     Подготавливает нужную последовательность входных аргументов
    #     и обуславливающих значений, соответсвующих заданной модели диффузии
    #     Также перенастраивает саму модельку
    #     """ 
    #     pass
    # # ================================================================================================================ #
        