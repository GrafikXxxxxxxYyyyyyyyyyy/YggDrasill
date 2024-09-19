import torch 

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from .stable_diffusion_model import (
    Conditions, 
    StableDiffusionModel, 
    StableDiffusionModelKey,
)
from .core.diffusion_pipeline import (
    DiffusionPipeline, 
    DiffusionPipelineInput,
)
from .pipelines.text_encoder_pipeline import (
    TextEncoderPipeline,
    TextEncoderPipelineInput,
)



@dataclass
class StableDiffusionPipelineInput(BaseOutput):
    diffusion_input: DiffusionPipelineInput
    use_refiner: bool = False
    guidance_scale: float = 5.0
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    te_input: Optional[TextEncoderPipelineInput] = None



@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



class StableDiffusionPipeline(
    DiffusionPipeline,
    TextEncoderPipeline,  
):  
    model: Optional[StableDiffusionModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
            # use_refiner: bool = False,
            # refiner_steps: Optional[int] = None,
            # refiner_scale: Optional[float] = None,
        model_key: Optional[StableDiffusionModelKey] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            self.model = StableDiffusionModel(**model_key)
            self.device = self.model.device
            self.dtype = self.model.dtype
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    def stable_diffusion_process(
        self,
        diffusion_input: DiffusionPipelineInput,
        use_refiner: bool = False,
        guidance_scale: float = 5.0,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        te_input: Optional[TextEncoderPipelineInput] = None,
        **kwargs,
    ):
        if "1. Собираем и преобразуем обуславливающую информацию":
            if self.model.use_text_encoder is not None and te_input is not None:
                te_output = self.encode_prompt(**te_input)

                self.do_cfg = te_output.do_cfg
                self.guidance_scale = guidance_scale
                # Изменяет размер тензора т.к может быть несколько картинок на промпт
                diffusion_input.batch_size = te_output.batch_size


        # Вызываем модель с доп аргументами, чтобы создать из эмбеддингов
        # входы для диффузионной модели
        conditions = self.model.get_conditions(
            te_output=te_output,
            use_refiner=use_refiner,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
        )

        # Создаём новый расширенный словиями класс инпута
        diffusion_input = DiffusionPipelineInput(
            # conditions=Conditions(**conditions),
            conditions=conditions,
            **diffusion_input,
        )
        if "2. Учитывая переданные аргументы, используем полученный/ые пайплайны":
            diffusion_output = self.diffusion_process(**diffusion_input)


        return StableDiffusionPipelineOutput(
            images=diffusion_output.images
        )  



    # ================================================================================================================ #
    def __call__(
        self,
        input: StableDiffusionPipelineInput,
        model: Optional[StableDiffusionModel] = None,
        **kwargs,
    ):
    # ================================================================================================================ #
        print("StableDiffusionPipeline --->")

        if (
            model is not None 
            and isinstance(model, StableDiffusionModel)
        ):
            self.model = model
            self.scheduler = model.scheduler

        return self.stable_diffusion_process(**input)
    # ================================================================================================================ #











