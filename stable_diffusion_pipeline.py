import torch 

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from YggDrasill.stable_diffusion_model import (
    Conditions, 
    StableDiffusionModel, 
    StableDiffusionModelKey,
)
from YggDrasill.core.diffusion_pipeline import (
    DiffusionPipeline, 
    DiffusionPipelineInput,
)
from YggDrasill.pipelines.text_encoder_pipeline import (
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



class StableDiffusionPipeline:  
    use_refiner: bool = False
    refiner_steps: Optional[int] = None
    refiner_scale: Optional[float] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
            # use_refiner: bool = False,
            # refiner_steps: Optional[int] = None,
            # refiner_scale: Optional[float] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        pass
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # ================================================================================================================ #
    def __call__(
        self,
        model: StableDiffusionModel,
        diffusion_input: DiffusionPipelineInput,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        te_input: Optional[TextEncoderPipelineInput] = None,
        **kwargs,
    ):  
    # ================================================================================================================ #
        if "1. Собираем и преобразуем обуславливающую информацию":
            if model.use_text_encoder and te_input is not None:
                # TODO: Заинитить TEPipe
                te_pipeline = TextEncoderPipeline()
                te_output = te_pipeline(
                    text_encoder=model,
                    **te_input
                )

                diffusion_input.do_cfg = te_output.do_cfg
                diffusion_input.batch_size = te_output.batch_size


        # Вызываем модель с доп аргументами, чтобы создать из эмбеддингов
        # входы для диффузионной модели
        conditions = model.get_conditions(
            te_output=te_output,
        )

        # TODO: Учесть тут процедуру рефайнера
        # <...>
        # Создаём новый расширенный Условиями класс инпута
        diffusion_input = DiffusionPipelineInput(
            # conditions=Conditions(**conditions),
            conditions=conditions,
            **diffusion_input,
        )
        if "2. Учитывая переданные аргументы, используем полученный/ые пайплайны":
            deffusion_pipeline = DiffusionPipeline()
            diffusion_output = deffusion_pipeline(
                diffuser=model,
                **diffusion_input
            )


        return StableDiffusionPipelineOutput(
            images=diffusion_output.images
        )  
    # ================================================================================================================ #











