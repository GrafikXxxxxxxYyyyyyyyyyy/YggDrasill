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
    DiffusionPipelineOutput,
)
from .pipelines.text_encoder_pipeline import (
    TextEncoderPipeline,
    TextEncoderPipelineInput,
    TextEncoderPipelineOutput,
)


@dataclass
class StableDiffusionPipelineInput(BaseOutput):
    diffusion_input: DiffusionPipelineInput
    use_refiner: bool = False
    guidance_scale: float = 5.0
    te_input: Optional[TextEncoderPipelineInput] = None


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



class StableDiffusionPipeline(
    DiffusionPipeline,
    TextEncoderPipeline,  
):  
    sd: StableDiffusionModel

    def __init__(
        self,
        model_key: Optional[StableDiffusionModelKey] = None,
        **kwargs,
    ):
        if model_key is not None:
            self.sd = StableDiffusionModel(**model_key)
            self.diffuser = self.sd.diffuser
            self.text_encoder = self.sd.text_encoder
            self.vae = self.sd.diffuser.vae
            self.predictor = self.sd.diffuser.predictor
            self.scheduler = self.sd.diffuser.scheduler


    def __call__(
        self,
        model: StableDiffusionModel,
        diffusion_input: DiffusionPipelineInput,
        te_input: Optional[TextEncoderPipelineInput] = None,
        use_refiner: bool = False,
        guidance_scale: float = 5.0,
            # refiner_steps: Optional[int] = None,
            # refiner_scale: Optional[float] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        **kwargs,
    ):
        print("StableDiffusionPipeline --->")

        if "1. Собираем и преобразуем обуславливающую информацию":
            if model.text_encoder is not None and te_input is not None:
                te_output = self.get_prompt_embeddings(**te_input)

                self.do_cfg = te_output.do_cfg
                self.guidance_scale = guidance_scale
                # Изменяет размер тензора т.к может быть несколько картинок на промпт
                diffusion_input.batch_size = te_output.batch_size


        # Вызываем модель с доп аргументами, чтобы создать из эмбеддингов
        # входы для диффузионной модели
        conditions = self.sd(
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











