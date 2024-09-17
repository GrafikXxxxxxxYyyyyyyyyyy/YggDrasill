import torch 

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from .pipelines.text_encoder_pipeline import (
    TextEncoderPipeline,
    TextEncoderPipelineInput,
)
from .stable_diffusion_model import StableDiffusionModel
from .core.diffusion_pipeline import (
    Conditions,
    DiffusionPipeline, 
    DiffusionPipelineInput
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
                te_output = self.get_prompt_embeddings(
                    model.text_encoder,
                    **te_input,
                )

                self.do_cfg = te_output.do_cfg
                self.guidance_scale = guidance_scale
                # Изменяет размер тензора т.к может быть несколько картинок на промпт
                diffusion_input.batch_size = te_output.batch_size


        # Вызываем модель с доп аргументами, чтобы создать из эмбеддингов
        # входы для диффузионной модели
        conditions = model(
            te_output=te_output,
            use_refiner=use_refiner,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
        )
        print(f"ConditionsSDP after model: {conditions}")
   
        diffusion_input.conditions = conditions
        if "2. Учитывая переданные аргументы, используем полученный/ые пайплайны":
            diffusion_output = self.diffusion_process(
                model.diffuser,
                **diffusion_input,
            )

        

        # return StableDiffusionPipelineOutput(
        #     images=output.images
        # )











