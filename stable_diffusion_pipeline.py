import torch 

from typing import List, Optional
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
    # ie_input: Optional[ImageEncoderPipelineInput] = None


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor



class StableDiffusionPipeline:
    model: StableDiffusionModel
    te_pipeline: TextEncoderPipeline
    diffusion_pipeline: DiffusionPipeline

    def __call__(
        self,
        model: StableDiffusionModel,
        diffusion_input: DiffusionPipelineInput,
        te_input: Optional[TextEncoderPipelineInput] = None,
        use_refiner: bool = False,
        guidance_scale: float = 5.0,
                # ip_adapter_image: Optional[PipelineImageInput] = None,
                # output_type: str = "pt",
        refiner_steps: Optional[int] = None,
        refiner_scale: Optional[float] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        **kwargs,
    ):
        print("StableDiffusionPipeline --->")
        self.model = model

        if "1. Собираем и преобразуем обуславливающую информацию":
            # Эта логика должна быть выстроена сверху вниз, чтобы 
            # сойтись с логикой диффузионного процесса, которая будет 
            # прописана снизу вверх для ветки core
            if model.text_encoder is not None and te_input is not None:
                te_pipeline = TextEncoderPipeline()
                te_output = te_pipeline(
                    model.text_encoder,
                    **te_input,
                )


        conditions = model(
            te_output=te_output,
            use_refiner=use_refiner,
            guidance_scale=guidance_scale,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
        )
   

        if "3. Учитывая переданные аргументы, используем полученный/ые пайплайны":
            diffusion_pipeline = DiffusionPipeline()
            output = diffusion_pipeline(
                diffuser=model.diffuser,
                conditions=conditions,
                **diffusion_input,
            )
            pass

        

        return StableDiffusionPipelineOutput(
            images=output.images
        )











