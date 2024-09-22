import torch 

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from .core.diffusion_pipeline import DiffusionPipeline, DiffusionPipelineInput
from .pipelines.text_encoder_pipeline import TextEncoderPipeline, TextEncoderPipelineInput
from .stable_diffusion_model import StableDiffusionModel, StableDiffusionModelKey, StableDiffusionConditions






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
    TextEncoderPipeline
):  
    model: Optional[StableDiffusionModel] = None

    # REFINER!!! #
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
            # aesthetic_score: float = 6.0,
            # negative_aesthetic_score: float = 2.5,
        te_input: Optional[TextEncoderPipelineInput] = None,
        **kwargs,
    ):  
    # ================================================================================================================ #
        self.model = model

        if "1. Собираем и преобразуем обуславливающую информацию":
            if self.model.use_text_encoder and te_input is not None:
                te_output = self.encode_prompt(**te_input)

                diffusion_input.do_cfg = te_output.do_cfg
                diffusion_input.batch_size = te_output.batch_size


        conditions = self.model.retrieve_conditions(
            te_output=te_output,
        )

        print(conditions)


        # # Создаём новый расширенный Условиями класс инпута
        # diffusion_input = DiffusionPipelineInput(
        #     # conditions=Conditions(**conditions),
        #     conditions=conditions,
        #     **diffusion_input,
        # )
        # if "2. Учитывая переданные аргументы, используем полученный/ые пайплайны":
        #     deffusion_pipeline = DiffusionPipeline()
        #     diffusion_output = deffusion_pipeline(
        #         diffuser=model,
        #         **diffusion_input
        #     )


        # return StableDiffusionPipelineOutput(
        #     images=diffusion_output.images
        # )  
    # ================================================================================================================ #











