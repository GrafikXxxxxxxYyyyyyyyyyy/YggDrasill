import torch 

from typing import Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput

from .core.diffusion_pipeline import DiffusionPipeline, DiffusionPipelineInput
from .conditioner.conditioner_pipeline import ConditionerPipeline, ConditionerPipelineInput
from .stable_diffusion_model import StableDiffusionModel, StableDiffusionModelKey, StableDiffusionConditions






@dataclass
class StableDiffusionPipelineInput(BaseOutput):
    diffusion_input: DiffusionPipelineInput
    # use_refiner: bool = False
    # aesthetic_score: float = 6.0
    # negative_aesthetic_score: float = 2.5
    conditioner_input: Optional[ConditionerPipelineInput] = None






@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    images: torch.FloatTensor






class StableDiffusionPipeline(ConditionerPipeline, DiffusionPipeline):  

    model: Optional[StableDiffusionModel] = None

    # REFINER!!! #
    use_refiner: bool = False
    refiner_steps: Optional[int] = None
    refiner_scale: Optional[float] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_key: Optional[StableDiffusionModelKey] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            self.model = StableDiffusionModel(**model_key)
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # ================================================================================================================ #
    def __call__(
        self,
        model: StableDiffusionModel,
        diffusion_input: DiffusionPipelineInput,
        # use_refiner: bool = False,
        # aesthetic_score: float = 6.0,
        # negative_aesthetic_score: float = 2.5,
        conditioner_input: Optional[ConditionerPipelineInput] = None,
        **kwargs,
    ):  
    # ================================================================================================================ #
        self.model = model

        if "1. Собираем и преобразуем обуславливающую информацию":
            conditioner_output = self.retrieve_external_conditions(**conditioner_input)

        print(conditioner_output)
            
        # if "2. Вызываем лежащую внутри модельку":
        #     conditions = self.model.get_diffusion_conditions(

        #     )

        if "2. Запускаем диффузионный процесс с учётом условной информации":
            diffusion_output = self.diffusion_process(
                **DiffusionPipelineInput(
                    conditions=conditions,
                    **diffusion_input,
                )
            )



        return StableDiffusionPipelineOutput(**diffusion_output)  
    # ================================================================================================================ #











