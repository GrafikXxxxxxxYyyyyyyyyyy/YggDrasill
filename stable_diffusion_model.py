import torch

from typing import Optional
from dataclasses import dataclass


from .conditioner.conditioner_model import ConditionerModel
from .core.diffusion_model import DiffusionModelKey, DiffusionModelConditions, DiffusionModel





@dataclass
class StableDiffusionModelKey(DiffusionModelKey):
    use_ip_adapter: bool = False
    use_text_encoder: bool = True





@dataclass
class StableDiffusionConditions(DiffusionModelConditions):
    use_refiner: bool = False






class StableDiffusionModel(DiffusionModel, ConditionerModel):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_text_encoder: bool = True,
        is_latent_model: bool = False,
        use_image_encoder: bool = False,
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ): 
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    
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

        # И возможно условную модель, если нужно обуславливание
        if use_text_encoder:
            ConditionerModel.__init__(
                self,
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
                use_image_encoder=use_image_encoder,
            )

        print("\t<<<StableDiffusionModel ready!>>>\t")
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    



    # def get_diffusion_conditions(
    #     self, 
    #     use_refiner:
    #     **kwargs
    # ):
    #     pass



    # ================================================================================================================ #
    def __call__(self, **kwargs):
    # ================================================================================================================ #
        print("DiffusionModel --->")

        return self.get_diffusion_conditions(**kwargs)
    # ================================================================================================================ #
        