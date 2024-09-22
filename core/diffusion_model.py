import torch

from typing import Optional
from dataclasses import dataclass

from .models.vae_model import VaeModel
from .models.backward_diffuser import ModelKey, Conditions, BackwardDiffuser



@dataclass
class DiffusionModelKey(ModelKey):
    is_latent_model: bool = True



@dataclass
class DiffusionConditions(Conditions):
    need_time_ids: bool = True
    need_timestep_cond: bool = False
    # ControlNet conditions
    # ...



class DiffusionModel(BackwardDiffuser):  
    vae: Optional[VaeModel] = None

    # Непонятно нужно ли пока использовать данные аргументы 
    # (ИЛИ ЛУЧШЕ ЗАПИХНУТЬ ИХ В ПАЙПЛАЙНЫ)
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    text_encoder_projection_dim: Optional[int] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        is_latent_model: bool = False,
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        # Инитим основную модель предсказания шума
        super().__init__( 
            dtype=dtype,
            device=device,
            model_path=model_path,
            model_type=model_type,
            scheduler_name=scheduler_name,
        )

        # Если латентная модель, то инитим ещё и vae
        if is_latent_model:
            self.vae = VaeModel(
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
            )
        self.is_latent_model = is_latent_model

        print("\t<<<DiffusionModel ready!>>>\t")


    @property
    def sample_size(self):
        return (
            self.predictor.config.sample_size * self.vae.scale_factor
            if self.is_latent_model else
            self.predictor.config.sample_size    
        )
    
    @property
    def num_channels(self):
        return (
            self.vae.config.latent_channels
            if self.is_latent_model else
            self.predictor.config.in_channels
        )
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # # TODO: добавить ControlNet
    # # ################################################################################################################ #
    # def get_extended_conditions(
    #     self,
    #     batch_size: int = 1,
    #     do_cfg: bool = False,
    #     **kwargs,
    # ) -> Optional[Conditions]:
    # # ################################################################################################################ #
    #     """
    #     Данный метод расширяет набор условий на хвод мордели своими внутренними условиями 
    #     или дополнительными условиями ControlNet модели
    #     """
    # # ################################################################################################################ #

    

    # # ================================================================================================================ #
    # def __call__(self, **kwargs):
    # # ================================================================================================================ #
    #     print("DiffusionModel --->")

    #     return self.get_extended_conditions(**kwargs)
    # # ================================================================================================================ #
    