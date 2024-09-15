import torch

from dataclasses import dataclass
from typing import Optional, Tuple

from .models.vae_model import VaeModel
from .models.noise_predictor import ModelKey, NoisePredictor
from .pipelines.backward_diffusion import Conditions, BackwardDiffusion


@dataclass
class DiffusionModelKey(ModelKey):
    is_latent_model: bool = True
    scheduler_name: str = "euler"


class DiffusionModel(DiffusionModelKey):
    # Основные параметры модели
    key: DiffusionModelKey
    predictor: NoisePredictor
    vae: Optional[VaeModel] = None

    # Параметры, которые можно вынести в функциональный класс
    batch_size: int = 1
    do_cfg: bool = False
    use_refiner: bool = False
    guidance_scale: float = 5.0
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    text_encoder_projection_dim: Optional[int] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        is_latent_model: bool = False,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):     
        self.key = DiffusionModelKey(
            scheduler_name=scheduler_name,
            is_latent_model=is_latent_model,
            **kwargs,
        )
        self.predictor = NoisePredictor(**kwargs)
        self.vae = VaeModel(**kwargs) if is_latent_model else None

    @property
    def dtype(self):
        return self.predictor.dtype

    @property
    def device(self):
        return self.predictor.device

    @property
    def sample_size(self):
        return (
            self.predictor.config.sample_size * self.vae.scale_factor
            if self.vae is not None else
            self.predictor.config.sample_size    
        )
    
    @property
    def num_channels(self):
        return (
            self.vae.config.latent_channels
            if self.vae is not None else
            self.predictor.config.in_channels
        )
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # def switch_to_refiner(self):
    #     pass


    
    # TODO: Вынести в отдельный функциональный класс  или чет типа того
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        aesthetic_score,
        negative_aesthetic_score,
        target_size,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        addition_time_embed_dim,
        expected_add_embed_dim,
        dtype,
        text_encoder_projection_dim,
        requires_aesthetics_score,
    ):
        if requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )

        if (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids



    # ================================================================================================================ #
    def __call__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        conditions: Optional[Conditions] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        masked_image: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Tuple[BackwardDiffusion, Optional[Conditions]]:
    # ================================================================================================================ #
        """
        Вызов данной модели работает как фабрика, выплёвывающая нужную реализацию
        пайплайна BackwardDiffusion
        """
        print("DiffusionModel --->")
        
        if "1. Дополняет условия в соответствии с моделью":
            if conditions is not None:
                if self.model_type == "sd15":
                    pass
                elif self.model_type == "sdxl":
                    # Для модели SDXL почему-то нужно обязательно расширить 
                    # дополнительные аргументы временными метками 
                    add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                        original_size = (height, width),
                        crops_coords_top_left = (0, 0),
                        aesthetic_score = self.aesthetic_score,
                        negative_aesthetic_score = self.negative_aesthetic_score,
                        target_size = (height, width),
                        negative_original_size = (height, width),
                        negative_crops_coords_top_left = (0, 0),
                        negative_target_size = (height, width),
                        addition_time_embed_dim = self.predictor.config.addition_time_embed_dim,
                        expected_add_embed_dim = self.predictor.add_embed_dim,
                        dtype = self.dtype,
                        text_encoder_projection_dim = self.text_encoder_projection_dim,
                        requires_aesthetics_score = self.use_refiner,
                    )
                    add_time_ids = add_time_ids.repeat(self.batch_size, 1)
                    add_neg_time_ids = add_neg_time_ids.repeat(self.batch_size, 1)
                    if self.do_cfg:
                        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
                    
                    conditions.added_cond_kwargs["time_ids"] = add_time_ids.to(self.device)

                elif self.model_type == "sd3":
                    pass
                elif self.model_type == "flux":
                    pass



        if "2. Формирует класс BacwardDiffusion":
            bacward_step_pipeline = BackwardDiffusion(**self.key)
            bacward_step_pipeline.predictor = self.predictor

            # Устанавливает в пайплайн метку cfg
            bacward_step_pipeline.do_cfg = self.do_cfg
            # Устанавливает там же guidance_scale
            bacward_step_pipeline.guidance_scale = self.guidance_scale

            # Правильно инитит маску и маскированную картинку в пайпе
            if mask_image is not None:
                mask_image = mask_image.repeat(
                    self.batch_size // mask_image.shape[0], 1, 1, 1
                )
                bacward_step_pipeline.mask_sample = (
                    torch.cat([mask_image] * 2)
                    if self.do_cfg else
                    mask_image
                )

            if masked_image is not None:
                masked_image = masked_image.repeat(
                    self.batch_size // masked_image.shape[0], 1, 1, 1
                )
                bacward_step_pipeline.masked_sample = (
                    torch.cat([masked_image] * 2)
                    if self.do_cfg else
                    masked_image
                )
            
            # Возвращает собранный экземпляр пайплайна и изменённые условия если те были
            return bacward_step_pipeline, conditions
    # ================================================================================================================ #