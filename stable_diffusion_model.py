import torch

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from .core.diffusion_model import Conditions, DiffusionModel, DiffusionModelKey
from .pipelines.text_encoder_pipeline import TextEncoderModel, TextEncoderPipelineOutput


@dataclass
class StableDiffusionModelKey(DiffusionModelKey):
    pass



class StableDiffusionModel:
    diffuser: DiffusionModel
    text_encoder: Optional[TextEncoderModel] = None
    # image_encoder: Optional[ImageEncoderModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        is_latent_model: bool = True,
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler_name: Optional[str] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        self.diffuser = DiffusionModel(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
            scheduler_name=scheduler_name,
            is_latent_model=is_latent_model,
        )

        self.text_encoder = TextEncoderModel(
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
        )

        # self.image_encoder = 
        self.model_path = model_path
        self.model_type = model_type or "sd15"
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #    



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
        use_refiner: bool = False,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        te_output: Optional[TextEncoderPipelineOutput] = None,
        # ie_output: Optional[ImageEncoderPipelineOutput] = None,
        **kwargs,
    ) -> None:
    # ================================================================================================================ #
        """
        Подготавливает нужную последовательность входных аргументов
        и обуславливающих значений, соответсвующих заданной модели диффузии
        Также перенастраивает саму модельку
        """
        print("StableDiffusionModel --->")

        # if use_refiner:
        #     self.switch_to_refiner()
        
        conditions = Conditions()

        if te_output is not None:
            self.diffuser.do_cfg = te_output.do_cfg
            self.diffuser.guidance_scale = guidance_scale

            conditions.cross_attention_kwargs = te_output.cross_attention_kwargs

            batch_size = len(te_output.clip_embeds_1)
            batch_size = batch_size * num_images_per_prompt

            if self.model_type == "sd15":
                conditions.prompt_embeds = te_output.clip_embeds_1

            elif self.model_type == "sdxl":
                # add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                #     original_size = (height, width),
                #     crops_coords_top_left = (0, 0),
                #     aesthetic_score = aesthetic_score,
                #     negative_aesthetic_score = negative_aesthetic_score,
                #     target_size = (height, width),
                #     negative_original_size = (height, width),
                #     negative_crops_coords_top_left = (0, 0),
                #     negative_target_size = (height, width),
                #     addition_time_embed_dim = self.diffuser.predictor.config.addition_time_embed_dim,
                #     expected_add_embed_dim = self.diffuser.predictor.add_embed_dim,
                #     dtype = self.dtype,
                #     text_encoder_projection_dim = self.text_encoder.clip_encoder.text_encoder_projection_dim,
                #     requires_aesthetics_score = use_refiner,
                # )
                # add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                # add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)

                # if te_output.do_cfg:
                #     add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                pooled_prompt_embeds = te_output.pooled_clip_embeds.repeat(1, num_images_per_prompt)
                pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size, -1)
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    # "time_ids": add_time_ids.to(self.device),
                }

                conditions.added_cond_kwargs = added_cond_kwargs
                conditions.prompt_embeds = (
                    te_output.clip_embeds_2
                    if use_refiner else
                    torch.concat([te_output.clip_embeds_1, te_output.clip_embeds_2], dim=-1)       
                )
            
            elif self.model_type == "sd3":
                pass

            elif self.model_type == "flux":
                pass
            

            _, seq_len, _ = conditions.prompt_embeds.shape

            prompt_embeds = conditions.prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        
        # Помещаем собранные условия внутрь модели в формате специального класса-обёртки
        self.diffuser.conditions = conditions
    # ================================================================================================================ #
        