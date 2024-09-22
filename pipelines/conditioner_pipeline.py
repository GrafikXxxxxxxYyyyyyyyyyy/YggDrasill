import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Union, Dict, Any

from ..models.conditioner_model import ConditionerModel
from ..stable_diffusion_model import StableDiffusionModelKey, StableDiffusionConditions
from .pipelines.text_encoder_pipeline import TextEncoderPipeline, TextEncoderPipelineInput, TextEncoderPipelineOutput





@dataclass
class ConditionerPipelineInput(BaseOutput):
    pass






@dataclass
class ConditionerPipelineOutput(BaseOutput):
    prompt_embeds: torch.FloatTensor 
    batch_size: int = 1
    do_cfg: bool = False
    cross_attention_kwargs: Optional[dict] = None
    text_embeds: Optional[torch.FloatTensor] = None
    refiner_prompt_embeds: Optional[torch.FloatTensor] = None





class ConditionerPipeline(
    TextEncoderPipeline,
        # ImageEncoderPipeline
):
    model: Optional[ConditionerModel] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_key: Optional[StableDiffusionModelKey] = None,
        **kwargs,
    ):
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        if model_key is not None:
            self.model = ConditionerModel(**model_key)
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
    


    # ################################################################################################################ #
    def retrieve_conditions(
        self,
        te_input: Optional[TextEncoderPipelineInput] = None,
            # ie_output: Optional[ImageEncoderPipelineOutput] = None,
        **kwargs,
    ):
    # ################################################################################################################ #
        # Собираем текстовые и картиночные условия генерации
        conditions = StableDiffusionConditions()

        te_output: Optional[TextEncoderPipelineOutput] = None
        if "1. Вызывам собственный энкодер":
            te_output = self.encode_prompt(**te_input)

            do_cfg = te_output.do_cfg
            batch_size = te_output.batch_size
            cross_attention_kwargs = te_output.cross_attention_kwargs



        if "2. Вызываем собственную модельку":
            (
                prompt_embeds, 
                text_embeds, 
                refiner_prompt_embeds
            ) = self.model.get_conditions_from_embeddings(
                **te_output
            )
        
        
        
        return ConditionerPipelineOutput(
            do_cfg=do_cfg,
            batch_size=batch_size,
            text_embeds=text_embeds,
            prompt_embeds=prompt_embeds,
            refiner_prompt_embeds=refiner_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )        


        # if "3. Собираем собственные условия под нужную модель":
        #     if self.model.model_type == "sd15":
        #         pass

        #     elif self.model.model_type == "sdxl":
        #         # Для модели SDXL почему-то нужно обязательно расширить 
        #         # дополнительные аргументы временными метками 
        #         add_time_ids, add_neg_time_ids = self._get_add_time_ids(
        #             original_size = (height, width),
        #             crops_coords_top_left = (0, 0),
        #             aesthetic_score = self.aesthetic_score,
        #             negative_aesthetic_score = self.negative_aesthetic_score,
        #             target_size = (height, width),
        #             negative_original_size = (height, width),
        #             negative_crops_coords_top_left = (0, 0),
        #             negative_target_size = (height, width),
        #             addition_time_embed_dim = self.predictor.config.addition_time_embed_dim,
        #             expected_add_embed_dim = self.add_embed_dim,
        #             dtype = self.model.dtype,
        #             text_encoder_projection_dim = self.text_encoder_projection_dim,
        #             requires_aesthetics_score = self.use_refiner,
        #         )
        #         add_time_ids = add_time_ids.repeat(batch_size, 1)
        #         add_neg_time_ids = add_neg_time_ids.repeat(batch_size, 1)
        #         if do_cfg:
        #             add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
                
        #         conditions.added_cond_kwargs["time_ids"] = add_time_ids.to(self.model.device)

        #     elif self.model.model_type == "sd3":
        #         pass

        #     elif self.model.model_type == "flux":
        #         pass
    # ################################################################################################################ #



    # ================================================================================================================ #
    def __call__(
        self,
        conditioner: Optional[ConditionerModel] = None,
        **kwargs,
    ) -> ConditionerPipelineOutput:
    # ================================================================================================================ #
        if (
            conditioner is not None 
            and isinstance(conditioner, ConditionerModel)
        ):
            self.model = conditioner

        return self.retrieve_conditions(**kwargs)
    # ================================================================================================================ #


    





