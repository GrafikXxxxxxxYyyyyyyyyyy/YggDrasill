import torch 

from dataclasses import dataclass
from typing import List, Optional, Union
from diffusers.utils import BaseOutput

# from .pipelines.clip_te_pipeline import (
#     CLIPTextEncoderPipeline,
#     CLIPTextEncoderPipelineInput,
# )
# from .pipelines.transformer_te_pipeline import (
#     TransformerTextEncoderPipeline,
#     TransformerTextEncoderPipelineInput,
#     TransformerTextEncoderPipelineOutput
# )
from ..models.text_encoder_model import TextEncoderModel


@dataclass
class TextEncoderPipelineInput(CLIPTextEncoderPipelineInput):
    prompt: Optional[Union[str, List[str]]] = None


@dataclass
class TextEncoderPipelineOutput(BaseOutput):
    do_cfg: bool
    clip_embeds_1: torch.FloatTensor
    lora_scale: Optional[float] = None
    clip_embeds_2: Optional[torch.FloatTensor] = None
    pooled_clip_embeds: Optional[torch.FloatTensor] = None
    # t5_embeds: Optional[torch.FloatTensor] = None



class TextEncoderPipeline:
    clip_pipeline: CLIPTextEncoderPipeline
    # transformer_pipeline: TransformerTextEncoderPipeline

    def __call__(
        self,
        text_encoder: TextEncoderModel,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        print("TextEncoderPipeline --->")




        if "1. Подготавливаем необходимые аргументы":
            # Устанавливаем метку do_cfg исходя из наличия негативного промпта
            do_cfg = True if negative_prompt is not None else False

            prompt = prompt or ""
            prompt = [prompt] if isinstance(prompt, str) else prompt
            if prompt_2 is not None:
                prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            
            if do_cfg:
                negative_prompt = negative_prompt or ""
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                if negative_prompt_2 is not None:
                    negative_prompt_2 = [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        


        # Получаем эмбеддинги с моделей CLIP
        clip_pipeline = CLIPTextEncoderPipeline()
        clip_output = clip_pipeline(
            text_encoder.clip_encoder,
            prompt=prompt,
            prompt_2=prompt_2,
            clip_skip=clip_skip,
            lora_scale=lora_scale,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
        )



        # Получаем эмбеддинги с модели Transformer
        # if text_encoder.transformer_encoder is not None:
        #     transformer_pipeline = TransformerTextEncoderPipeline()
        #     output.t5_embeds = transformer_pipeline(
        #         text_encoder.transformer_encoder,
        #         **te_input,
        #     )



        return TextEncoderPipelineOutput(
            do_cfg=do_cfg,
            clip_embeds_1=clip_output.prompt_embeds_1,
            lora_scale = lora_scale,
            clip_embeds_2 = clip_output.prompt_embeds_2,
            pooled_clip_embeds = clip_output.pooled_prompt_embeds,
        )
    

