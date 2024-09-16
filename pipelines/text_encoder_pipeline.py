import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Union, Dict, Any

from .pipelines.clip_te_pipeline import (
    CLIPTextEncoderPipeline, 
    CLIPTextEncoderPipelineInput
)
# from .pipelines.transformer_te_pipeline import (
#     TransformerTextEncoderPipeline,
#     TransformerTextEncoderPipelineInput,
# )
from ..models.text_encoder_model import TextEncoderModel


@dataclass
class TextEncoderPipelineInput(CLIPTextEncoderPipelineInput):
    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None


@dataclass
class TextEncoderPipelineOutput(BaseOutput):
    do_cfg: bool
    batch_size: int
    clip_embeds_1: torch.FloatTensor
    clip_embeds_2: Optional[torch.FloatTensor] = None
        # transformer_embeds: Optional[torch.FloatTensor] = None
    pooled_clip_embeds: Optional[torch.FloatTensor] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None



class TextEncoderPipeline:
    text_encoder: TextEncoderModel
    clip_pipeline: CLIPTextEncoderPipeline
        # transformer_pipeline: TransformerTextEncoderPipeline

    def __init__(
        self,
        **kwargs,
    ):
        self.clip_pipeline = CLIPTextEncoderPipeline()
    

    def __call__(
        self,
        text_encoder: Optional[TextEncoderModel] = None,
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        print("TextEncoderPipeline --->")
        # Устанавливаем модель TextEncoderModel
        if text_encoder is not None:
            self.text_encoder = text_encoder


        if "1. Подготавливаем необходимые аргументы":
            # Устанавливаем метку do_cfg исходя из наличия негативного промпта
            do_cfg = True if negative_prompt is not None else False

            prompt = prompt or ""
            prompt = [prompt] if isinstance(prompt, str) else prompt
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            if do_cfg:
                negative_prompt = negative_prompt or ""
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                if len(prompt) != len(negative_prompt):
                    # Если негатив промпт не совпал с обычным, тупо зануляем все негативы
                    negative_prompt = [""] * len(prompt)

                negative_prompt_2 = negative_prompt_2 or negative_prompt
                negative_prompt_2 = [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        
            batch_size = len(prompt) * num_images_per_prompt

            
        
        if "2. Получаем эмбеддинги с моделей":
            # Устанавливаем в класс правильную модель
            self.clip_pipeline.clip_encoder = self.text_encoder.clip_encoder
            # Вызываем сам класс
            clip_output = self.clip_pipeline(
                prompt=prompt,
                prompt_2=prompt_2,
                clip_skip=clip_skip,
                lora_scale=lora_scale,
                num_images_per_prompt=num_images_per_prompt,
            )
            print(clip_output.prompt_embeds_1.shape)
            print(clip_output.prompt_embeds_2.shape)
            print(clip_output.pooled_prompt_embeds.shape)
            # И применяем инструкции cfg если необходимо
            if do_cfg:
                negative_clip_output = self.clip_pipeline(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    clip_skip=clip_skip,
                    lora_scale=lora_scale,
                    num_images_per_prompt=num_images_per_prompt,
                )
                
                clip_output.prompt_embeds_1 = torch.cat([
                    negative_clip_output.prompt_embeds_1, clip_output.prompt_embeds_1
                ], dim=0)
                clip_output.prompt_embeds_2 = torch.cat([
                    negative_clip_output.prompt_embeds_2, clip_output.prompt_embeds_2
                ], dim=0)
                clip_output.pooled_prompt_embeds = torch.cat([
                    negative_clip_output.pooled_prompt_embeds, clip_output.pooled_prompt_embeds
                ], dim=0)

                print(clip_output.prompt_embeds_1.shape)
                print(clip_output.prompt_embeds_2.shape)
                print(clip_output.pooled_prompt_embeds.shape)

            # Получаем эмбеддинги с модели Transformer
            # if text_encoder.transformer_encoder is not None:
            #     transformer_pipeline = TransformerTextEncoderPipeline()
            #     output.t5_embeds = transformer_pipeline(
            #         text_encoder.transformer_encoder,
            #         **te_input,
            #     )


        # TODO: Поскольку пока трансформер модель не прикручена, то будем
        # передавать эмбеддинги только клиповских моделей сразу в аутпут 
        return TextEncoderPipelineOutput(
            do_cfg=do_cfg,
            batch_size=batch_size,
            clip_embeds_1=clip_output.prompt_embeds_1,
            clip_embeds_2 = clip_output.prompt_embeds_2,
            pooled_clip_embeds = clip_output.pooled_prompt_embeds,
            cross_attention_kwargs = (
                {"scale": lora_scale}
                if lora_scale is not None else
                None
            ),
        )
    

