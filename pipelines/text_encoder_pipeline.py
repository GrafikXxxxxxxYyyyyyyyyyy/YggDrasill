import torch 

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import List, Optional, Union, Dict, Any

from .pipelines.clip_te_pipeline import (
    CLIPTextEncoderPipeline, 
    CLIPTextEncoderPipelineInput,
)
from YggDrasill.models.text_encoder_model import TextEncoderModel



@dataclass
class TextEncoderPipelineInput(CLIPTextEncoderPipelineInput):
    prompt: Optional[Union[str, List[str]]] = None,
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
    def __init__(
        self,
        **kwargs,
    ):
        pass
    
    

    def __call__(
        self,
        text_encoder: TextEncoderModel,
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> TextEncoderPipelineOutput:
        """
        """
        # Устанавливаем метку do_cfg исходя из наличия негативного промпта
        do_cfg = True if negative_prompt is not None else False
        
        # TODO: Вытащить это на сторону CLIP и трансформер пайплайнов
        if "1. Подготавливаем необходимые аргументы":
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
            clip_pipeline = CLIPTextEncoderPipeline()

            clip_output = clip_pipeline(
                clip_encoder = text_encoder,
                prompt = prompt,
                prompt_2 = prompt_2,
                clip_skip = clip_skip,
                lora_scale = lora_scale,
                num_images_per_prompt = num_images_per_prompt,
            )
            print(clip_output.prompt_embeds_1.shape)
            print(clip_output.prompt_embeds_2.shape)
            print(clip_output.pooled_prompt_embeds.shape)

            # И применяем инструкции cfg если необходимо
            if do_cfg:
                negative_clip_output = clip_pipeline(
                    clip_encoder = text_encoder,
                    prompt = negative_prompt,
                    prompt_2 = negative_prompt_2,
                    clip_skip = clip_skip,
                    lora_scale = lora_scale,
                    num_images_per_prompt = num_images_per_prompt,
                )
                
                clip_output.prompt_embeds_1 = torch.cat([negative_clip_output.prompt_embeds_1, clip_output.prompt_embeds_1], dim=0)
                clip_output.prompt_embeds_2 = torch.cat([negative_clip_output.prompt_embeds_2, clip_output.prompt_embeds_2], dim=0)
                clip_output.pooled_prompt_embeds = torch.cat([negative_clip_output.pooled_prompt_embeds, clip_output.pooled_prompt_embeds], dim=0)

                print(clip_output.prompt_embeds_1.shape)
                print(clip_output.prompt_embeds_2.shape)
                print(clip_output.pooled_prompt_embeds.shape)


            # TODO: Получаем эмбеддинги с модели Transformer
            # <...>

        cross_attention_kwargs = (
            {"scale": lora_scale}
            if lora_scale is not None else
            None
        )

        return TextEncoderPipelineOutput(
            do_cfg=do_cfg,
            batch_size=batch_size,
            clip_embeds_1=clip_output.prompt_embeds_1,
            clip_embeds_2=clip_output.prompt_embeds_2,
            pooled_clip_embeds=clip_output.pooled_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs
        )

    

