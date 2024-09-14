import torch 

from dataclasses import dataclass
from typing import List, Optional, Union
from diffusers.utils import BaseOutput

from ...models.models.clip_te_model import CLIPTextEncoderModel


@dataclass
class CLIPTextEncoderPipelineInput(BaseOutput):
    prompt: List[str]
    clip_skip: Optional[int] = None
    lora_scale: Optional[float] = None
    prompt_2: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None


@dataclass
class CLIPTextEncoderPipelineOutput(BaseOutput):
    prompt_embeds_1: torch.FloatTensor
    prompt_embeds_2: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None



class CLIPTextEncoderPipeline:
    def __call__(
        self,
        clip_encoder: CLIPTextEncoderModel,
        prompt: List[str],
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt_2: Optional[List[str]] = None,
        negative_prompt: Optional[List[str]] = None,
        negative_prompt_2: Optional[List[str]] = None,
        **kwargs,
    ) -> CLIPTextEncoderPipelineOutput:  
        print("CLIPTextEncoderPipeline --->")


        do_cfg = True if negative_prompt is not None else False

        # Получаем выходы всех энкодеров модели
        (
            prompt_embeds_1, 
            prompt_embeds_2, 
            pooled_prompt_embeds
        ) = clip_encoder(
            prompt=prompt,
            prompt_2=prompt_2,
            clip_skip=clip_skip,
            lora_scale=lora_scale,
        )
        
        # И применяем инструкции cfg если необходимо
        if do_cfg:
            (
                negative_prompt_embeds_1, 
                negative_prompt_embeds_2, 
                negative_pooled_prompt_embeds
            ) = clip_encoder(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                clip_skip=clip_skip,
                lora_scale=lora_scale,
            )
            prompt_embeds_1 = torch.cat([negative_prompt_embeds_1, prompt_embeds_1], dim=0)
            prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)


        return CLIPTextEncoderPipelineOutput(
            prompt_embeds_1=prompt_embeds_1,
            prompt_embeds_2=prompt_embeds_2,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )        
    




