import torch 

from dataclasses import dataclass
from typing import List, Optional, Union
from diffusers.utils import BaseOutput

from ...stable_diffusion_model import StableDiffusionModelKey
from ...models.models.clip_te_model import CLIPTextEncoderModel


@dataclass
class CLIPTextEncoderPipelineInput(BaseOutput):
    prompt: List[str]
    num_images_per_prompt: int = 1
    clip_skip: Optional[int] = None
    lora_scale: Optional[float] = None
    prompt_2: Optional[Union[str, List[str]]] = None


@dataclass
class CLIPTextEncoderPipelineOutput(BaseOutput):
    prompt_embeds_1: torch.FloatTensor
    prompt_embeds_2: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None



class CLIPTextEncoderPipeline:
    clip_encoder: Optional[CLIPTextEncoderModel] = None

    def __init__(
        self,
        model_key: Optional[StableDiffusionModelKey] = None,
        **kwargs,
    ):
        if model_key is not None:
            self.clip_encoder = CLIPTextEncoderModel(**model_key)


    def encode_clip_prompt(
        self,
        prompt: List[str],
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt_2: Optional[List[str]] = None,
    ) -> CLIPTextEncoderPipelineOutput:  
        # Получаем выходы всех энкодеров модели
        (
            prompt_embeds_1, 
            prompt_embeds_2, 
            pooled_prompt_embeds
        ) = self.clip_encoder(
            prompt=prompt,
            prompt_2=prompt_2,
            clip_skip=clip_skip,
            lora_scale=lora_scale,
        )
        
        # Запоминаем форму тензора
        bs_embed, seq_len, _ = prompt_embeds_1.shape
        # Мультиплицируем количество промптов под заданные параметры
        prompt_embeds_1 = prompt_embeds_1.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_1 = prompt_embeds_1.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.repeat(1, num_images_per_prompt, 1)
            prompt_embeds_2 = prompt_embeds_2.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)

        
        return CLIPTextEncoderPipelineOutput(
            prompt_embeds_1=prompt_embeds_1,
            prompt_embeds_2=prompt_embeds_2,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        


    def __call__(
        self,
        input: CLIPTextEncoderPipelineInput,
        clip_encoder: Optional[CLIPTextEncoderModel] = None,
        **kwargs,
    ) -> CLIPTextEncoderPipelineOutput:  
        print("CLIPTextEncoderPipeline --->")

        if clip_encoder is not None:
            self.clip_encoder = clip_encoder
        
        return self.encode_clip_prompt(**input)
    




