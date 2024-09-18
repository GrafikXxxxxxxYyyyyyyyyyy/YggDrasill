import torch

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from typing import List, Optional, Union
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers



class CLIPTextEncoderModel:
    tokenizer_1: CLIPTokenizer
    text_encoder_1: Union[
        CLIPTextModel,
        CLIPTextModelWithProjection
    ]
    tokenizer_2: Optional[CLIPTokenizer] = None
    text_encoder_2: Optional[CLIPTextModelWithProjection] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> None:  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        model_type = model_type or "sd15"

        if model_type == "sd15":
            if hasattr(self, "tokenizer_2"):
                delattr(self, "tokenizer_2")
            if hasattr(self, "text_encoder_2"):
                delattr(self, "text_encoder_2")
            
            # инитим нужные
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(
                model_path, 
                subfolder="tokenizer",
            )
            self.text_encoder_1 = CLIPTextModel.from_pretrained(
                model_path, 
                subfolder="text_encoder", 
                torch_dtype=dtype,
                variant='fp16',
                use_safetensors=True
            )
        elif model_type == "sdxl":
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(
                model_path, 
                subfolder="tokenizer"
            )
            self.text_encoder_1 = CLIPTextModel.from_pretrained(
                model_path, 
                subfolder="text_encoder", 
                torch_dtype=dtype,
                variant='fp16',
                use_safetensors=True
            )
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder='tokenizer_2'
            )
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                model_path,
                subfolder='text_encoder_2', 
                torch_dtype=dtype,
                variant='fp16',
                use_safetensors=True
            )
        elif model_type == "sd3":
            pass
        else:
            raise ValueError(f"Unknown model_type '{self.type}'")   
        self.to(device)

        self.model_type = model_type

        print(f"TextEncoder model has successfully loaded from '{model_path}' checkpoint!")


    @property
    def device(self):
        return self.text_encoder_1.device

    @property
    def projection_dim(self):
        return (
            self.text_encoder_2.config.projection_dim 
            if hasattr(self, "text_encoder_2") else
            None
        )

    def to(self, device, dtype=None):
        self.text_encoder_1 = self.text_encoder_1.to(device, dtype=dtype)
        if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
            self.text_encoder_2 = self.text_encoder_2.to(device, dtype=dtype)
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #



    # Вынесем в отдельный метод, который будет подтягиваться в методе call
    def get_clip_embeddings(
        self,
        prompt: List[str],
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt_2: Optional[List[str]] = None,
        **kwargs,
    ):
        if lora_scale is not None:
            scale_lora_layers(self.text_encoder_1, lora_scale) 
            if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
                scale_lora_layers(self.text_encoder_2, lora_scale) 
            

        text_input_ids = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids = text_input_ids.to(self.device)
        
        encoder_output = self.text_encoder_1(
            text_input_ids, 
            output_hidden_states=True
        )

        if self.model_type == "sd15":
            prompt_embeds_1 = (
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm layer.
                self.text_encoder_1.text_model.final_layer_norm(encoder_output[-1][-(clip_skip + 1)])
                if clip_skip is not None else
                encoder_output[0]
            )
        else:
            prompt_embeds_1 = (
                encoder_output.hidden_states[-(clip_skip + 2)]
                if clip_skip is not None else
                encoder_output.hidden_states[-2]
            )


        prompt_embeds_2: Optional[torch.FloatTensor] = None
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None
        if (
            prompt_2 is not None
            and hasattr(self, "tokenizer_2")    
            and self.tokenizer_2 is not None
            and hasattr(self, "text_encoder_2")
            and self.text_encoder_2 is not None
        ):
            text_input_ids = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            text_input_ids = text_input_ids.to(self.device)

            encoder_output = self.text_encoder_2(
                text_input_ids, 
                output_hidden_states=True
            )

            prompt_embeds_2 = (
                encoder_output.hidden_states[-(clip_skip + 2)]
                if clip_skip is not None else
                encoder_output.hidden_states[-2]
            )                
            pooled_prompt_embeds = encoder_output[0]


        if lora_scale is not None:
            unscale_lora_layers(self.text_encoder_1, lora_scale)
            if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
                unscale_lora_layers(self.text_encoder_2, lora_scale) 
        

        return prompt_embeds_1, prompt_embeds_2, pooled_prompt_embeds



    # ================================================================================================================ #
    def __call__(
        self,
        prompt: List[str],
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt_2: Optional[List[str]] = None,
        **kwargs,
    ):
    # ================================================================================================================ #
        """
        Вызов модели возвращает закодированные представления со всех 
        используемых текстовых моделей
        """
        print("CLIPTextEncoderModel --->")

        (
            prompt_embeds_1, 
            prompt_embeds_2, 
            pooled_prompt_embeds
        ) = self.get_clip_embeddings(
            prompt=prompt,
            prompt_2=prompt_2,
            clip_skip=clip_skip,
            lora_scale=lora_scale,
            **kwargs,
        )        

        return prompt_embeds_1, prompt_embeds_2, pooled_prompt_embeds
    # ================================================================================================================ #