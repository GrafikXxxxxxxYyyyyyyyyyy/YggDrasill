import torch

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from typing import List, Optional, Union
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers



class CLIPModel:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, 
            subfolder="tokenizer",
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, 
            subfolder="text_encoder", 
            torch_dtype=dtype,
            variant='fp16',
            use_safetensors=True
        )
        self.text_encoder.to(device=device, dtype=dtype)

        self.model_path = model_path
        self.model_type = model_type or "sd15"


    def __call__(
        self,
        prompt: List[str],
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        """
        
        """
        if lora_scale is not None:
            scale_lora_layers(self.text_encoder, lora_scale) 
            

        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids = text_input_ids.to(self.text_encoder.device)
        
        encoder_output = self.text_encoder(
            text_input_ids, 
            output_hidden_states=True
        )

        if self.model_type == "sd15":
            prompt_embeds = (
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm layer.
                self.text_encoder.text_model.final_layer_norm(encoder_output[-1][-(clip_skip + 1)])
                if clip_skip is not None else
                encoder_output[0]
            )
        else:
            prompt_embeds = (
                encoder_output.hidden_states[-(clip_skip + 2)]
                if clip_skip is not None else
                encoder_output.hidden_states[-2]
            )


        if lora_scale is not None:
            unscale_lora_layers(self.text_encoder, lora_scale)


        return prompt_embeds







class CLIPModelWithProjection(CLIPModel):
    text_encoder: CLIPTextModelWithProjection

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, 
            subfolder="tokenizer",
        )

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            subfolder='text_encoder_2', 
            torch_dtype=dtype,
            variant='fp16',
            use_safetensors=True
        )

        self.text_encoder.to(device=device, dtype=dtype)

        self.model_path = model_path
        self.model_type = model_type or "sd15"

    
    def __call__(
        self,
        prompt: List[str],
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        """
        """
        if lora_scale is not None:
            scale_lora_layers(self.text_encoder, lora_scale) 


        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_input_ids = text_input_ids.to(self.text_encoder.device)

        encoder_output = self.text_encoder(
            text_input_ids, 
            output_hidden_states=True
        )

        prompt_embeds = (
            encoder_output.hidden_states[-(clip_skip + 2)]
            if clip_skip is not None else
            encoder_output.hidden_states[-2]
        )                
        pooled_prompt_embeds = encoder_output[0]


        if lora_scale is not None:
            unscale_lora_layers(self.text_encoder, lora_scale)


        return prompt_embeds, pooled_prompt_embeds
    





class CLIPTextEncoderModel:
    clip_encoder_1: CLIPModel
    clip_encoder_2: Optional[CLIPModelWithProjection] = None

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):  
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
        model_type = model_type or "sd15"

        if model_type == "sd15":
            self.clip_encoder_1 = CLIPModel(
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
            )
        
        elif model_type == "sdxl":
            self.clip_encoder_1 = CLIPModel(
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
            )

            self.clip_encoder_2 = CLIPModelWithProjection(
                dtype=dtype,
                device=device,
                model_path=model_path,
                model_type=model_type,
            )
        
        elif model_type == "sd3":
            pass

        elif model_type == "flux":
            pass

        else:
            raise ValueError(f"Unknown model_type '{model_type}'")   

        # Инициализируем константы
        self.dtype = dtype
        self.model_type = model_type 
        self.model_path = model_path
        self.device = torch.device(device)

        print(f"CLIPTextEncoderModel model has successfully loaded from '{model_path}' checkpoint!")

    @property
    def projection_dim(self):
        return (
            self.clip_encoder_2.text_encoder.config.projection_dim 
            if self.clip_encoder_2 is not None else
            None
        )
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    


    def get_clip_embeddings(
        self,
        prompt: List[str],
        clip_skip: Optional[int] = None,
        lora_scale: Optional[float] = None,
        prompt_2: Optional[List[str]] = None,
        **kwargs,
    ):  
        # Энкодим первый промпт первой моделью
        prompt_embeds_1 = self.clip_encoder_1(
            prompt=prompt,
            clip_skip=clip_skip,
            lora_scale=lora_scale,
        )
        output = (prompt_embeds_1, )

        # И второй если есть, второй моделью 
        if prompt_2 is not None:
            prompt_embeds_2, pooled_prompt_embeds = self.clip_encoder_2(
                prompt=prompt_2,
                clip_skip=clip_skip,
                lora_scale=lora_scale,
            )
            output += (prompt_embeds_2, pooled_prompt_embeds)
        
        return output
    


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
        return self.get_clip_embeddings()
    # ================================================================================================================ #




