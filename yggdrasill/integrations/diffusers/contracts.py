"""Canonical port names and payload contracts for diffusion graphs.

These constants define the standard interface between diffusion nodes,
ensuring consistency across SD1.5, SDXL, and future model families.
Port names are used in graph builders; payload keys describe what flows
through the edges at runtime.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Canonical port names — prompt encoding
# ---------------------------------------------------------------------------

PORT_PROMPT = "prompt"
PORT_NEGATIVE_PROMPT = "negative_prompt"
PORT_PROMPT_2 = "prompt_2"
PORT_NEGATIVE_PROMPT_2 = "negative_prompt_2"
# Tokenizer (Converter) outputs — token_ids for Conjector
PORT_INPUT_IDS = "input_ids"
PORT_INPUT_IDS_2 = "input_ids_2"
PORT_NEGATIVE_INPUT_IDS = "negative_input_ids"
PORT_NEGATIVE_INPUT_IDS_2 = "negative_input_ids_2"
PORT_ATTENTION_MASK = "attention_mask"
PORT_ATTENTION_MASK_2 = "attention_mask_2"
PORT_PROMPT_EMBEDS = "prompt_embeds"
PORT_NEGATIVE_PROMPT_EMBEDS = "negative_prompt_embeds"
PORT_POOLED_PROMPT_EMBEDS = "pooled_prompt_embeds"
PORT_NEGATIVE_POOLED_PROMPT_EMBEDS = "negative_pooled_prompt_embeds"

# ---------------------------------------------------------------------------
# Canonical port names — latent space
# ---------------------------------------------------------------------------

PORT_LATENTS = "latents"
PORT_INIT_LATENTS = "init_latents"
PORT_NOISE = "noise"
PORT_INIT_IMAGE = "init_image"
PORT_MASK_IMAGE = "mask_image"
PORT_MASKED_IMAGE_LATENTS = "masked_image_latents"
PORT_MASK_LATENTS = "mask_latents"

# ---------------------------------------------------------------------------
# Canonical port names — denoising loop
# ---------------------------------------------------------------------------

PORT_TIMESTEP = "timestep"
PORT_TIMESTEPS = "timesteps"
PORT_NOISE_PRED = "noise_pred"
PORT_ENCODER_HIDDEN_STATES = "encoder_hidden_states"
PORT_ADDED_COND_KWARGS = "added_cond_kwargs"
PORT_DOWN_BLOCK_RESIDUALS = "down_block_additional_residuals"
PORT_MID_BLOCK_RESIDUAL = "mid_block_additional_residual"
PORT_IMAGE_EMBEDS = "image_embeds"

# ---------------------------------------------------------------------------
# Canonical port names — scheduler
# ---------------------------------------------------------------------------

PORT_SCHEDULER_STATE = "scheduler_state"
PORT_SCHEDULER_OUTPUT = "scheduler_output"
PORT_SCALED_INPUT = "scaled_input"

# ---------------------------------------------------------------------------
# Canonical port names — output
# ---------------------------------------------------------------------------

PORT_DECODED_IMAGE = "decoded_image"
PORT_OUTPUT_IMAGE = "output_image"
PORT_NSFW_DETECTED = "nsfw_content_detected"

# ---------------------------------------------------------------------------
# Canonical port names — SDXL extras
# ---------------------------------------------------------------------------

PORT_ADD_TEXT_EMBEDS = "add_text_embeds"
PORT_ADD_TIME_IDS = "add_time_ids"
PORT_NEGATIVE_ADD_TIME_IDS = "negative_add_time_ids"

# ---------------------------------------------------------------------------
# Canonical port names — adapters
# ---------------------------------------------------------------------------

PORT_CONTROL_IMAGE = "control_image"
PORT_CONTROL_RESIDUALS = "control_residuals"
PORT_IP_ADAPTER_IMAGE = "ip_adapter_image"
PORT_LORA_SCALE = "lora_scale"

# ---------------------------------------------------------------------------
# Canonical port names — FLUX extras
# ---------------------------------------------------------------------------

PORT_GUIDANCE = "guidance"
PORT_IMG_IDS = "img_ids"
PORT_TXT_IDS = "txt_ids"
PORT_PACKED_LATENTS = "packed_latents"
PORT_POOLED_PROJECTIONS = "pooled_projections"
PORT_CONTROLNET_BLOCK_SAMPLES = "controlnet_block_samples"
PORT_CONTROLNET_SINGLE_BLOCK_SAMPLES = "controlnet_single_block_samples"
PORT_MAX_SEQUENCE_LENGTH = "max_sequence_length"

# ---------------------------------------------------------------------------
# Config keys (used in node config dicts)
# ---------------------------------------------------------------------------

CFG_REPO_ID = "repo_id"
CFG_SUBFOLDER = "subfolder"
CFG_LOCAL_PATH = "local_path"
CFG_VARIANT = "variant"
CFG_REVISION = "revision"
CFG_TORCH_DTYPE = "torch_dtype"
CFG_DEVICE = "device"
CFG_SCHEDULER_CLASS = "scheduler_class"
CFG_SAFETY_CHECKER = "safety_checker"
CFG_CLIP_SKIP = "clip_skip"
CFG_IP_ADAPTER_SCALE = "ip_adapter_scale"
