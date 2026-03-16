"""Tests for diffusion contracts and types."""
from __future__ import annotations


from yggdrasill.integrations.diffusers import contracts as C
from yggdrasill.integrations.diffusers.types import (
    DiffusionTask,
    GenerationParams,
    GuidanceConfig,
    ImageSize,
    ModelDType,
    ModelRef,
    ModelVariant,
    OutputType,
    SchedulerConfig,
    SDXLMicroConditioning,
)


class TestContractConstants:
    """Verify canonical port names are strings and unique."""

    def test_all_port_names_are_strings(self):
        port_attrs = [
            attr for attr in dir(C)
            if attr.startswith("PORT_") or attr.startswith("CFG_")
        ]
        for attr in port_attrs:
            val = getattr(C, attr)
            assert isinstance(val, str), f"{attr} is not a string"

    def test_port_names_unique(self):
        port_names = [
            getattr(C, attr) for attr in dir(C) if attr.startswith("PORT_")
        ]
        assert len(port_names) == len(set(port_names)), "Duplicate port names found"

    def test_core_ports_exist(self):
        assert C.PORT_PROMPT == "prompt"
        assert C.PORT_LATENTS == "latents"
        assert C.PORT_INIT_LATENTS == "init_latents"
        assert C.PORT_NOISE_PRED == "noise_pred"
        assert C.PORT_DECODED_IMAGE == "decoded_image"

    def test_sdxl_ports_exist(self):
        assert C.PORT_PROMPT_2 == "prompt_2"
        assert C.PORT_ADD_TEXT_EMBEDS == "add_text_embeds"
        assert C.PORT_ADD_TIME_IDS == "add_time_ids"
        assert C.PORT_POOLED_PROMPT_EMBEDS == "pooled_prompt_embeds"

    def test_adapter_ports_exist(self):
        assert C.PORT_CONTROL_IMAGE == "control_image"
        assert C.PORT_IP_ADAPTER_IMAGE == "ip_adapter_image"
        assert C.PORT_LORA_SCALE == "lora_scale"

    def test_config_keys_exist(self):
        assert C.CFG_REPO_ID == "repo_id"
        assert C.CFG_TORCH_DTYPE == "torch_dtype"
        assert C.CFG_DEVICE == "device"


class TestDiffusionTypes:

    def test_diffusion_task_enum(self):
        assert DiffusionTask.TEXT2IMG.value == "text2img"
        assert DiffusionTask.IMG2IMG.value == "img2img"
        assert DiffusionTask.INPAINT.value == "inpaint"
        assert DiffusionTask.UPSCALE.value == "upscale"

    def test_output_type_enum(self):
        assert OutputType.PIL.value == "pil"
        assert OutputType.LATENT.value == "latent"
        assert OutputType.NUMPY.value == "np"
        assert OutputType.TORCH.value == "pt"

    def test_model_dtype_enum(self):
        assert ModelDType.FP32.value == "fp32"
        assert ModelDType.FP16.value == "fp16"
        assert ModelDType.BF16.value == "bf16"

    def test_model_variant(self):
        assert ModelVariant.DEFAULT.value == ""
        assert ModelVariant.FP16.value == "fp16"

    def test_model_ref_defaults(self):
        ref = ModelRef()
        assert ref.repo_id == ""
        assert not ref.is_local

    def test_model_ref_local(self):
        ref = ModelRef(local_path="/models/sd15")
        assert ref.is_local

    def test_scheduler_config_defaults(self):
        sc = SchedulerConfig()
        assert sc.num_inference_steps == 50
        assert sc.strength == 1.0
        assert sc.denoising_start is None

    def test_guidance_config_auto_cfg(self):
        gc = GuidanceConfig(guidance_scale=7.5)
        assert gc.do_classifier_free_guidance is True

        gc2 = GuidanceConfig(guidance_scale=1.0)
        assert gc2.do_classifier_free_guidance is False

    def test_image_size_latent_dims(self):
        size = ImageSize(512, 512)
        assert size.latent_height == 64
        assert size.latent_width == 64

        size_xl = ImageSize(1024, 1024)
        assert size_xl.latent_height == 128
        assert size_xl.latent_width == 128

    def test_sdxl_micro_conditioning_defaults(self):
        mc = SDXLMicroConditioning()
        assert mc.original_size == (1024, 1024)
        assert mc.target_size == (1024, 1024)
        assert mc.crops_coords_top_left == (0, 0)
        assert mc.aesthetic_score == 6.0

    def test_generation_params_defaults(self):
        gp = GenerationParams()
        assert gp.image_size.height == 512
        assert gp.scheduler.num_inference_steps == 50
        assert gp.guidance.guidance_scale == 7.5
        assert gp.output_type == OutputType.PIL
        assert gp.seed is None
        assert gp.num_images_per_prompt == 1
