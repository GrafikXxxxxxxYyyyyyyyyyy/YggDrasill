"""Single-step SDXL refiner LoRA training objective."""
from __future__ import annotations

from yggdrasill.integrations.diffusers.training.sdxl_img2img_objective import SDXLImg2ImgLoRAObjective


class SDXLRefinerLoRAObjective(SDXLImg2ImgLoRAObjective):
    """Refiner training reuses the SDXL img2img objective with aesthetic conditioning."""
