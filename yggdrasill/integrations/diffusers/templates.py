"""High-level named templates for common diffusion graphs and workflows."""
from __future__ import annotations

from typing import Any, Callable, Dict, Final, Tuple


def _build_sd15_text2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sd15_pipeline
    return build_sd15_pipeline(task="text2img", **kwargs)


def _build_sd15_img2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sd15_pipeline
    return build_sd15_pipeline(task="img2img", **kwargs)


def _build_sd15_inpaint(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sd15_pipeline
    return build_sd15_pipeline(task="inpaint", **kwargs)


def _build_sd15_animatediff_text2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sd15_animatediff_pipeline
    return build_sd15_animatediff_pipeline(task="text2img", **kwargs)


def _build_sd15_animatediff_img2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sd15_animatediff_pipeline
    return build_sd15_animatediff_pipeline(task="img2img", **kwargs)


def _build_sd15_animatediff_inpaint(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sd15_animatediff_pipeline
    return build_sd15_animatediff_pipeline(task="inpaint", **kwargs)


def _build_sd15_animatediff_video2video(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sd15_animatediff_pipeline
    return build_sd15_animatediff_pipeline(task="video2video", **kwargs)


def _build_sd15_animatediff_sparsectrl(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sd15_animatediff_pipeline
    return build_sd15_animatediff_pipeline(task="sparsectrl", **kwargs)


def _build_sdxl_animatediff_text2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sdxl_animatediff_pipeline
    return build_sdxl_animatediff_pipeline(**kwargs)


def _build_sdxl_text2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sdxl_pipeline
    return build_sdxl_pipeline(task="text2img", **kwargs)


def _build_sdxl_img2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sdxl_pipeline
    return build_sdxl_pipeline(task="img2img", **kwargs)


def _build_sdxl_inpaint(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sdxl_pipeline
    return build_sdxl_pipeline(task="inpaint", **kwargs)


def _build_sdxl_base_refiner(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_sdxl_base_refiner
    return build_sdxl_base_refiner(**kwargs)


def _build_flux_text2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_flux_pipeline
    return build_flux_pipeline(task="text2img", **kwargs)


def _build_flux_img2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_flux_pipeline
    return build_flux_pipeline(task="img2img", **kwargs)


def _build_flux_inpaint(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_flux_pipeline
    return build_flux_pipeline(task="inpaint", **kwargs)


def _build_flux_controlnet_text2img(**kwargs: Any) -> Any:
    from yggdrasill.integrations.diffusers.factory import build_flux_pipeline
    return build_flux_pipeline(task="controlnet_text2img", **kwargs)


_TEMPLATE_BUILDERS: Dict[str, Callable[..., Any]] = {
    "sd15_text2img": _build_sd15_text2img,
    "sd15_text2image": _build_sd15_text2img,
    "sd15_img2img": _build_sd15_img2img,
    "sd15_inpaint": _build_sd15_inpaint,
    "sd15_animatediff_text2img": _build_sd15_animatediff_text2img,
    "sd15_animatediff_img2img": _build_sd15_animatediff_img2img,
    "sd15_animatediff_inpaint": _build_sd15_animatediff_inpaint,
    "sd15_animatediff_video2video": _build_sd15_animatediff_video2video,
    "sd15_animatediff_sparsectrl": _build_sd15_animatediff_sparsectrl,
    "sdxl_animatediff_text2img": _build_sdxl_animatediff_text2img,
    "sdxl_text2img": _build_sdxl_text2img,
    "sdxl_img2img": _build_sdxl_img2img,
    "sdxl_inpaint": _build_sdxl_inpaint,
    "sdxl_base_refiner": _build_sdxl_base_refiner,
    "flux_text2img": _build_flux_text2img,
    "flux_img2img": _build_flux_img2img,
    "flux_inpaint": _build_flux_inpaint,
    "flux_controlnet_text2img": _build_flux_controlnet_text2img,
}


GRAPH_TEMPLATES: Final[Tuple[str, ...]] = (
    "sd15_text2image",
    "sd15_text2img",
    "sd15_img2img",
    "sd15_inpaint",
    "sd15_animatediff_text2img",
    "sd15_animatediff_img2img",
    "sd15_animatediff_inpaint",
    "sd15_animatediff_video2video",
    "sd15_animatediff_sparsectrl",
    "sdxl_animatediff_text2img",
    "sdxl_text2img",
    "sdxl_img2img",
    "sdxl_inpaint",
    "flux_text2img",
    "flux_img2img",
    "flux_inpaint",
    "flux_controlnet_text2img",
)

WORKFLOW_TEMPLATES: Final[Tuple[str, ...]] = ("sdxl_base_refiner",)


def list_templates() -> Tuple[str, ...]:
    """Return all registered template names."""
    return tuple(sorted(_TEMPLATE_BUILDERS.keys()))


def build_template(template_name: str, **kwargs: Any) -> Any:
    """Build a structure from a short template name."""
    key = template_name.strip().lower()
    builder = _TEMPLATE_BUILDERS.get(key)
    if builder is None:
        raise KeyError(
            f"Unknown template '{template_name}'. "
            f"Available templates: {list_templates()}"
        )
    return builder(**kwargs)


# Alias for from_template (plan compatibility)
def from_template(template_name: str, **kwargs: Any) -> Any:
    """Build a diffusion graph from a template name. Same as build_template."""
    return build_template(template_name, **kwargs)
