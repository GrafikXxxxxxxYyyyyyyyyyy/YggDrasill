"""Image pre/post-processing utilities matching Diffusers conventions."""
from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
from PIL import Image, ImageOps


def load_mask_image(image: Any) -> Any:
    """Load a spatial IP-Adapter mask (URL, path, or PIL).

    Unlike :func:`load_image`, this does **not** force ``RGB``. Palette images with transparency
    and RGBA masks are reduced to a single ``L`` channel so :class:`diffusers.image_processor.
    IPAdapterMaskProcessor` sees correct soft/hard edges (Diffusers ``load_image`` drops palette
    transparency when converting to RGB, which corrupts common mask PNGs).
    """
    if hasattr(image, "save") and not isinstance(image, str):
        return _pil_to_mask_luminance(image)

    if not isinstance(image, str):
        return image

    if image.startswith("http://") or image.startswith("https://"):
        try:
            from diffusers.utils import DIFFUSERS_REQUEST_TIMEOUT
            import requests

            img = Image.open(
                requests.get(image, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT).raw
            )
        except ImportError:
            import requests
            from io import BytesIO

            r = requests.get(image, timeout=60)
            if not r.ok:
                raise RuntimeError(
                    f"Failed to load mask URL (HTTP {r.status_code}): {image!r}"
                ) from None
            img = Image.open(BytesIO(r.content))
    elif os.path.isfile(image):
        img = Image.open(image)
    else:
        raise ValueError(
            f"Incorrect path or URL for mask: {image!r}. "
            "URLs must start with http:// or https://."
        )

    img = ImageOps.exif_transpose(img)
    return _pil_to_mask_luminance(img)


def _pil_to_mask_luminance(img: Any) -> Any:
    """Return PIL ``L`` image: luma, respecting alpha when present."""
    if not hasattr(img, "convert"):
        return img

    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode == "LA":
        l_ch, a_ch = img.split()
        img = Image.merge("RGBA", (l_ch, l_ch, l_ch, a_ch))
    if img.mode == "RGBA":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        out = np.clip(luma * a, 0.0, 1.0)
        return Image.fromarray((out * 255.0).round().astype(np.uint8), mode="L")
    if img.mode != "L":
        img = img.convert("L")
    return img


def load_image(image: Any) -> Any:
    """Load image from URL or file path. Returns PIL Image. Pass-through for PIL/numpy/tensor."""

    def _load_one(img: Any) -> Any:
        if not isinstance(img, str):
            return img
        try:
            from diffusers.utils import load_image as _load
            from PIL import Image as PILImage

            out = _load(img)
            if isinstance(out, PILImage.Image) and (out.size[0] < 8 or out.size[1] < 8):
                raise RuntimeError(
                    f"Loaded image is too small {out.size}; URL may be invalid or return a placeholder: {img!r}"
                )
            return out
        except ImportError:
            from PIL import Image
            import requests
            from io import BytesIO
            if img.startswith(("http://", "https://")):
                r = requests.get(img, timeout=60)
                if not r.ok:
                    raise RuntimeError(
                        f"Failed to load image URL (HTTP {r.status_code}): {img!r}. "
                        "Broken or blocked URLs often produce garbage conditioning and noisy output."
                    )
                if len(r.content) < 256:
                    raise RuntimeError(
                        f"Image URL returned too little data ({len(r.content)} bytes): {img!r}. "
                        "You may have received an HTML error page instead of an image."
                    )
                return Image.open(BytesIO(r.content)).convert("RGB")
            return Image.open(img).convert("RGB")

    if isinstance(image, list):
        return [_load_one(x) for x in image]
    return _load_one(image)


def _import_torch() -> Any:
    import torch
    return torch


def apply_canny_for_controlnet_conditioning(
    image: Any,
    *,
    height: int,
    width: int,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Any:
    """Build a 3-channel Canny edge map for ``*controlnet-canny`` models.

    The Canny ControlNets (e.g. ``lllyasviel/sd-controlnet-canny``) were trained on
    Canny edges, not raw RGB photos. Feeding an unprocessed photo yields wrong
    ``controlnet_cond`` and unstable UNet residuals (often colorful noise after VAE).

    Matches the usual diffusers example: ``cv2.Canny`` on the resized RGB image, then
    stack grayscale edges to 3 channels. Uses OpenCV when available; otherwise a
    lightweight Sobel magnitude fallback (install ``opencv-python-headless`` for
    training-consistent edges).
    """
    import numpy as np
    from PIL import Image as PILImage

    image = load_image(image)
    if not isinstance(image, PILImage.Image):
        raise TypeError(
            f"Canny conditioning expects PIL/str/URL; got {type(image)}. "
            "Pass a pre-baked edge tensor yourself if using tensor inputs."
        )
    image = image.convert("RGB").resize((width, height), PILImage.Resampling.LANCZOS)
    rgb = np.array(image)
    try:
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    except ImportError:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy)
        mag /= mag.max() + 1e-8
        edges = ((mag > 0.12) * 255).astype(np.uint8)
    edges3 = np.stack([edges, edges, edges], axis=-1)
    return PILImage.fromarray(edges3)


def preprocess_image(
    image: Any,
    height: int = 512,
    width: int = 512,
    *,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
    do_normalize: bool = True,
    do_convert_rgb: bool = False,
) -> Any:
    """Convert PIL/numpy/torch image to model-ready tensor ``[B,C,H,W]``.

    By default (``do_normalize=True``) matches VAE input: values in **[-1, 1]** (see
    ``VaeImageProcessor`` defaults).

    For **ControlNet** conditioning, ``StableDiffusionControlNetPipeline`` uses a
    dedicated ``control_image_processor`` with ``do_normalize=False`` and
    ``do_convert_rgb=True`` — pixels stay in **[0, 1]** before ``.to(dtype=...)``.
    Pass ``do_normalize=False`` (and usually ``do_convert_rgb=True``) to match
    `pipeline_controlnet <https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/pipelines/controlnet/pipeline_controlnet.py>`_.

    Accepts URL strings and file paths (loaded via load_image).
    Delegates to ``diffusers.image_processor.VaeImageProcessor`` when available,
    with a pure-tensor fallback.
    """
    image = load_image(image)
    torch = _import_torch()

    try:
        from diffusers.image_processor import VaeImageProcessor

        processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_convert_rgb=do_convert_rgb,
            do_normalize=do_normalize,
        )
        tensor = processor.preprocess(image, height=height, width=width)
    except ImportError:
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            if do_convert_rgb:
                image = image.convert("RGB")
            image = image.resize((width, height))
            arr = np.array(image).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = arr[:, :, None]
            arr = arr.transpose(2, 0, 1)
            tensor = torch.from_numpy(arr).unsqueeze(0)
            if do_normalize:
                tensor = tensor * 2.0 - 1.0
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            tensor = torch.from_numpy(image).unsqueeze(0).float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            if do_normalize:
                tensor = tensor * 2.0 - 1.0
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def preprocess_mask(
    mask: Any,
    height: int = 512,
    width: int = 512,
    *,
    dtype: Optional[Any] = None,
    device: Optional[str] = None,
) -> Any:
    """Convert mask to tensor [B,1,H/8,W/8] in [0,1]."""
    torch = _import_torch()
    mask = load_image(mask)

    try:
        from diffusers.image_processor import VaeImageProcessor
        processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        tensor = processor.preprocess(mask, height=height, width=width)
    except ImportError:
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(mask, PILImage.Image):
            mask = mask.convert("L").resize((width // 8, height // 8))
            arr = np.array(mask).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        elif isinstance(mask, np.ndarray):
            tensor = torch.from_numpy(mask).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif isinstance(mask, torch.Tensor):
            tensor = mask
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def postprocess_image(
    images: Any,
    output_type: str = "pil",
) -> Any:
    """Convert model output tensor to the requested format.

    Accepts tensors in [0,1] range with shape [B,C,H,W].
    """
    torch = _import_torch()
    import numpy as np

    if output_type == "latent":
        return images

    if isinstance(images, torch.Tensor):
        images = images.clamp(0, 1).cpu().float()
        images_np = images.permute(0, 2, 3, 1).numpy()
    elif isinstance(images, np.ndarray):
        images_np = images
    else:
        return images

    if output_type == "np":
        return images_np

    if output_type == "pt":
        return torch.from_numpy(images_np).permute(0, 3, 1, 2)

    return numpy_to_pil(images_np)


def numpy_to_pil(images: Any) -> List[Any]:
    """Convert numpy [B,H,W,C] float32 in [0,1] to list of PIL images."""
    import numpy as np
    from PIL import Image as PILImage

    if images.ndim == 3:
        images = images[np.newaxis, ...]
    images = np.asarray(images, dtype=np.float64)
    images = np.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
    images = np.clip(images, 0.0, 1.0)
    images = (images * 255.0).round().astype(np.uint8)
    return [PILImage.fromarray(img) for img in images]
