"""InsightFace FaceID embedding preparation (for IP-Adapter FaceID checkpoints).

Yggdrasill's IP-Adapter supports providing precomputed embeddings directly via
``graph.run(..., ip_adapter_image_embeds=...)``. For FaceID checkpoints those embeddings are
InsightFace recognition vectors of shape ``[num_faces, 512]``.

This helper optionally runs InsightFace to extract those vectors from reference face images.

The produced tensors are *conditional-only* (no CFG-uncond half). Yggdrasill will pack them into
the CFG batch in UNet via :func:`yggdrasill.integrations.diffusers.common.ip_adapter_embeds.format_ip_adapter_image_embeds`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union


def _as_sequence(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def prepare_faceid_image_embeds(
    face_images: Union[Any, Sequence[Any]],
    *,
    insightface_app: Any = None,
    # How to select faces when multiple are detected in one image.
    # - "largest": choose the single face with the largest bbox area
    # - "first": choose the first detected face
    # - "all": return embeddings for all detected faces (up to max_faces_per_image)
    select: str = "largest",
    max_faces_per_image: int = 1,
    expected_embedding_dim: Optional[int] = 512,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
    # InsightFace detector / recognition configuration (used only when insightface_app is not provided).
    insightface_model_name: str = "buffalo_l",
    ctx_id: Optional[int] = None,
    det_size: Tuple[int, int] = (640, 640),
) -> Any:
    """Compute conditional-only FaceID embeddings for IP-Adapter FaceID.

    Args:
        face_images: One image or a sequence. Each entry may be:
            URL string, local path, PIL Image, or a numpy-like array accepted by InsightFace.
        insightface_app: Optional pre-initialized InsightFace ``FaceAnalysis`` instance.
        select: How to handle multiple detected faces per image.
        max_faces_per_image: Upper bound on faces taken from each input image.
        expected_embedding_dim: If set, validates embedding dim of extracted vectors.
        device: Torch device for the returned tensor (if torch is available).
        dtype: Torch dtype for the returned tensor.
        insightface_model_name: InsightFace recognition model name, e.g. ``"buffalo_l"``.
        ctx_id: InsightFace execution context id. If None, auto-detects (CUDA when possible).
        det_size: InsightFace face detector size.

    Returns:
        ``torch.Tensor`` with shape ``[num_faces, embedding_dim]`` (conditional-only).
    """
    import numpy as np

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch is required to return embeddings as tensors. Install with: pip install torch"
        ) from exc

    # Fast path: already precomputed.
    if torch.is_tensor(face_images):
        t = face_images
        if expected_embedding_dim is not None and int(t.shape[-1]) != int(expected_embedding_dim):
            raise ValueError(
                f"FaceID embeddings dim mismatch: expected {expected_embedding_dim}, got {int(t.shape[-1])}."
            )
        if device is not None:
            t = t.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    # Normalize to a sequence.
    imgs: List[Any] = _as_sequence(face_images)

    # Load common image types through existing helper (strings/paths/URLs).
    from yggdrasill.integrations.diffusers.common.image_utils import load_image as _load_image

    norm_imgs: List[Any] = []
    for im in imgs:
        if isinstance(im, (str, Path)):
            norm_imgs.append(_load_image(im))
        else:
            norm_imgs.append(im)

    if insightface_app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "insightface is required for prepare_faceid_image_embeds. "
                "Install with: pip install insightface"
            ) from exc

        app = FaceAnalysis(name=insightface_model_name)
        if ctx_id is None:
            ctx_id = 0 if torch.cuda.is_available() else -1
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        insightface_app = app

    embeddings: List[np.ndarray] = []

    for im in norm_imgs:
        if hasattr(im, "convert"):
            # PIL Image
            arr = np.array(im.convert("RGB"))
        elif isinstance(im, np.ndarray):
            arr = im
        else:
            # Let InsightFace decide for non-standard array-like objects.
            arr = im

        faces = insightface_app.get(arr) or []
        if not faces:
            continue

        # Select faces.
        picked: List[Any]
        if select == "all":
            picked = list(faces)[: int(max_faces_per_image)]
        elif select == "first":
            picked = [faces[0]] if faces else []
        elif select == "largest":
            def area(f: Any) -> float:
                bbox = getattr(f, "bbox", None)
                if bbox is None:
                    return 0.0
                try:
                    x1, y1, x2, y2 = bbox[:4]
                    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                except Exception:
                    return 0.0

            faces_sorted = sorted(list(faces), key=area, reverse=True)
            picked = faces_sorted[: int(max_faces_per_image)]
        else:
            raise ValueError(f"Unknown select={select!r}; expected 'largest', 'first', or 'all'.")

        for f in picked:
            emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            emb_arr = np.asarray(emb, dtype=np.float32)
            if expected_embedding_dim is not None and int(emb_arr.shape[-1]) != int(expected_embedding_dim):
                raise ValueError(
                    f"FaceID embedding dim mismatch: expected {expected_embedding_dim}, got {int(emb_arr.shape[-1])}."
                )
            embeddings.append(emb_arr)

    if not embeddings:
        raise ValueError("prepare_faceid_image_embeds: no faces found in provided images.")

    t = torch.as_tensor(np.stack(embeddings, axis=0))
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t
