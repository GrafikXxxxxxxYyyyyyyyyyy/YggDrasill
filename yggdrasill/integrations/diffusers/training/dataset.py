"""Dataset helpers for diffusion training recipes."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.transforms import build_image_transform, build_mask_transform

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class CaptionSample:
    image_path: Optional[Path]
    caption: str
    image: Any = None
    prompt_2: Optional[str] = None
    init_image_path: Optional[Path] = None
    init_image: Any = None
    mask_path: Optional[Path] = None
    mask_image: Any = None
    masked_image_path: Optional[Path] = None
    masked_image: Any = None
    aesthetic_score: Optional[float] = None


def _supported_images(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


def _read_caption_txt(image_path: Path) -> Optional[str]:
    txt_path = image_path.with_suffix(".txt")
    if not txt_path.exists():
        return None
    text = txt_path.read_text(encoding="utf-8").strip()
    return text or None


def _extract_row_path(row: Dict[str, Any], image_column: str) -> str:
    for key in (image_column, "image", "image_path", "file_name", "path"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError("Manifest row does not contain an image path field")


def _extract_row_caption(row: Dict[str, Any], caption_column: str) -> str:
    for key in (caption_column, "caption", "text", "prompt"):
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    raise ValueError("Manifest row does not contain a usable caption field")


def _extract_optional_text(row: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_optional_image_ref(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _resolve_image_ref(data_dir: Path, value: Any) -> tuple[Optional[Path], Any]:
    if value is None:
        return None, None
    if isinstance(value, (str, Path)):
        image_path = Path(value)
        if not image_path.is_absolute():
            image_path = data_dir / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Manifest references missing image: {image_path}")
        return image_path, None
    return None, value


def _load_manifest_samples(
    data_dir: Path,
    *,
    image_column: str,
    caption_column: str,
    prompt_2_column: Optional[str] = None,
    init_image_column: Optional[str] = None,
    mask_column: Optional[str] = None,
    masked_image_column: Optional[str] = None,
    aesthetic_score_column: Optional[str] = None,
    prompt_2_fallback_to_caption: bool = True,
) -> List[CaptionSample]:
    jsonl_path = data_dir / "metadata.jsonl"
    csv_path = data_dir / "metadata.csv"
    rows: Iterable[Dict[str, Any]]
    if jsonl_path.exists():
        rows = (
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    elif csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        return []

    samples: List[CaptionSample] = []
    for row in rows:
        image_path, image = _resolve_image_ref(data_dir, _extract_optional_image_ref(row, image_column, "image", "file_name", "path"))
        if image_path is None and image is None:
            raise ValueError("Manifest row does not contain an image field")
        caption = _extract_row_caption(row, caption_column)
        prompt_2 = None
        if prompt_2_column:
            prompt_2 = _extract_optional_text(row, prompt_2_column, "prompt_2", "caption_2")
        if prompt_2 is None and prompt_2_fallback_to_caption:
            prompt_2 = caption

        init_image_path, init_image = _resolve_image_ref(
            data_dir,
            _extract_optional_image_ref(row, *(k for k in (init_image_column, "init_image", "source_image") if k)),
        )
        mask_path, mask_image = _resolve_image_ref(
            data_dir,
            _extract_optional_image_ref(row, *(k for k in (mask_column, "mask", "mask_image") if k)),
        )
        masked_image_path, masked_image = _resolve_image_ref(
            data_dir,
            _extract_optional_image_ref(row, *(k for k in (masked_image_column, "masked_image") if k)),
        )
        aesthetic_score = None
        if aesthetic_score_column:
            raw_score = row.get(aesthetic_score_column)
            if raw_score is not None:
                aesthetic_score = float(raw_score)
        samples.append(
            CaptionSample(
                image_path=image_path,
                caption=caption,
                image=image,
                prompt_2=prompt_2,
                init_image_path=init_image_path,
                init_image=init_image,
                mask_path=mask_path,
                mask_image=mask_image,
                masked_image_path=masked_image_path,
                masked_image=masked_image,
                aesthetic_score=aesthetic_score,
            )
        )
    return samples


def discover_caption_samples(
    data_dir: str | Path,
    *,
    image_column: str = "image",
    caption_column: str = "caption",
    prompt_2_column: Optional[str] = None,
    init_image_column: Optional[str] = None,
    mask_column: Optional[str] = None,
    masked_image_column: Optional[str] = None,
    aesthetic_score_column: Optional[str] = None,
    prompt_2_fallback_to_caption: bool = True,
) -> List[CaptionSample]:
    """Discover caption/image pairs using manifest-first, txt-sidecar fallback."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    manifest_samples = _load_manifest_samples(
        root,
        image_column=image_column,
        caption_column=caption_column,
        prompt_2_column=prompt_2_column,
        init_image_column=init_image_column,
        mask_column=mask_column,
        masked_image_column=masked_image_column,
        aesthetic_score_column=aesthetic_score_column,
        prompt_2_fallback_to_caption=prompt_2_fallback_to_caption,
    )
    if manifest_samples:
        return manifest_samples

    samples: List[CaptionSample] = []
    for image_path in _supported_images(root):
        caption = _read_caption_txt(image_path)
        if caption is None:
            raise FileNotFoundError(
                f"Missing caption for image {image_path.name}. "
                "Provide a sibling .txt file or a metadata.jsonl / metadata.csv manifest."
            )
        samples.append(CaptionSample(image_path=image_path, caption=caption, prompt_2=caption))

    if not samples:
        raise ValueError(f"No training samples found in {root}")
    return samples


def _load_hf_dataset(
    dataset_name: str,
    *,
    split: str,
    dataset_config_name: Optional[str],
) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to load Hugging Face datasets "
            "for diffusion training. Install the diffusion extra dependencies."
        ) from exc

    return load_dataset(dataset_name, name=dataset_config_name, split=split)


def _discover_hf_samples(
    dataset_name: str,
    *,
    split: str,
    dataset_config_name: Optional[str],
    image_column: str,
    caption_column: str,
    prompt_2_column: Optional[str] = None,
    init_image_column: Optional[str] = None,
    mask_column: Optional[str] = None,
    masked_image_column: Optional[str] = None,
    aesthetic_score_column: Optional[str] = None,
    prompt_2_fallback_to_caption: bool = True,
) -> List[CaptionSample]:
    dataset = _load_hf_dataset(
        dataset_name,
        split=split,
        dataset_config_name=dataset_config_name,
    )
    samples: List[CaptionSample] = []
    for row in dataset:
        image = row.get(image_column) if isinstance(row, dict) else None
        if image is None:
            for key in (image_column, "image"):
                value = row.get(key) if isinstance(row, dict) else None
                if value is not None:
                    image = value
                    break
        if image is None:
            raise ValueError(
                f"Hugging Face dataset row does not contain image column {image_column!r}"
            )
        caption = _extract_row_caption(row, caption_column)
        prompt_2 = None
        if prompt_2_column:
            prompt_2 = _extract_optional_text(row, prompt_2_column, "prompt_2", "caption_2")
        if prompt_2 is None and prompt_2_fallback_to_caption:
            prompt_2 = caption
        aesthetic_score = None
        if aesthetic_score_column:
            raw_score = row.get(aesthetic_score_column) if isinstance(row, dict) else None
            if raw_score is not None:
                aesthetic_score = float(raw_score)
        samples.append(
            CaptionSample(
                image_path=None,
                caption=caption,
                image=image,
                prompt_2=prompt_2,
                init_image=_extract_optional_image_ref(row, *(k for k in (init_image_column, "init_image", "source_image") if k)),
                mask_image=_extract_optional_image_ref(row, *(k for k in (mask_column, "mask", "mask_image") if k)),
                masked_image=_extract_optional_image_ref(row, *(k for k in (masked_image_column, "masked_image") if k)),
                aesthetic_score=aesthetic_score,
            )
        )
    if not samples:
        raise ValueError(f"No training samples found in Hugging Face dataset {dataset_name!r}")
    return samples


def _coerce_image_to_pil(image_value: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for diffusion training datasets. "
            "Install the diffusion extra dependencies."
        ) from exc

    if image_value is None:
        raise ValueError("Dataset sample does not contain an image")
    if hasattr(image_value, "convert"):
        return image_value
    if isinstance(image_value, (str, Path)):
        with Image.open(image_value) as image:
            return image.convert("RGB")
    if isinstance(image_value, dict):
        path = image_value.get("path")
        if path:
            with Image.open(path) as image:
                return image.convert("RGB")
        raw = image_value.get("bytes")
        if raw is not None:
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
    raise TypeError(f"Unsupported dataset image value type: {type(image_value)!r}")


class DiffusionTrainingDataset:
    """Recipe-aware dataset for local folders and Hugging Face datasets."""

    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:
        root = Path(config.data_dir)
        if root.exists() and root.is_dir():
            self._samples = discover_caption_samples(
                config.data_dir,
                image_column=config.image_column,
                caption_column=config.caption_column,
                prompt_2_column=config.prompt_2_column,
                init_image_column=config.init_image_column,
                mask_column=config.mask_column,
                masked_image_column=config.masked_image_column,
                aesthetic_score_column=config.aesthetic_score_column,
                prompt_2_fallback_to_caption=config.prompt_2_fallback_to_caption,
            )
            self._source_kind = "local"
        else:
            self._samples = _discover_hf_samples(
                str(config.data_dir),
                split=config.dataset_split,
                dataset_config_name=config.dataset_config_name,
                image_column=config.image_column,
                caption_column=config.caption_column,
                prompt_2_column=config.prompt_2_column,
                init_image_column=config.init_image_column,
                mask_column=config.mask_column,
                masked_image_column=config.masked_image_column,
                aesthetic_score_column=config.aesthetic_score_column,
                prompt_2_fallback_to_caption=config.prompt_2_fallback_to_caption,
            )
            self._source_kind = "hf"
        self._config = config
        self._image_transform = build_image_transform(resolution=config.resolution)
        self._mask_transform = build_mask_transform(resolution=config.resolution)

    def __len__(self) -> int:
        return len(self._samples)

    def _load_rgb(self, *, path: Optional[Path], image: Any) -> tuple[Any, Optional[str]]:
        value = image if image is not None else path
        if value is None:
            raise ValueError("Dataset sample does not contain an image")
        pil_image = _coerce_image_to_pil(value).convert("RGB")
        return self._image_transform(pil_image), str(path) if path is not None else None

    def _load_mask(self, *, path: Optional[Path], image: Any) -> tuple[Any, Optional[str]]:
        value = image if image is not None else path
        if value is None:
            raise ValueError("Dataset sample does not contain a mask image")
        pil_image = _coerce_image_to_pil(value).convert("L")
        return self._mask_transform(pil_image), str(path) if path is not None else None

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        pixel_values, image_path = self._load_rgb(path=sample.image_path, image=sample.image)
        item: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "caption": sample.caption,
            "prompt_2": sample.prompt_2 or sample.caption,
            "image_path": image_path,
            "source_kind": self._source_kind,
        }
        if self._config.task in {"img2img", "inpaint", "refiner"}:
            init_pixels, init_image_path = self._load_rgb(path=sample.init_image_path, image=sample.init_image)
            item["init_pixel_values"] = init_pixels
            item["init_image_path"] = init_image_path
        if self._config.task == "inpaint":
            mask_values, mask_path = self._load_mask(path=sample.mask_path, image=sample.mask_image)
            item["mask_values"] = mask_values
            item["mask_path"] = mask_path
            if sample.masked_image_path is not None or sample.masked_image is not None:
                masked_pixels, masked_path = self._load_rgb(path=sample.masked_image_path, image=sample.masked_image)
                item["masked_pixel_values"] = masked_pixels
                item["masked_image_path"] = masked_path
        if self._config.requires_aesthetics_score:
            item["aesthetic_score"] = (
                sample.aesthetic_score
                if sample.aesthetic_score is not None
                else self._config.aesthetic_score
            )
        return item


class FolderCaptionDataset(DiffusionTrainingDataset):
    """Backward-compatible wrapper for the original text2img dataset."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        resolution: int,
        image_column: str = "image",
        caption_column: str = "caption",
        dataset_split: str = "train",
        dataset_config_name: Optional[str] = None,
    ) -> None:
        config = TrainingConfig(
            pretrained_model_name_or_path="stub",
            data_dir=str(data_dir),
            output_path="stub.safetensors",
            resolution=resolution,
            image_column=image_column,
            caption_column=caption_column,
            dataset_split=dataset_split,
            dataset_config_name=dataset_config_name,
        )
        super().__init__(config)
