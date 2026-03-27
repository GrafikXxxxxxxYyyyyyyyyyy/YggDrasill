"""Dataset helpers for the minimal diffusion training subsystem."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from yggdrasill.integrations.diffusers.training.transforms import build_image_transform

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class CaptionSample:
    image_path: Optional[Path]
    caption: str
    image: Any = None


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


def _load_manifest_samples(
    data_dir: Path,
    *,
    image_column: str,
    caption_column: str,
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
        rel_path = _extract_row_path(row, image_column)
        image_path = Path(rel_path)
        if not image_path.is_absolute():
            image_path = data_dir / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Manifest references missing image: {image_path}")
        caption = _extract_row_caption(row, caption_column)
        samples.append(CaptionSample(image_path=image_path, caption=caption))
    return samples


def discover_caption_samples(
    data_dir: str | Path,
    *,
    image_column: str = "image",
    caption_column: str = "caption",
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
        samples.append(CaptionSample(image_path=image_path, caption=caption))

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
        samples.append(CaptionSample(image_path=None, caption=caption, image=image))
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
        return image_value.convert("RGB")
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


class FolderCaptionDataset:
    """Caption dataset for local folders and Hugging Face datasets."""

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
        root = Path(data_dir)
        if root.exists() and root.is_dir():
            self._samples = discover_caption_samples(
                data_dir,
                image_column=image_column,
                caption_column=caption_column,
            )
            self._source_kind = "local"
        else:
            self._samples = _discover_hf_samples(
                str(data_dir),
                split=dataset_split,
                dataset_config_name=dataset_config_name,
                image_column=image_column,
                caption_column=caption_column,
            )
            self._source_kind = "hf"
        self._transform = build_image_transform(resolution=resolution)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        if sample.image is not None:
            image = _coerce_image_to_pil(sample.image)
            pixel_values = self._transform(image)
            image_path = None
        else:
            assert sample.image_path is not None
            pil_image = _coerce_image_to_pil(sample.image_path)
            pixel_values = self._transform(pil_image)
            image_path = str(sample.image_path)
        return {
            "pixel_values": pixel_values,
            "caption": sample.caption,
            "image_path": image_path,
            "source_kind": self._source_kind,
        }
