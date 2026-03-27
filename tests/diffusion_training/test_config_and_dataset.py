from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.dataset import FolderCaptionDataset, discover_caption_samples


def test_training_config_builds_final_output_path(tmp_path) -> None:
    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    assert config.final_output_path == tmp_path / "out" / "pytorch_lora_weights.safetensors"


def test_training_config_rejects_vae_training(tmp_path) -> None:
    with pytest.raises(NotImplementedError):
        TrainingConfig(
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            train_vae=True,
        )


def test_discover_caption_samples_from_sidecars(tmp_path) -> None:
    PIL = pytest.importorskip("PIL.Image")
    image = PIL.new("RGB", (8, 8), color="white")
    image_path = tmp_path / "sample.png"
    image.save(image_path)
    (tmp_path / "sample.txt").write_text("astronaut", encoding="utf-8")

    samples = discover_caption_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].image_path == image_path
    assert samples[0].caption == "astronaut"


def test_discover_caption_samples_from_manifest(tmp_path) -> None:
    PIL = pytest.importorskip("PIL.Image")
    image = PIL.new("RGB", (8, 8), color="black")
    image_path = tmp_path / "sample.png"
    image.save(image_path)
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "sample.png", "caption": "castle"}) + "\n",
        encoding="utf-8",
    )

    samples = discover_caption_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].caption == "castle"


def test_folder_caption_dataset_returns_tensor_and_caption(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    PIL = pytest.importorskip("PIL.Image")
    image = PIL.new("RGB", (16, 16), color="red")
    image.save(tmp_path / "item.png")
    (tmp_path / "item.txt").write_text("red square", encoding="utf-8")

    dataset = FolderCaptionDataset(tmp_path, resolution=16)
    item = dataset[0]
    assert item["caption"] == "red square"
    assert tuple(item["pixel_values"].shape) == (3, 16, 16)


def test_training_config_supports_hf_dataset_options(tmp_path) -> None:
    config = TrainingConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        data_dir="lambdalabs/pokemon-blip-captions",
        output_path=str(tmp_path / "adapter.safetensors"),
        dataset_split="train",
        dataset_config_name=None,
    )
    assert config.dataset_split == "train"
    assert config.dataset_config_name is None


def test_folder_caption_dataset_supports_huggingface_dataset(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    PIL = pytest.importorskip("PIL.Image")

    fake_image = PIL.new("RGB", (16, 16), color="green")

    def _fake_load_dataset(dataset_name, name=None, split=None):
        assert dataset_name == "org/example-dataset"
        assert split == "train"
        assert name is None
        return [{"image": fake_image, "caption": "green square"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=_fake_load_dataset))

    dataset = FolderCaptionDataset(
        "org/example-dataset",
        resolution=16,
        dataset_split="train",
    )
    item = dataset[0]
    assert item["caption"] == "green square"
    assert item["image_path"] is None
    assert item["source_kind"] == "hf"
