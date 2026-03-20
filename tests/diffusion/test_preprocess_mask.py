"""preprocess_mask must accept URL/path strings like preprocess_image."""
from __future__ import annotations

from PIL import Image


def test_preprocess_mask_loads_string_through_load_image(monkeypatch):
    from yggdrasill.integrations.diffusers.common import image_utils as iu

    seen: list[object] = []

    def fake_load(m: object) -> Image.Image:
        seen.append(m)
        return Image.new("RGB", (128, 128), color=(255, 255, 255))

    monkeypatch.setattr(iu, "load_image", fake_load)
    t = iu.preprocess_mask("https://example.com/mask.png", height=512, width=512)
    assert seen == ["https://example.com/mask.png"]
    assert hasattr(t, "shape")
