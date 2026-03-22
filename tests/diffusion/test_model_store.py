"""Tests for the Diffusers model store."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from yggdrasill.integrations.diffusers.model_store import ModelStore
from yggdrasill.integrations.diffusers.types import ModelDType


class TestModelStoreBasics:

    def setup_method(self):
        ModelStore.reset()

    def teardown_method(self):
        ModelStore.reset()

    def test_singleton_default(self):
        s1 = ModelStore.default()
        s2 = ModelStore.default()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        s1 = ModelStore.default()
        ModelStore.reset()
        s2 = ModelStore.default()
        assert s1 is not s2

    def test_cache_key(self):
        store = ModelStore()
        key = store.cache_key("repo/model", "unet", "UNet2DConditionModel")
        assert key == ("repo/model", "unet", "UNet2DConditionModel")

    def test_put_and_get(self):
        store = ModelStore()
        key = store.cache_key("repo", "vae", "AutoencoderKL")
        store.put(key, "fake_component")
        assert store.get(key) == "fake_component"
        assert len(store) == 1

    def test_get_missing_returns_none(self):
        store = ModelStore()
        assert store.get(("missing", "", "")) is None

    def test_clear(self):
        store = ModelStore()
        store.put(("a", "b", "c"), "val")
        assert len(store) == 1
        store.clear()
        assert len(store) == 0

    def test_device_property(self):
        store = ModelStore()
        assert store.device == "cpu"
        store.device = "cuda:0"
        assert store.device == "cuda:0"

    def test_dtype_property(self):
        store = ModelStore()
        assert store.dtype is None
        store.dtype = ModelDType.FP16
        assert store.dtype == ModelDType.FP16

    def test_get_torch_dtype_none(self):
        store = ModelStore()
        assert store.get_torch_dtype() is None

    def test_load_components_by_keys_sdxl_defaults_fp16_variant(self):
        """SDXL loads with torch.float16 and variant fp16 when args omitted (Hub parity)."""
        store = ModelStore()
        with patch.object(store, "load_component_by_key", return_value=object()) as m:
            store.load_components_by_keys(
                "sdxl",
                ["unet"],
                "stabilityai/stable-diffusion-xl-base-1.0",
            )
        kw = m.call_args.kwargs
        assert kw["variant"] == "fp16"
        assert kw["torch_dtype"] == torch.float16

    def test_build_sdxl_pipeline_uses_explicit_store_when_cache_empty(self):
        """``ModelStore`` defines ``__len__`` → empty store is falsy; factory must not replace it."""
        from yggdrasill.integrations.diffusers.factory import build_sdxl_pipeline

        store = ModelStore()
        assert len(store) == 0

        def fake_comp():
            m = MagicMock()
            m.to = MagicMock(return_value=m)
            return m

        with patch.object(store, "load_component_by_key", side_effect=lambda *a, **k: fake_comp()) as m:
            build_sdxl_pipeline(task="img2img", device="cpu", store=store)

        assert len(m.call_args_list) == 7
        for call in m.call_args_list:
            assert call.kwargs.get("torch_dtype") == torch.float16
            assert call.kwargs.get("variant") == "fp16"

    def test_load_components_by_keys_flux_defaults_bfloat16(self):
        store = ModelStore()
        with patch.object(store, "load_component_by_key", return_value=object()) as m:
            store.load_components_by_keys(
                "flux",
                ["transformer"],
                "black-forest-labs/FLUX.1-dev",
            )
        kw = m.call_args.kwargs
        assert kw["variant"] == ""
        assert kw["torch_dtype"] == torch.bfloat16

    def test_load_component_retries_without_variant_when_fp16_weights_missing(self):
        """Hub repos that only ship diffusion_pytorch_model.safetensors (no .fp16.)."""
        store = ModelStore()
        calls: list = []

        class FakeModel:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                calls.append(dict(kwargs))
                if kwargs.get("variant") == "fp16":
                    raise OSError(
                        "x does not appear to have a file named diffusion_pytorch_model.fp16.safetensors."
                    )
                return "ok"

        out = store.load_component(
            FakeModel,
            "xinsir/controlnet-scribble-sdxl-1.0",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        assert out == "ok"
        assert len(calls) >= 2
        assert "variant" not in calls[-1]

    def test_move_to_device_with_to(self):
        store = ModelStore()
        store.device = "cuda"

        class FakeModel:
            moved_to = None
            def to(self, device):
                self.moved_to = device
                return self

        model = FakeModel()
        store.move_to_device(model)
        assert model.moved_to == "cuda"

    def test_move_to_device_without_to(self):
        store = ModelStore()
        result = store.move_to_device(42)
        assert result == 42


class TestModelStoreConfig:
    """Test integration config classes."""

    def test_sd15_pipeline_config(self):
        from yggdrasill.integrations.diffusers.config import SD15PipelineConfig
        cfg = SD15PipelineConfig()
        assert cfg.device == "cuda"
        assert cfg.torch_dtype == "fp16"
        assert cfg.enable_safety_checker is True

    def test_sdxl_pipeline_config(self):
        from yggdrasill.integrations.diffusers.config import SDXLPipelineConfig
        cfg = SDXLPipelineConfig()
        assert cfg.device == "cuda"
        assert cfg.force_zeros_for_empty_prompt is True
        assert cfg.refiner is None

    def test_component_config_source(self):
        from yggdrasill.integrations.diffusers.config import DiffusersComponentConfig
        cfg = DiffusersComponentConfig(repo_id="stabilityai/sdxl")
        assert cfg.source == "stabilityai/sdxl"
        assert cfg.dtype_enum == ModelDType.FP16

    def test_component_config_local(self):
        from yggdrasill.integrations.diffusers.config import DiffusersComponentConfig
        cfg = DiffusersComponentConfig(local_path="/models/sd15")
        assert cfg.source == "/models/sd15"
