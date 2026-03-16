"""Tests for the Diffusers model store."""
from __future__ import annotations


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
