"""Shared mock components for diffusion tests.

All models are replaced with lightweight mocks that return correctly-shaped
tensors, allowing full graph structure and wiring tests without GPU or
real model downloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

import importlib.util
torch_available = importlib.util.find_spec("torch") is not None

requires_torch = pytest.mark.skipif(not torch_available, reason="torch not installed")


@dataclass
class FakeLatentDist:
    sample_tensor: Any

    def sample(self) -> Any:
        return self.sample_tensor


@dataclass
class FakeVAEOutput:
    latent_dist: Any


class FakeTensor:
    """Minimal tensor mock that supports basic ops needed by nodes."""

    def __init__(self, shape: tuple, value: float = 0.0):
        self._shape = shape
        self._value = value

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return "cpu"

    @property
    def dtype(self):
        return "float32"

    @property
    def ndim(self):
        return len(self._shape)

    def to(self, *args, **kwargs):
        return self

    def clamp(self, *args, **kwargs):
        return self

    def item(self):
        return float(self._value)

    def flatten(self):
        return [float(self._value)]

    def cpu(self):
        return self

    def float(self):
        return self

    def chunk(self, n):
        return [FakeTensor(self._shape, self._value) for _ in range(n)]

    def unsqueeze(self, dim):
        new_shape = list(self._shape)
        new_shape.insert(dim if dim >= 0 else len(new_shape) + 1 + dim, 1)
        return FakeTensor(tuple(new_shape), self._value)

    def expand(self, *args):
        return FakeTensor(args if len(args) > 1 else args[0], self._value)

    def repeat_interleave(self, repeats, dim=0):
        r = int(repeats)
        shape = list(self._shape)
        if not shape:
            return self
        dim = int(dim)
        if dim < 0:
            dim += len(shape)
        shape[dim] = shape[dim] * r
        return FakeTensor(tuple(shape), self._value)

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __len__(self):
        return self._shape[0] if self._shape else 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self._shape[0]
            new_len = max(0, stop - start)
            return FakeTensor((new_len,) + self._shape[1:], self._value)
        return FakeTensor(self._shape[1:], self._value)


class FakeTokenizer:
    """Mock CLIPTokenizer."""
    model_max_length = 77

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=FakeTensor((1, 77)))


class FakeTextEncoder:
    """Mock CLIPTextModel."""
    device = "cpu"

    def __call__(self, input_ids, output_hidden_states=False, **kwargs):
        bs = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
        hidden = FakeTensor((bs, 77, 768))
        pooled = FakeTensor((bs, 768))
        if output_hidden_states:
            return SimpleNamespace(
                hidden_states=[FakeTensor((bs, 77, 768)) for _ in range(13)],
                __getitem__=lambda self, idx: pooled,
            )
        return (hidden,)

    def to(self, *args, **kwargs):
        return self


class FakeTextEncoder2:
    """Mock SDXL CLIPTextModelWithProjection."""
    device = "cpu"

    def __call__(self, input_ids, output_hidden_states=False, **kwargs):
        bs = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
        hidden = FakeTensor((bs, 77, 1280))
        pooled = FakeTensor((bs, 1280))
        if output_hidden_states:
            hidden_states = [FakeTensor((bs, 77, 1280)) for _ in range(25)]
            ns = SimpleNamespace(hidden_states=hidden_states)
            ns.__class__ = type('FakeOutput', (), {
                '__getitem__': lambda self, idx: pooled,
                'hidden_states': hidden_states,
            })
            return ns
        return (hidden,)

    def to(self, *args, **kwargs):
        return self


class FakeUNet:
    """Mock UNet2DConditionModel."""
    device = "cpu"
    dtype = "float32"

    def __init__(self, channels: int = 4):
        self.config = SimpleNamespace(
            in_channels=channels,
            time_cond_proj_dim=None,
        )
        self._param = None
        if torch_available:
            import torch
            self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def parameters(self):
        if self._param is not None:
            yield self._param

    def __call__(self, latents, timestep, **kwargs):
        shape = latents.shape
        if len(shape) == 4 and shape[1] == 9:
            shape = (shape[0], 4, shape[2], shape[3])
        return SimpleNamespace(sample=FakeTensor(shape))

    def to(self, *args, **kwargs):
        return self


class FakeVAE:
    """Mock AutoencoderKL."""
    device = "cpu"
    dtype = "float32"

    def __init__(self, scaling_factor: float = 0.18215):
        self.config = SimpleNamespace(
            scaling_factor=scaling_factor,
            shift_factor=None,
        )

    def encode(self, x):
        b = x.shape[0] if hasattr(x, 'shape') else 1
        return FakeVAEOutput(
            latent_dist=FakeLatentDist(FakeTensor((b, 4, 64, 64)))
        )

    def decode(self, latents, return_dict=True):
        b = latents.shape[0] if hasattr(latents, 'shape') else 1
        image = FakeTensor((b, 3, 512, 512))
        if return_dict:
            return SimpleNamespace(sample=image)
        return (image,)

    def to(self, *args, **kwargs):
        return self


class FakeScheduler:
    """Mock scheduler matching Diffusers SchedulerMixin interface."""

    def __init__(self):
        self.timesteps = FakeTensor((50,))
        self.init_noise_sigma = 1.0
        self.order = 1
        self.counter = 0

    def set_timesteps(self, num_steps, device=None, **kwargs):
        self.timesteps = FakeTensor((num_steps,))

    def set_begin_index(self, index: int) -> None:
        """Match Diffusers SchedulerMixin (no-op for fake)."""
        self._begin_index = index

    def add_noise(self, original_samples, noise, timesteps):
        """Match Diffusers API; return latents unchanged for graph tests."""
        return original_samples

    def scale_model_input(self, latents, timestep):
        return latents

    def step(self, noise_pred, timestep, latents, **kwargs):
        self.counter += 1
        return SimpleNamespace(prev_sample=FakeTensor(latents.shape))


class FakeControlNet:
    """Mock ControlNetModel."""
    device = "cpu"

    def __call__(self, latents, timestep, controlnet_cond=None, **kwargs):
        down = [FakeTensor((1, 320, 64, 64)) for _ in range(12)]
        mid = FakeTensor((1, 1280, 8, 8))
        return down, mid

    def to(self, *args, **kwargs):
        return self


class FakeImageEncoder:
    """Mock CLIPVisionModelWithProjection."""
    device = "cpu"
    dtype = "float32"

    def __call__(self, pixel_values, output_hidden_states=False, **kwargs):
        b = pixel_values.shape[0] if hasattr(pixel_values, 'shape') else 1
        if output_hidden_states:
            seq, dim = 16, 1024
            h = FakeTensor((b, seq, dim))
            return SimpleNamespace(
                image_embeds=FakeTensor((b, dim)),
                hidden_states=[h, FakeTensor((b, seq, dim))],
            )
        return SimpleNamespace(image_embeds=FakeTensor((b, 1024)))

    def to(self, *args, **kwargs):
        return self


class FakeFeatureExtractor:
    """Mock CLIPImageProcessor."""

    def __call__(self, images, return_tensors="pt"):
        return SimpleNamespace(pixel_values=FakeTensor((len(images) if isinstance(images, list) else 1, 3, 224, 224)))


class FakeT5Tokenizer:
    """Mock T5TokenizerFast."""
    model_max_length = 512

    def __call__(self, text, **kwargs):
        max_len = kwargs.get("max_length", 512)
        return SimpleNamespace(input_ids=FakeTensor((1, max_len)))


class FakeT5Encoder:
    """Mock T5EncoderModel (returns sequence embeddings, no pooler)."""
    device = "cpu"

    def __call__(self, input_ids, **kwargs):
        bs = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
        seq_len = input_ids.shape[1] if hasattr(input_ids, 'shape') and len(input_ids.shape) > 1 else 512
        hidden = FakeTensor((bs, seq_len, 4096))
        return (hidden,)

    def to(self, *args, **kwargs):
        return self


class FakeCLIPEncoderWithPooler:
    """Mock CLIPTextModel that returns pooler_output (for FLUX CLIP path)."""
    device = "cpu"

    def __call__(self, input_ids, output_hidden_states=False, **kwargs):
        bs = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
        hidden = FakeTensor((bs, 77, 768))
        pooled = FakeTensor((bs, 768))
        if output_hidden_states:
            return SimpleNamespace(
                hidden_states=[FakeTensor((bs, 77, 768)) for _ in range(13)],
                pooler_output=pooled,
            )
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=pooled,
        )

    def to(self, *args, **kwargs):
        return self


class FakeFluxTransformer:
    """Mock FluxTransformer2DModel."""
    device = "cpu"
    dtype = "bfloat16"

    def __init__(self, guidance_embeds: bool = True):
        self.config = SimpleNamespace(guidance_embeds=guidance_embeds)

    def __call__(self, hidden_states=None, **kwargs):
        shape = hidden_states.shape if hasattr(hidden_states, 'shape') else (1, 4096, 64)
        return (FakeTensor(shape),)

    def to(self, *args, **kwargs):
        return self


class FakeFluxVAE:
    """Mock AutoencoderKL for FLUX (16 latent channels)."""
    device = "cpu"
    dtype = "bfloat16"

    def __init__(self):
        self.config = SimpleNamespace(
            scaling_factor=0.3611,
            shift_factor=0.1159,
        )

    def encode(self, x):
        b = x.shape[0] if hasattr(x, 'shape') else 1
        return FakeVAEOutput(
            latent_dist=FakeLatentDist(FakeTensor((b, 16, 128, 128)))
        )

    def decode(self, latents, return_dict=True):
        b = latents.shape[0] if hasattr(latents, 'shape') else 1
        image = FakeTensor((b, 3, 1024, 1024))
        if return_dict:
            return SimpleNamespace(sample=image)
        return (image,)

    def to(self, *args, **kwargs):
        return self


class FakeFlowMatchScheduler:
    """Mock FlowMatchEulerDiscreteScheduler."""

    def __init__(self):
        self.timesteps = FakeTensor((28,))
        self.init_noise_sigma = 1.0
        self.order = 1
        self.config = SimpleNamespace(
            base_image_seq_len=256,
            max_image_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15,
        )

    def set_timesteps(self, num_steps, device=None, **kwargs):
        self.timesteps = FakeTensor((num_steps,))

    def scale_model_input(self, latents, timestep):
        return latents

    def step(self, noise_pred, timestep, latents, return_dict=False):
        result = SimpleNamespace(prev_sample=FakeTensor(latents.shape))
        if return_dict:
            return result
        return (FakeTensor(latents.shape),)

    def scale_noise(self, sample, timestep, noise):
        return sample


class FakeFluxControlNet:
    """Mock FluxControlNetModel."""
    device = "cpu"

    def __call__(self, hidden_states=None, **kwargs):
        block_samples = [FakeTensor((1, 4096, 64)) for _ in range(19)]
        single_samples = [FakeTensor((1, 4096, 64)) for _ in range(38)]
        return (block_samples, single_samples)

    def to(self, *args, **kwargs):
        return self


@pytest.fixture
def flux_components():
    return {
        "tokenizer": FakeTokenizer(),
        "tokenizer_2": FakeT5Tokenizer(),
        "text_encoder": FakeCLIPEncoderWithPooler(),
        "text_encoder_2": FakeT5Encoder(),
        "transformer": FakeFluxTransformer(),
        "vae": FakeFluxVAE(),
        "scheduler": FakeFlowMatchScheduler(),
    }


@pytest.fixture
def sd15_components():
    return {
        "tokenizer": FakeTokenizer(),
        "text_encoder": FakeTextEncoder(),
        "unet": FakeUNet(channels=4),
        "vae": FakeVAE(scaling_factor=0.18215),
        "scheduler": FakeScheduler(),
        "safety_checker": None,
        "feature_extractor": FakeFeatureExtractor(),
    }


@pytest.fixture
def sdxl_components():
    return {
        "tokenizer": FakeTokenizer(),
        "tokenizer_2": FakeTokenizer(),
        "text_encoder": FakeTextEncoder(),
        "text_encoder_2": FakeTextEncoder2(),
        "unet": FakeUNet(channels=4),
        "vae": FakeVAE(scaling_factor=0.13025),
        "scheduler": FakeScheduler(),
    }
