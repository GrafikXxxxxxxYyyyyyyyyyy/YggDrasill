from __future__ import annotations

from types import SimpleNamespace

import pytest

from yggdrasill.integrations.diffusers.training.config import TrainingConfig
from yggdrasill.integrations.diffusers.training.flux_text2img_objective import FluxText2ImgLoRAObjective
from yggdrasill.integrations.diffusers.training.types import TrainingComponents, TrainingTargetSetup


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed")
def test_flux_text2img_objective_uses_packed_latents_and_flux_conditioning(tmp_path) -> None:
    import torch

    class _Tokenizer:
        model_max_length = 8

        def __call__(self, captions, **kwargs):
            batch = len(captions)
            length = kwargs.get("max_length", self.model_max_length)
            return SimpleNamespace(input_ids=torch.ones((batch, length), dtype=torch.long))

    class _ClipEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(16, 4)

        def forward(self, input_ids, **kwargs):
            hidden = self.emb(input_ids)
            pooled = hidden.mean(dim=1)
            return SimpleNamespace(pooler_output=pooled)

    class _T5Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(16, 6)

        def forward(self, input_ids, **kwargs):
            return (self.emb(input_ids),)

    class _LatentDist:
        def __init__(self, sample):
            self._sample = sample

        def sample(self):
            return self._sample

    class _VAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

        def encode(self, x):
            pooled = torch.nn.functional.avg_pool2d(x, kernel_size=8)
            latents = torch.cat([pooled, pooled, pooled, pooled, pooled, pooled], dim=1)[:, :4]
            latents = torch.cat([latents, latents, latents, latents], dim=1)[:, :16]
            return SimpleNamespace(latent_dist=_LatentDist(latents))

    class _Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(guidance_embeds=True)
            self.last_kwargs = None

        def forward(self, **kwargs):
            self.last_kwargs = kwargs
            hidden_states = kwargs["hidden_states"]
            return (hidden_states * self.weight,)

    class _Scheduler:
        config = SimpleNamespace(prediction_type="epsilon")

    transformer = _Transformer()
    components = TrainingComponents(
        tokenizer=_Tokenizer(),
        tokenizer_2=_Tokenizer(),
        text_encoder=_ClipEncoder(),
        text_encoder_2=_T5Encoder(),
        backbone=transformer,
        backbone_key="transformer",
        vae=_VAE(),
        scheduler=_Scheduler(),
    )
    targets = TrainingTargetSetup(
        backbone=transformer,
        backbone_key="transformer",
        text_encoder=components.text_encoder,
        text_encoder_2=components.text_encoder_2,
        trainable_parameters=list(transformer.parameters()),
    )
    objective = FluxText2ImgLoRAObjective(
        components=components,
        targets=targets,
        config=TrainingConfig(
            pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
            data_dir=str(tmp_path),
            output_path=str(tmp_path / "adapter.safetensors"),
            family="flux",
            guidance=3.5,
        ),
        device=torch.device("cpu"),
    )

    batch = {
        "pixel_values": torch.randn(1, 3, 16, 16),
        "caption": ["forest"],
        "prompt_2": ["misty forest"],
    }
    loss = objective.compute_loss(batch)

    assert loss.ndim == 0
    assert transformer.last_kwargs["hidden_states"].ndim == 3
    assert transformer.last_kwargs["pooled_projections"].shape[0] == 1
    assert transformer.last_kwargs["txt_ids"].shape[-1] == 3
