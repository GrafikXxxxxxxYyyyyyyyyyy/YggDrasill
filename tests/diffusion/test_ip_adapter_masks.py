"""IP-Adapter spatial masking (IPAdapterMaskProcessor + UNet cross_attention_kwargs)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")

import torch

from PIL import Image
from yggdrasill.engine.edge import Edge

from yggdrasill.integrations.diffusers.common.ip_adapter_mask_prep import (
    IPAdapterMaskPrepNode,
    prepare_ip_adapter_masks_tensor,
)


def test_prepare_ip_adapter_masks_tensor_resolves_string_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def _fake_load(url: object) -> Image.Image:
        assert isinstance(url, str)
        seen.append(url)
        return Image.new("L", (16, 16), 200)

    monkeypatch.setattr(
        "yggdrasill.integrations.diffusers.common.ip_adapter_mask_prep.load_mask_image",
        _fake_load,
    )
    packed = prepare_ip_adapter_masks_tensor(
        ["https://example.com/a.png", "https://example.com/b.png"],
        height=32,
        width=32,
    )
    assert seen == ["https://example.com/a.png", "https://example.com/b.png"]
    assert packed.shape[0] == 1 and packed.shape[1] == 2


def test_prepare_ip_adapter_masks_tensor_shape() -> None:
    m1 = Image.new("L", (64, 64), 255)
    m2 = Image.new("L", (64, 64), 0)
    packed = prepare_ip_adapter_masks_tensor([m1, m2], height=1024, width=1024)
    assert packed.dim() == 4
    assert packed.shape[0] == 1
    assert packed.shape[1] == 2


def test_ip_adapter_mask_prep_node_forward() -> None:
    node = IPAdapterMaskPrepNode("mask_prep", config={"height": 512, "width": 512})
    m1 = Image.new("L", (32, 32), 200)
    m2 = Image.new("L", (32, 32), 50)
    out = node.forward({"ip_adapter_mask_images": [m1, m2]})
    t = out["ip_adapter_masks"]
    assert t.shape[0] == 1 and t.shape[1] == 2


def test_ip_adapter_mask_prep_node_forward_none_masks() -> None:
    node = IPAdapterMaskPrepNode("mask_prep", config={})
    out = node.forward({})
    assert out["ip_adapter_masks"] is None


def test_prepare_diffusion_pins_mask_prep_when_no_mask_images() -> None:
    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers import contracts as C
    from yggdrasill.integrations.diffusers.run import _prepare_diffusion_run

    g = Hypergraph(graph_id="t")
    g.add_node("ip_mask_prep", IPAdapterMaskPrepNode("ip_mask_prep", config={}))
    g.expose_input("ip_mask_prep", C.PORT_IP_ADAPTER_MASK_IMAGES, C.PORT_IP_ADAPTER_MASK_IMAGES)

    run_kw: dict = {}
    _prepare_diffusion_run(g, run_kw, merged_inputs={})
    assert run_kw["pin_data"]["ip_mask_prep"][C.PORT_IP_ADAPTER_MASKS] is None


def test_ensure_ip_adapter_mask_prep_adds_wired_node() -> None:
    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers import contracts as C
    from yggdrasill.integrations.diffusers.adapters.ip_adapter import IPAdapterNode
    from yggdrasill.integrations.diffusers.builder import _ensure_ip_adapter_mask_prep
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode

    g = Hypergraph(graph_id="t")
    g.add_node("unet", SDXLUNetNode("unet", unet=None))
    g.add_node("ip", IPAdapterNode("ip"))
    _ensure_ip_adapter_mask_prep(g)
    assert "ip_mask_prep" in g.node_ids
    assert g.metadata.get("ip_mask_prep_auto") is True
    assert any(
        e.source_node == "ip_mask_prep"
        and e.source_port == C.PORT_IP_ADAPTER_MASKS
        and e.target_node == "unet"
        and e.target_port == C.PORT_IP_ADAPTER_MASKS
        for e in g.get_edges()
    )


def test_prepare_diffusion_clears_mask_prep_pin_when_images_provided() -> None:
    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers import contracts as C
    from yggdrasill.integrations.diffusers.run import _prepare_diffusion_run

    g = Hypergraph(graph_id="t")
    g.add_node("ip_mask_prep", IPAdapterMaskPrepNode("ip_mask_prep", config={}))
    g.expose_input("ip_mask_prep", C.PORT_IP_ADAPTER_MASK_IMAGES, C.PORT_IP_ADAPTER_MASK_IMAGES)

    sentinel = object()
    run_kw: dict = {"pin_data": {"ip_mask_prep": {C.PORT_IP_ADAPTER_MASKS: sentinel}}}
    merged = {C.PORT_IP_ADAPTER_MASK_IMAGES: [Image.new("L", (4, 4), 0)]}
    _prepare_diffusion_run(g, run_kw, merged_inputs=merged)
    assert "ip_mask_prep" not in run_kw["pin_data"]


def test_prepare_diffusion_skips_mask_prep_when_unet_masks_prepacked() -> None:
    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers import contracts as C
    from yggdrasill.integrations.diffusers.run import _prepare_diffusion_run
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode

    g = Hypergraph(graph_id="t")
    g.add_node("unet", SDXLUNetNode("unet", unet=None))
    g.add_node("ip_mask_prep", IPAdapterMaskPrepNode("ip_mask_prep", config={}))
    g.add_edge(
        Edge("ip_mask_prep", C.PORT_IP_ADAPTER_MASKS, "unet", C.PORT_IP_ADAPTER_MASKS),
    )
    g.expose_input("unet", C.PORT_IP_ADAPTER_MASKS, C.PORT_IP_ADAPTER_MASKS)

    mask_t = torch.zeros(1, 1, 8, 8)
    run_kw: dict = {}
    _prepare_diffusion_run(
        g, run_kw, merged_inputs={C.PORT_IP_ADAPTER_MASKS: mask_t},
    )
    assert "ip_mask_prep" in (run_kw.get("skip_node_ids") or ())
    assert "ip_mask_prep" not in run_kw["pin_data"]


def test_prepare_diffusion_syncs_ip_mask_prep_size_from_latent_init() -> None:
    """Masks must be resized to the generation canvas; latent_init defaults when run omits w/h."""
    from yggdrasill.engine.structure import Hypergraph
    from yggdrasill.integrations.diffusers.run import _prepare_diffusion_run

    class LatentStub:
        block_type = "sdxl/latent_init"
        _config = {"width": 896, "height": 1152}

    g = Hypergraph(graph_id="t")
    g.add_node("latent_init", LatentStub())
    g.add_node("ip_mask_prep", IPAdapterMaskPrepNode("ip_mask_prep", config={}))

    run_kw: dict = {"pin_data": {}}
    _prepare_diffusion_run(g, run_kw, merged_inputs={})
    prep = g.get_node("ip_mask_prep")
    assert prep._config.get("width") == 896
    assert prep._config.get("height") == 1152

    run_kw2 = {"pin_data": {}, "width": 512}
    _prepare_diffusion_run(g, run_kw2, merged_inputs={})
    assert prep._config.get("width") == 512
    assert prep._config.get("height") == 1152


def test_sdxl_unet_passes_cross_attention_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from yggdrasill.integrations.diffusers.sdxl.unet import SDXLUNetNode

    captured: dict = {}

    class DummyUNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.config = type("C", (), {"in_channels": 4, "time_cond_proj_dim": None})()

        def forward(self, sample, timestep, **kwargs):
            captured.update(kwargs)
            return type("O", (), {"sample": torch.zeros_like(sample)})()

    unet = DummyUNet()
    node = SDXLUNetNode("u", unet=unet, config={"guidance_scale": 1.0})
    lat = torch.randn(1, 4, 8, 8)
    ts = torch.tensor([500], dtype=torch.long)
    pe = torch.randn(1, 77, 2048)
    ate = torch.randn(1, 1280)
    ati = torch.randn(1, 6)
    mask = torch.zeros(1, 2, 128, 128)
    node.forward({
        "latents": lat,
        "timestep": ts,
        "prompt_embeds": pe,
        "add_text_embeds": ate,
        "add_time_ids": ati,
        "ip_adapter_masks": mask,
    })
    assert "cross_attention_kwargs" in captured
    assert "ip_adapter_masks" in captured["cross_attention_kwargs"]
    assert len(captured["cross_attention_kwargs"]["ip_adapter_masks"]) == 1
