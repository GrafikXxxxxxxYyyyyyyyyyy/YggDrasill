from __future__ import annotations

import importlib
import sys


def test_top_level_import_does_not_eager_import_diffusion() -> None:
    sys.modules.pop("yggdrasill", None)
    sys.modules.pop("yggdrasill.integrations.diffusers", None)
    sys.modules.pop("yggdrasill.integrations.diffusers.builder", None)

    pkg = importlib.import_module("yggdrasill")

    assert hasattr(pkg, "Hypergraph")
    assert "yggdrasill.integrations.diffusers" not in sys.modules


def test_diffusion_builder_is_lazy_top_level_attribute() -> None:
    sys.modules.pop("yggdrasill", None)
    sys.modules.pop("yggdrasill.integrations.diffusers", None)
    sys.modules.pop("yggdrasill.integrations.diffusers.builder", None)

    pkg = importlib.import_module("yggdrasill")
    builder = pkg.DiffusionGraphBuilder

    assert builder.__name__ == "DiffusionGraphBuilder"
    assert "yggdrasill.integrations.diffusers" in sys.modules


def test_diffusers_import_does_not_patch_hypergraph_run() -> None:
    from yggdrasill.engine.structure import Hypergraph

    original = Hypergraph.run
    sys.modules.pop("yggdrasill.integrations.diffusers", None)
    sys.modules.pop("yggdrasill.integrations.diffusers.builder", None)
    importlib.import_module("yggdrasill.integrations.diffusers")

    assert Hypergraph.run is original
