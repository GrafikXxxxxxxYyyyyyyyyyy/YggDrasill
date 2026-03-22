"""Tests for named template API."""
from __future__ import annotations

import pytest

from yggdrasill.engine.structure import Hypergraph
from yggdrasill.integrations.diffusers import build_template, list_templates
import yggdrasill.integrations.diffusers.templates as templates
from yggdrasill.integrations.diffusers.builder import DiffusionGraphBuilder
from yggdrasill.workflow.workflow import Workflow


class TestTemplateRegistry:
    def test_list_templates_contains_sdxl_shortcuts(self):
        names = list_templates()
        assert "sdxl_text2img" in names
        assert "sdxl_img2img" in names
        assert "sdxl_base_refiner" in names

    def test_list_templates_contains_flux_shortcuts(self):
        names = list_templates()
        assert "flux_text2img" in names
        assert "flux_img2img" in names
        assert "flux_inpaint" in names
        assert "flux_controlnet_text2img" in names

    def test_build_template_unknown_raises(self):
        with pytest.raises(KeyError):
            build_template("does_not_exist")


class TestHypergraphFromTemplate:
    def test_hypergraph_from_template_returns_graph(self, monkeypatch):
        graph = Hypergraph(graph_id="stub_graph")

        def _stub_builder(**kwargs):
            assert kwargs["device"] == "cpu"
            return graph

        monkeypatch.setitem(
            templates._TEMPLATE_BUILDERS,
            "stub_graph",
            _stub_builder,
        )

        built = Hypergraph.from_template("stub_graph", device="cpu")
        assert built is graph

    def test_hypergraph_from_template_rejects_workflow(self, monkeypatch):
        workflow = Workflow(workflow_id="stub_workflow")

        monkeypatch.setitem(
            templates._TEMPLATE_BUILDERS,
            "stub_workflow",
            lambda **kwargs: workflow,
        )

        with pytest.raises(TypeError):
            Hypergraph.from_template("stub_workflow")


class TestDiffusionGraphBuilderFromTemplate:
    def test_from_template_wraps_graph(self, monkeypatch):
        graph = Hypergraph(graph_id="stub_graph")

        def _stub_builder(**kwargs):
            assert kwargs["device"] == "cpu"
            return graph

        monkeypatch.setitem(
            templates._TEMPLATE_BUILDERS,
            "stub_graph",
            _stub_builder,
        )

        builder = DiffusionGraphBuilder.from_template("stub_graph", device="cpu")
        assert builder.graph is graph

    def test_from_template_rejects_workflow(self, monkeypatch):
        workflow = Workflow(workflow_id="stub_workflow")

        monkeypatch.setitem(
            templates._TEMPLATE_BUILDERS,
            "stub_workflow",
            lambda **kwargs: workflow,
        )

        with pytest.raises(TypeError):
            DiffusionGraphBuilder.from_template("stub_workflow")


class TestWorkflowFromTemplate:
    def test_workflow_from_template_returns_workflow(self, monkeypatch):
        workflow = Workflow(workflow_id="stub_workflow")

        monkeypatch.setitem(
            templates._TEMPLATE_BUILDERS,
            "stub_workflow_ok",
            lambda **kwargs: workflow,
        )

        built = Workflow.from_template("stub_workflow_ok")
        assert built is workflow

    def test_workflow_from_template_rejects_graph(self, monkeypatch):
        graph = Hypergraph(graph_id="stub_graph")

        monkeypatch.setitem(
            templates._TEMPLATE_BUILDERS,
            "stub_graph_bad",
            lambda **kwargs: graph,
        )

        with pytest.raises(TypeError):
            Workflow.from_template("stub_graph_bad")
