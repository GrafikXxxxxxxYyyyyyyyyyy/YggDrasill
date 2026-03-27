"""Domain scenario integration tests — comprehensive coverage of all declared domains."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from yggdrasill.engine.executor import RunResult, run, run_stream
from yggdrasill.engine.planner import build_plan, clear_plan_cache
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.foundation.port import Port, PortDirection, PortType
from yggdrasill.foundation.registry import BlockRegistry
from yggdrasill.hypergraph.serialization import load_hypergraph, save_hypergraph
from yggdrasill.task_nodes.abstract import (
    AbstractBackbone,
    AbstractConjector,
    AbstractConverter,
    AbstractHelper,
    AbstractInjector,
    AbstractInnerModule,
    AbstractOuterModule,
)
from yggdrasill.workflow.workflow import Workflow


# ============================================================================
# Stubs — Diffusion domain (DAG variants)
# ============================================================================

class FakeVAEEncoder(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "diffusion/vae_encoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"latent({inputs.get('input')})"}


class FakeVAEDecoder(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "diffusion/vae_decoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"decoded({inputs.get('input', '')})"}


class FakeCLIPEncoder(AbstractConjector):
    @property
    def block_type(self) -> str:
        return "diffusion/clip_encoder"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"condition": f"clip({inputs.get('input')})"}


class FakeUNet(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "diffusion/unet"

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, PortType.ANY),
            Port("conditioning", PortDirection.IN, PortType.ANY),
            Port("pred", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"pred": f"unet({inputs.get('latent')},{inputs.get('conditioning')})"}


class FakeScheduler(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "diffusion/scheduler"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"scheduled({inputs.get('query', '')})"}


# ============================================================================
# Stubs — Diffusion domain (cycle-capable, canonical ports)
# ============================================================================

class FakeDiffusionBackbone(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "diffusion/backbone_v2"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        lat = inputs.get("latent", "")
        ts = inputs.get("timestep", "")
        cond = inputs.get("condition", "")
        return {"pred": f"unet({lat},{ts},{cond})"}


class FakeDiffusionSolver(AbstractInnerModule):
    @property
    def block_type(self) -> str:
        return "diffusion/solver_v2"

    def declare_ports(self) -> List[Port]:
        return [
            Port("pred", PortDirection.IN, PortType.ANY),
            Port("next_latent", PortDirection.OUT, PortType.ANY),
            Port("next_timestep", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "next_latent": f"step({inputs.get('pred', '')})",
            "next_timestep": "t_next",
        }


class FakeNoiseGen(AbstractOuterModule):
    @property
    def block_type(self) -> str:
        return "diffusion/noise_gen"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY, optional=True),
            Port("latent", PortDirection.OUT, PortType.ANY),
            Port("timestep", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"latent": "noise_init", "timestep": "T=1000"}


class FakeLoRA(AbstractInjector):
    @property
    def block_type(self) -> str:
        return "diffusion/lora"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"adapted": f"lora({inputs.get('condition', '')},{inputs.get('hidden', '')})"}


# ============================================================================
# Stubs — LLM domain
# ============================================================================

class FakeLLM(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "llm/backbone"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"llm({inputs.get('input')})"}


class FakeTokenizer(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "llm/tokenizer"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"tokens({inputs.get('input')})"}


class FakeDetokenizer(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "llm/detokenizer"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"text({inputs.get('input')})"}


class FakeLLMGen(AbstractBackbone):
    """LLM backbone for autoregressive generation loop."""

    @property
    def block_type(self) -> str:
        return "llm/generator"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.TEXT),
            Port("context", PortDirection.IN, PortType.ANY, optional=True),
            Port("logits", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"logits": f"logits({inputs.get('input')},{inputs.get('context')})"}


class FakeSampler(AbstractInnerModule):
    """Token sampler for LLM generation loop."""

    @property
    def block_type(self) -> str:
        return "llm/sampler"

    def declare_ports(self) -> List[Port]:
        return [
            Port("logits", PortDirection.IN, PortType.ANY),
            Port("next_input", PortDirection.OUT, PortType.TEXT),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"next_input": f"tok({inputs.get('logits')})"}


class FakePrefill(AbstractOuterModule):
    @property
    def block_type(self) -> str:
        return "llm/prefill"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"prefill({inputs.get('input', '')})"}


# ============================================================================
# Stubs — Agent domain
# ============================================================================

class FakeRouter(AbstractInnerModule):
    @property
    def block_type(self) -> str:
        return "agent/router"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"route({inputs.get('input')})"}


class FakeAgentWithToolCalls(AbstractInnerModule):
    @property
    def block_type(self) -> str:
        return "agent/with_tool_calls"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "output": {
                "response": f"thinking about {inputs.get('input')}",
                "tool_calls": [
                    {"tool_id": "search", "args": {"query": "test"}},
                    {"tool_id": "calc", "args": {"expr": "2+2"}},
                ],
            }
        }


class FakeToolExecutor(AbstractConjector):
    @property
    def block_type(self) -> str:
        return "agent/tool_exec"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"condition": f"tool_result({inputs.get('input')})"}


class FakeMemory(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "agent/memory"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"mem({inputs.get('query', '')})"}


class FakeAgentLLM(AbstractBackbone):
    """Agent backbone that returns tool_calls on first call, final answer after."""

    def __init__(
        self, node_id: str, block_id: str | None = None, *, config: dict | None = None,
    ) -> None:
        super().__init__(node_id=node_id, block_id=block_id, config=config)
        self._call_count = 0

    @property
    def block_type(self) -> str:
        return "agent/llm"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.TEXT),
            Port("output", PortDirection.OUT, PortType.TEXT),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            return {
                "output": f"thinking({inputs.get('input')})",
                "tool_calls": [
                    {"tool_id": "search", "args": {"query": "test"}},
                    {"tool_id": "calc", "args": {"expr": "2+2"}},
                ],
            }
        tr = inputs.get("tool_results", [])
        return {"output": f"final(tools={len(tr)})"}


class FakePersistentAgent(AbstractBackbone):
    """Agent that always returns tool_calls (for max_steps testing)."""

    @property
    def block_type(self) -> str:
        return "agent/persistent"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.TEXT),
            Port("output", PortDirection.OUT, PortType.TEXT),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "output": "still_thinking",
            "tool_calls": [{"tool_id": "search", "args": {"query": "more"}}],
        }


class FakeSearchTool(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "tool/search"

    def declare_ports(self) -> List[Port]:
        return [
            Port("query", PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"found({inputs.get('query', '')})"}


class FakeCalcTool(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "tool/calc"

    def declare_ports(self) -> List[Port]:
        return [
            Port("expr", PortDirection.IN, PortType.ANY, optional=True),
            Port("result", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"computed({inputs.get('expr', '')})"}


# ============================================================================
# Stubs — Audio domain
# ============================================================================

class FakeAudioBackbone(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "audio/backbone"

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, PortType.AUDIO),
            Port("timestep", PortDirection.IN, PortType.ANY),
            Port("condition", PortDirection.IN, PortType.ANY, optional=True),
            Port("pred", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"pred": f"audio_unet({inputs.get('latent')},{inputs.get('timestep')})"}


class FakeAudioSolver(AbstractInnerModule):
    @property
    def block_type(self) -> str:
        return "audio/solver"

    def declare_ports(self) -> List[Port]:
        return [
            Port("pred", PortDirection.IN, PortType.ANY),
            Port("next_latent", PortDirection.OUT, PortType.AUDIO),
            Port("next_timestep", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "next_latent": f"audio_step({inputs.get('pred')})",
            "next_timestep": "at_next",
        }


class FakeAudioDecoder(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "audio/decoder"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.AUDIO),
            Port("output", PortDirection.OUT, PortType.AUDIO),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"audio_decode({inputs.get('input')})"}


class FakeCLAP(AbstractConjector):
    @property
    def block_type(self) -> str:
        return "audio/clap"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"condition": f"clap({inputs.get('input')})"}


# ============================================================================
# Stubs — Video domain
# ============================================================================

class FakeVideoBackbone(AbstractBackbone):
    @property
    def block_type(self) -> str:
        return "video/backbone"

    def declare_ports(self) -> List[Port]:
        return [
            Port("latent", PortDirection.IN, PortType.VIDEO),
            Port("timestep", PortDirection.IN, PortType.ANY),
            Port("condition", PortDirection.IN, PortType.ANY, optional=True),
            Port("pred", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"pred": f"video_unet({inputs.get('latent')},{inputs.get('timestep')})"}


class FakeVideoSolver(AbstractInnerModule):
    @property
    def block_type(self) -> str:
        return "video/solver"

    def declare_ports(self) -> List[Port]:
        return [
            Port("pred", PortDirection.IN, PortType.ANY),
            Port("next_latent", PortDirection.OUT, PortType.VIDEO),
            Port("next_timestep", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "next_latent": f"video_step({inputs.get('pred')})",
            "next_timestep": "vt_next",
        }


class FakeVideoDecoder(AbstractConverter):
    @property
    def block_type(self) -> str:
        return "video/decoder"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.VIDEO),
            Port("output", PortDirection.OUT, PortType.VIDEO),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"video_decode({inputs.get('input')})"}


# ============================================================================
# Stubs — RAG domain
# ============================================================================

class FakeRetriever(AbstractHelper):
    @property
    def block_type(self) -> str:
        return "rag/retriever"

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"docs_for({inputs.get('query', '')})"}


class FakeMerger(AbstractConjector):
    """Merges retrieved documents with the original query."""

    @property
    def block_type(self) -> str:
        return "rag/merger"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("context", PortDirection.IN, PortType.ANY, optional=True),
            Port("condition", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"condition": f"merged({inputs.get('input')},{inputs.get('context')})"}


# ============================================================================
# Fixture
# ============================================================================

@pytest.fixture()
def domain_registry():
    r = BlockRegistry()
    # Diffusion
    r.register("diffusion/vae_encoder", FakeVAEEncoder)
    r.register("diffusion/vae_decoder", FakeVAEDecoder)
    r.register("diffusion/clip_encoder", FakeCLIPEncoder)
    r.register("diffusion/unet", FakeUNet)
    r.register("diffusion/scheduler", FakeScheduler)
    r.register("diffusion/backbone_v2", FakeDiffusionBackbone)
    r.register("diffusion/solver_v2", FakeDiffusionSolver)
    r.register("diffusion/noise_gen", FakeNoiseGen)
    r.register("diffusion/lora", FakeLoRA)
    # LLM
    r.register("llm/backbone", FakeLLM)
    r.register("llm/tokenizer", FakeTokenizer)
    r.register("llm/detokenizer", FakeDetokenizer)
    r.register("llm/generator", FakeLLMGen)
    r.register("llm/sampler", FakeSampler)
    r.register("llm/prefill", FakePrefill)
    # Agent
    r.register("agent/router", FakeRouter)
    r.register("agent/tool_exec", FakeToolExecutor)
    r.register("agent/memory", FakeMemory)
    r.register("agent/with_tool_calls", FakeAgentWithToolCalls)
    r.register("agent/llm", FakeAgentLLM)
    r.register("agent/persistent", FakePersistentAgent)
    r.register("tool/search", FakeSearchTool)
    r.register("tool/calc", FakeCalcTool)
    # Audio
    r.register("audio/backbone", FakeAudioBackbone)
    r.register("audio/solver", FakeAudioSolver)
    r.register("audio/decoder", FakeAudioDecoder)
    r.register("audio/clap", FakeCLAP)
    # Video
    r.register("video/backbone", FakeVideoBackbone)
    r.register("video/solver", FakeVideoSolver)
    r.register("video/decoder", FakeVideoDecoder)
    # RAG
    r.register("rag/retriever", FakeRetriever)
    r.register("rag/merger", FakeMerger)
    return r


# ============================================================================
# B-1: Diffusion pipeline tests
# ============================================================================

class TestDiffusionPipeline:
    def test_diffusion_hypergraph(self, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "diffusion_pipeline",
            "graph_kind": "task_hypergraph",
            "nodes": [
                {"node_id": "vae_enc", "block_type": "diffusion/vae_encoder"},
                {"node_id": "clip", "block_type": "diffusion/clip_encoder"},
                {"node_id": "unet", "block_type": "diffusion/unet"},
                {"node_id": "sched", "block_type": "diffusion/scheduler"},
                {"node_id": "vae_dec", "block_type": "diffusion/vae_decoder"},
            ],
            "edges": [
                {"source_node": "vae_enc", "source_port": "output", "target_node": "unet", "target_port": "latent"},
                {"source_node": "clip", "source_port": "condition", "target_node": "unet", "target_port": "conditioning"},
                {"source_node": "unet", "source_port": "pred", "target_node": "sched", "target_port": "query"},
                {"source_node": "sched", "source_port": "result", "target_node": "vae_dec", "target_port": "input"},
            ],
            "exposed_inputs": [
                {"node_id": "vae_enc", "port_name": "input", "name": "image"},
                {"node_id": "clip", "port_name": "input", "name": "prompt"},
            ],
            "exposed_outputs": [
                {"node_id": "vae_dec", "port_name": "output", "name": "output"},
            ],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        out = h.run({"image": "img_data", "prompt": "a cat"})
        assert "output" in out
        assert "unet" in out["output"]
        assert "clip" in out["output"]
        assert "latent" in out["output"]

    def test_diffusion_serialization_roundtrip(self, tmp_path, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "diff_ser",
            "nodes": [
                {"node_id": "enc", "block_type": "diffusion/vae_encoder"},
                {"node_id": "dec", "block_type": "diffusion/vae_decoder"},
            ],
            "edges": [
                {"source_node": "enc", "source_port": "output", "target_node": "dec", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "enc", "port_name": "input", "name": "x"}],
            "exposed_outputs": [{"node_id": "dec", "port_name": "output", "name": "y"}],
        }
        h1 = Hypergraph.from_config(cfg, registry=domain_registry)
        out1 = h1.run({"x": "pixel"})
        save_hypergraph(h1, tmp_path / "diff")
        h2 = load_hypergraph(tmp_path / "diff", registry=domain_registry)
        out2 = h2.run({"x": "pixel"})
        assert out1 == out2

    def test_diffusion_t2i_with_cycle(self, domain_registry):
        """Full T2I: CLIP → Backbone ↔ Solver (K iterations) → VAE decode."""
        clear_plan_cache()
        h = Hypergraph(graph_id="t2i")
        h.add_node_from_config("clip", "diffusion/clip_encoder", registry=domain_registry)
        h.add_node_from_config("backbone", "diffusion/backbone_v2", registry=domain_registry)
        h.add_node_from_config("solver", "diffusion/solver_v2", registry=domain_registry)
        h.add_node_from_config("vae_dec", "diffusion/vae_decoder", registry=domain_registry)

        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("clip", "condition", "backbone", "condition"))
        h.add_edge(Edge("backbone", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "backbone", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "backbone", "timestep"))
        h.add_edge(Edge("solver", "next_latent", "vae_dec", "input"))

        h.expose_input("backbone", "latent", "noise")
        h.expose_input("backbone", "timestep", "timestep")
        h.expose_input("clip", "input", "prompt")
        h.expose_output("vae_dec", "output", "image")

        plan = build_plan(h)
        step_types = [s[0] for s in plan]
        assert "cycle" in step_types
        assert "node" in step_types

        out = h.run(
            {"noise": "z0", "timestep": "T0", "prompt": "a cat"},
            num_loop_steps=3,
            validate_before=False,
        )
        assert "image" in out
        assert "decoded" in out["image"]
        assert "step" in out["image"]

    def test_diffusion_with_injector(self, domain_registry):
        """Diffusion pipeline with LoRA injector feeding into backbone."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("lora", "diffusion/lora", registry=domain_registry)
        h.add_node_from_config("backbone", "diffusion/backbone_v2", registry=domain_registry)
        h.add_node_from_config("solver", "diffusion/solver_v2", registry=domain_registry)

        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("lora", "adapted", "backbone", "condition"))
        h.add_edge(Edge("backbone", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "backbone", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "backbone", "timestep"))

        h.expose_input("lora", "condition", "lora_cond")
        h.expose_input("backbone", "latent", "noise")
        h.expose_input("backbone", "timestep", "ts")
        h.expose_output("solver", "next_latent", "output")

        out = h.run(
            {"lora_cond": "style", "noise": "z", "ts": "T"},
            num_loop_steps=2, validate_before=False,
        )
        assert "output" in out
        assert "lora" in out["output"]

    def test_diffusion_workflow_t2i_to_upscale(self, domain_registry):
        """Workflow: T2I hypergraph → Upscale hypergraph."""
        clear_plan_cache()
        t2i = Hypergraph(graph_id="t2i")
        t2i.add_node_from_config("enc", "diffusion/vae_encoder", registry=domain_registry)
        t2i.add_node_from_config("dec", "diffusion/vae_decoder", registry=domain_registry)
        from yggdrasill.engine.edge import Edge
        t2i.add_edge(Edge("enc", "output", "dec", "input"))
        t2i.expose_input("enc", "input", "image")
        t2i.expose_output("dec", "output", "output")

        upscale = Hypergraph(graph_id="upscale")
        upscale.add_node_from_config("up_enc", "diffusion/vae_encoder", registry=domain_registry)
        upscale.add_node_from_config("up_dec", "diffusion/vae_decoder", registry=domain_registry)
        upscale.add_edge(Edge("up_enc", "output", "up_dec", "input"))
        upscale.expose_input("up_enc", "input", "input")
        upscale.expose_output("up_dec", "output", "output")

        w = Workflow(workflow_id="t2i_upscale")
        w.add_node("gen", t2i)
        w.add_node("up", upscale)
        w.add_edge("gen", "output", "up", "input")
        w.expose_input("gen", "input", "image")
        w.expose_output("up", "output", "result")

        result = w.run({"image": "pixels"}, validate_before=False)
        assert "result" in result
        assert "decoded" in result["result"]


# ============================================================================
# B-2: LLM pipeline tests
# ============================================================================

class TestLLMChain:
    def test_llm_hypergraph(self, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "llm_pipe",
            "nodes": [
                {"node_id": "tok", "block_type": "llm/tokenizer"},
                {"node_id": "llm", "block_type": "llm/backbone"},
                {"node_id": "detok", "block_type": "llm/detokenizer"},
            ],
            "edges": [
                {"source_node": "tok", "source_port": "output", "target_node": "llm", "target_port": "input"},
                {"source_node": "llm", "source_port": "output", "target_node": "detok", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "tok", "port_name": "input", "name": "text"}],
            "exposed_outputs": [{"node_id": "detok", "port_name": "output", "name": "response"}],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        out = h.run({"text": "hello world"})
        assert "response" in out
        assert "llm" in out["response"]
        assert "tokens" in out["response"]

    def test_llm_workflow_two_steps(self, domain_registry):
        clear_plan_cache()
        llm_cfg = {
            "graph_id": "llm_step",
            "nodes": [
                {"node_id": "tok", "block_type": "llm/tokenizer"},
                {"node_id": "llm", "block_type": "llm/backbone"},
                {"node_id": "detok", "block_type": "llm/detokenizer"},
            ],
            "edges": [
                {"source_node": "tok", "source_port": "output", "target_node": "llm", "target_port": "input"},
                {"source_node": "llm", "source_port": "output", "target_node": "detok", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "tok", "port_name": "input", "name": "input"}],
            "exposed_outputs": [{"node_id": "detok", "port_name": "output", "name": "output"}],
        }
        step1 = Hypergraph.from_config(llm_cfg, registry=domain_registry)
        step2 = Hypergraph.from_config(llm_cfg, registry=domain_registry)
        w = Workflow(workflow_id="llm_chain")
        w.add_node("summarize", step1)
        w.add_node("refine", step2)
        w.add_edge("summarize", "output", "refine", "input")
        w.expose_input("summarize", "input", "input")
        w.expose_output("refine", "output", "output")
        out = w.run({"input": "long document"}, validate_before=False)
        assert "output" in out
        assert "llm" in out["output"]

    def test_llm_completion_loop(self, domain_registry):
        """LLM generation loop: Tokenizer → [LLM ↔ Sampler] x K → Detokenizer."""
        clear_plan_cache()
        h = Hypergraph(graph_id="llm_gen")
        h.add_node_from_config("tok", "llm/tokenizer", registry=domain_registry)
        h.add_node_from_config("llm", "llm/generator", registry=domain_registry)
        h.add_node_from_config("sampler", "llm/sampler", registry=domain_registry)
        h.add_node_from_config("detok", "llm/detokenizer", registry=domain_registry)

        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("tok", "output", "llm", "context"))
        h.add_edge(Edge("llm", "logits", "sampler", "logits"))
        h.add_edge(Edge("sampler", "next_input", "llm", "input"))
        h.add_edge(Edge("sampler", "next_input", "detok", "input"))

        h.expose_input("tok", "input", "prompt")
        h.expose_input("llm", "input", "start_token")
        h.expose_output("detok", "output", "text")

        plan = build_plan(h)
        assert any(s[0] == "cycle" for s in plan)

        out = h.run(
            {"prompt": "hello", "start_token": "<bos>"},
            num_loop_steps=3, validate_before=False,
        )
        assert "text" in out
        assert "tok" in out["text"]


# ============================================================================
# B-7: RAG pipeline tests
# ============================================================================

class TestRAGPipeline:
    def test_rag_standalone(self, domain_registry):
        """RAG: Retriever → Merger + query → LLM → answer."""
        clear_plan_cache()
        h = Hypergraph(graph_id="rag")
        h.graph_kind = "rag"
        h.add_node_from_config("retriever", "rag/retriever", registry=domain_registry)
        h.add_node_from_config("merger", "rag/merger", registry=domain_registry)
        h.add_node_from_config("llm", "llm/backbone", registry=domain_registry)

        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("retriever", "result", "merger", "context"))
        h.add_edge(Edge("merger", "condition", "llm", "input"))

        h.expose_input("retriever", "query", "query")
        h.expose_input("merger", "input", "original_query")
        h.expose_output("llm", "output", "answer")

        out = h.run(
            {"query": "quantum computing", "original_query": "explain quantum"},
            validate_before=False,
        )
        assert "answer" in out
        assert "merged" in out["answer"]
        assert "docs_for" in out["answer"]

    def test_rag_metadata_graph_kind(self, domain_registry):
        """Verify graph_kind='rag' is preserved in config roundtrip."""
        clear_plan_cache()
        h = Hypergraph(graph_id="rag_test")
        h.graph_kind = "rag"
        h.add_node_from_config("r", "rag/retriever", registry=domain_registry)
        h.expose_input("r", "query", "q")
        h.expose_output("r", "result", "docs")

        cfg = h.to_config()
        assert cfg["graph_kind"] == "rag"
        h2 = Hypergraph.from_config(cfg, registry=domain_registry)
        assert h2.graph_kind == "rag"


# ============================================================================
# B-3: Agent with engine agent_loop
# ============================================================================

class TestAgentLoop:
    def test_agent_hypergraph_with_cycle(self, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "agent_loop",
            "metadata": {"num_loop_steps": 3},
            "nodes": [
                {"node_id": "router", "block_type": "agent/router"},
                {"node_id": "tool", "block_type": "agent/tool_exec"},
                {"node_id": "mem", "block_type": "agent/memory"},
            ],
            "edges": [
                {"source_node": "router", "source_port": "output", "target_node": "tool", "target_port": "input"},
                {"source_node": "tool", "source_port": "condition", "target_node": "mem", "target_port": "query"},
                {"source_node": "mem", "source_port": "result", "target_node": "router", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "router", "port_name": "input", "name": "query"}],
            "exposed_outputs": [{"node_id": "mem", "port_name": "result", "name": "answer"}],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        out = h.run({"query": "what is 2+2?"}, num_loop_steps=3, validate_before=False)
        assert "answer" in out
        assert "route" in out["answer"]
        assert "tool_result" in out["answer"]
        assert "mem" in out["answer"]

    def test_agent_tool_calls_contract(self, domain_registry):
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("agent", "agent/with_tool_calls", registry=domain_registry)
        h.expose_input("agent", "input", "query")
        h.expose_output("agent", "output", "result")
        out = h.run({"query": "what is 2+2?"}, validate_before=False)
        result = out["result"]
        assert isinstance(result, dict)
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 2

    def test_agent_workflow_with_cycle(self, domain_registry):
        clear_plan_cache()
        agent_cfg = {
            "graph_id": "agent_step",
            "metadata": {"num_loop_steps": 2},
            "nodes": [
                {"node_id": "router", "block_type": "agent/router"},
                {"node_id": "tool", "block_type": "agent/tool_exec"},
            ],
            "edges": [
                {"source_node": "router", "source_port": "output", "target_node": "tool", "target_port": "input"},
                {"source_node": "tool", "source_port": "condition", "target_node": "router", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "router", "port_name": "input", "name": "input"}],
            "exposed_outputs": [{"node_id": "tool", "port_name": "condition", "name": "condition"}],
        }
        step1 = Hypergraph.from_config(agent_cfg, registry=domain_registry)
        step2 = Hypergraph.from_config(agent_cfg, registry=domain_registry)
        w = Workflow(workflow_id="agent_wf")
        w.add_node("think", step1)
        w.add_node("act", step2)
        w.add_edge("think", "condition", "act", "input")
        w.expose_input("think", "input", "q")
        w.expose_output("act", "condition", "a")
        result = w.run({"q": "plan"}, validate_before=False)
        assert "a" in result
        assert "route" in result["a"]

    def test_agent_loop_with_tools(self, domain_registry):
        """Engine agent_loop: agent returns tool_calls → tools execute → repeat."""
        clear_plan_cache()
        h = Hypergraph(graph_id="agent_with_tools")
        h.add_node_from_config("agent", "agent/llm", registry=domain_registry)
        h.add_node_from_config("search_tool", "tool/search", registry=domain_registry)
        h.add_node_from_config("calc_tool", "tool/calc", registry=domain_registry)
        h.metadata = {
            "agent_node_ids": ["agent"],
            "tool_id_to_node_id": {"search": "search_tool", "calc": "calc_tool"},
        }
        h.expose_input("agent", "input", "query")
        h.expose_output("agent", "output", "answer")

        plan = build_plan(h)
        assert plan == [("agent_loop", "agent")]

        out = h.run({"query": "what is 2+2?"}, validate_before=False)
        assert "final" in out["answer"]
        assert "tools=2" in out["answer"]

    def test_agent_loop_max_steps(self, domain_registry):
        """Agent that always returns tool_calls is stopped by max_steps."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("agent", "agent/persistent", registry=domain_registry)
        h.add_node_from_config("search_tool", "tool/search", registry=domain_registry)
        h.metadata = {
            "agent_node_ids": ["agent"],
            "tool_id_to_node_id": {"search": "search_tool"},
        }
        h.expose_input("agent", "input", "q")
        h.expose_output("agent", "output", "a")

        log: list = []
        h.run(
            {"q": "loop"}, max_steps=3, validate_before=False,
            callbacks=[lambda p, i: log.append(p)],
        )
        agent_steps = [x for x in log if x == "agent_step"]
        assert len(agent_steps) == 3

    def test_agent_loop_callbacks(self, domain_registry):
        """Verify agent_loop fires proper callback phases."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("agent", "agent/llm", registry=domain_registry)
        h.add_node_from_config("s", "tool/search", registry=domain_registry)
        h.add_node_from_config("c", "tool/calc", registry=domain_registry)
        h.metadata = {
            "agent_node_ids": ["agent"],
            "tool_id_to_node_id": {"search": "s", "calc": "c"},
        }
        h.expose_input("agent", "input", "q")
        h.expose_output("agent", "output", "a")

        phases: list = []
        h.run(
            {"q": "test"}, validate_before=False,
            callbacks=[lambda p, _i: phases.append(p)],
        )
        assert "agent_step" in phases
        assert "tool_call" in phases
        assert "agent_loop_done" in phases


# ============================================================================
# B-4: Audio pipeline tests
# ============================================================================

class TestAudioPipeline:
    def test_audio_ldm_hypergraph(self, domain_registry):
        """AudioLDM-like: CLAP → [AudioBackbone ↔ AudioSolver] → AudioDecoder."""
        clear_plan_cache()
        h = Hypergraph(graph_id="audio_ldm")
        h.add_node_from_config("clap", "audio/clap", registry=domain_registry)
        h.add_node_from_config("backbone", "audio/backbone", registry=domain_registry)
        h.add_node_from_config("solver", "audio/solver", registry=domain_registry)
        h.add_node_from_config("decoder", "audio/decoder", registry=domain_registry)

        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("clap", "condition", "backbone", "condition"))
        h.add_edge(Edge("backbone", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "backbone", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "backbone", "timestep"))
        h.add_edge(Edge("solver", "next_latent", "decoder", "input"))

        h.expose_input("backbone", "latent", "noise")
        h.expose_input("backbone", "timestep", "ts")
        h.expose_input("clap", "input", "text")
        h.expose_output("decoder", "output", "audio")

        out = h.run(
            {"noise": "audio_z", "ts": "AT0", "text": "piano music"},
            num_loop_steps=2, validate_before=False,
        )
        assert "audio" in out
        assert "audio_decode" in out["audio"]
        assert "audio_step" in out["audio"]


# ============================================================================
# B-5: Video pipeline tests
# ============================================================================

class TestVideoPipeline:
    def test_video_svd_hypergraph(self, domain_registry):
        """SVD-like: [VideoBackbone ↔ VideoSolver] → VideoDecoder."""
        clear_plan_cache()
        h = Hypergraph(graph_id="video_svd")
        h.add_node_from_config("backbone", "video/backbone", registry=domain_registry)
        h.add_node_from_config("solver", "video/solver", registry=domain_registry)
        h.add_node_from_config("decoder", "video/decoder", registry=domain_registry)

        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("backbone", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "backbone", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "backbone", "timestep"))
        h.add_edge(Edge("solver", "next_latent", "decoder", "input"))

        h.expose_input("backbone", "latent", "video_noise")
        h.expose_input("backbone", "timestep", "vts")
        h.expose_output("decoder", "output", "video")

        out = h.run(
            {"video_noise": "vz", "vts": "VT0"},
            num_loop_steps=2, validate_before=False,
        )
        assert "video" in out
        assert "video_decode" in out["video"]
        assert "video_step" in out["video"]


# ============================================================================
# B-6: Cross-domain workflow
# ============================================================================

class TestCrossDomainWorkflow:
    def test_llm_to_diffusion_workflow(self, domain_registry):
        """Workflow: LLM generates description → Diffusion generates image."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        llm_hg = Hypergraph(graph_id="llm")
        llm_hg.add_node_from_config("tok", "llm/tokenizer", registry=domain_registry)
        llm_hg.add_node_from_config("llm", "llm/backbone", registry=domain_registry)
        llm_hg.add_node_from_config("detok", "llm/detokenizer", registry=domain_registry)
        llm_hg.add_edge(Edge("tok", "output", "llm", "input"))
        llm_hg.add_edge(Edge("llm", "output", "detok", "input"))
        llm_hg.expose_input("tok", "input", "input")
        llm_hg.expose_output("detok", "output", "output")

        diff_hg = Hypergraph(graph_id="diff")
        diff_hg.add_node_from_config("enc", "diffusion/vae_encoder", registry=domain_registry)
        diff_hg.add_node_from_config("dec", "diffusion/vae_decoder", registry=domain_registry)
        diff_hg.add_edge(Edge("enc", "output", "dec", "input"))
        diff_hg.expose_input("enc", "input", "input")
        diff_hg.expose_output("dec", "output", "output")

        w = Workflow(workflow_id="llm_to_diff")
        w.add_node("text_gen", llm_hg)
        w.add_node("image_gen", diff_hg)
        w.add_edge("text_gen", "output", "image_gen", "input")
        w.expose_input("text_gen", "input", "prompt")
        w.expose_output("image_gen", "output", "image")

        out = w.run({"prompt": "describe a sunset"}, validate_before=False)
        assert "image" in out
        assert "decoded" in out["image"]
        assert "llm" in out["image"]


# ============================================================================
# Engine feature tests (pin_data, interrupt, stream, partial, seed)
# ============================================================================

class TestEngineFeatures:
    def test_pin_data_skips_node(self, domain_registry):
        """Pin a node's output; it should not execute, outputs come from pin_data."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("tok", "llm/tokenizer", registry=domain_registry)
        h.add_node_from_config("llm", "llm/backbone", registry=domain_registry)
        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("tok", "output", "llm", "input"))
        h.expose_input("tok", "input", "text")
        h.expose_output("llm", "output", "result")

        log: list = []
        out = run(
            h, {"text": "hello"},
            pin_data={"tok": {"output": "PINNED_TOKENS"}},
            validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        assert out["result"] == "llm(PINNED_TOKENS)"
        assert ("pinned", "tok") in log
        assert ("before", "tok") not in log

    def test_interrupt_and_resume(self):
        """Interrupt before a node, then resume."""
        from tests.engine.helpers import make_chain
        clear_plan_cache()
        h = make_chain("A", "B", "C")

        result = run(h, {"x": 42}, interrupt_on=["B"], validate_before=False)
        assert isinstance(result, RunResult)
        assert result.suspended is True
        assert result.run_data is not None
        assert result.outputs["y"] is None

        final = run(h, {"x": 42}, run_data=result.run_data, validate_before=False)
        assert final["y"] == 42

    def test_partial_run_with_dirty_nodes(self):
        """Only dirty nodes and their dependents should execute."""
        from tests.engine.helpers import make_chain
        clear_plan_cache()
        h = make_chain("A", "B", "C")

        log: list = []
        run(
            h, {"x": 42},
            run_data={"A": {"out": 42}},
            dirty_node_ids=["B"],
            validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        executed = [nid for phase, nid in log if phase == "before"]
        assert "A" not in executed
        assert "B" in executed
        assert "C" in executed

    def test_run_stream(self):
        """run_stream yields intermediate outputs after each step."""
        from tests.engine.helpers import make_chain
        clear_plan_cache()
        h = make_chain("A", "B", "C")

        snapshots = list(run_stream(h, {"x": 42}, validate_before=False))
        assert len(snapshots) == 3
        assert snapshots[0]["y"] is None
        assert snapshots[1]["y"] is None
        assert snapshots[2]["y"] == 42

    def test_seed_propagation(self):
        """Verify seed is set on nodes that support it."""
        from tests.foundation.helpers import IdentityTaskNode
        clear_plan_cache()
        h = Hypergraph()
        node = IdentityTaskNode(node_id="N")
        h.add_node("N", node)
        h.expose_input("N", "in", "x")
        h.expose_output("N", "out", "y")

        run(h, {"x": 1}, seed=42, validate_before=False)
        if hasattr(node, "seed"):
            assert node.seed == 42

    def test_destination_node_stops_execution(self):
        """destination_node_id stops execution after the target node."""
        from tests.engine.helpers import make_chain
        clear_plan_cache()
        h = make_chain("A", "B", "C")

        log: list = []
        run(
            h, {"x": 42}, destination_node_id="B", validate_before=False,
            callbacks=[lambda p, i: log.append((p, i.get("node_id")))],
        )
        executed = [nid for phase, nid in log if phase == "before"]
        assert "A" in executed
        assert "B" in executed
        assert "C" not in executed

    def test_run_stream_with_cycle(self):
        """run_stream yields intermediate outputs for cycles."""
        from tests.engine.helpers import make_cycle
        clear_plan_cache()
        h = make_cycle("A", "B")
        snapshots = list(run_stream(
            h, {"x": 42}, num_loop_steps=2, validate_before=False,
        ))
        assert len(snapshots) == 2
        assert snapshots[-1]["y"] == 42

    def test_run_stream_with_agent_loop(self, domain_registry):
        """run_stream yields a snapshot after each agent-loop iteration."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("agent", "agent/llm", registry=domain_registry)
        h.add_node_from_config("s", "tool/search", registry=domain_registry)
        h.add_node_from_config("c", "tool/calc", registry=domain_registry)
        h.metadata = {
            "agent_node_ids": ["agent"],
            "tool_id_to_node_id": {"search": "s", "calc": "c"},
        }
        h.expose_input("agent", "input", "q")
        h.expose_output("agent", "output", "a")
        snapshots = list(run_stream(
            h, {"q": "test"}, validate_before=False,
        ))
        assert len(snapshots) == 2
        assert "thinking" in snapshots[0]["a"]
        assert "final" in snapshots[-1]["a"]

    def test_interrupt_in_cycle(self):
        """Interrupt inside a cycle body returns suspended result."""
        from tests.engine.helpers import make_cycle
        clear_plan_cache()
        h = make_cycle("A", "B")
        result = run(
            h, {"x": 42}, interrupt_on=["B"],
            num_loop_steps=3, validate_before=False,
        )
        assert isinstance(result, RunResult)
        assert result.suspended is True

    def test_pin_data_in_agent_loop(self, domain_registry):
        """Pin data for an agent node skips the agent_loop entirely."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("agent", "agent/llm", registry=domain_registry)
        h.add_node_from_config("s", "tool/search", registry=domain_registry)
        h.metadata = {
            "agent_node_ids": ["agent"],
            "tool_id_to_node_id": {"search": "s"},
        }
        h.expose_input("agent", "input", "q")
        h.expose_output("agent", "output", "a")

        out = run(
            h, {"q": "test"},
            pin_data={"agent": {"output": "PINNED_ANSWER"}},
            validate_before=False,
        )
        assert out["a"] == "PINNED_ANSWER"


# ============================================================================
# C-1 / C-2: Serialization roundtrip tests for all domains
# ============================================================================

class TestSerializationRoundtrips:
    def _roundtrip_hg(self, tmp_path, registry, h, inputs, subdir="rt"):
        """Helper: run → save → load → run → assert same outputs."""
        out1 = h.run(inputs, validate_before=False)
        save_hypergraph(h, tmp_path / subdir)
        h2 = load_hypergraph(tmp_path / subdir, registry=registry)
        out2 = h2.run(inputs, validate_before=False)
        assert out1 == out2
        return out1

    def test_diffusion_cycle_roundtrip(self, tmp_path, domain_registry):
        clear_plan_cache()
        h = Hypergraph(graph_id="diff_rt")
        h.add_node_from_config("backbone", "diffusion/backbone_v2", registry=domain_registry)
        h.add_node_from_config("solver", "diffusion/solver_v2", registry=domain_registry)
        h.add_node_from_config("dec", "diffusion/vae_decoder", registry=domain_registry)
        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("backbone", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "backbone", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "backbone", "timestep"))
        h.add_edge(Edge("solver", "next_latent", "dec", "input"))
        h.expose_input("backbone", "latent", "z")
        h.expose_input("backbone", "timestep", "t")
        h.expose_output("dec", "output", "img")
        h.metadata = {"num_loop_steps": 2}
        self._roundtrip_hg(tmp_path, domain_registry, h, {"z": "n", "t": "T"})

    def test_llm_roundtrip(self, tmp_path, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "llm_rt",
            "nodes": [
                {"node_id": "tok", "block_type": "llm/tokenizer"},
                {"node_id": "llm", "block_type": "llm/backbone"},
                {"node_id": "detok", "block_type": "llm/detokenizer"},
            ],
            "edges": [
                {"source_node": "tok", "source_port": "output", "target_node": "llm", "target_port": "input"},
                {"source_node": "llm", "source_port": "output", "target_node": "detok", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "tok", "port_name": "input", "name": "text"}],
            "exposed_outputs": [{"node_id": "detok", "port_name": "output", "name": "out"}],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        self._roundtrip_hg(tmp_path, domain_registry, h, {"text": "hi"}, "llm_rt")

    def test_agent_roundtrip(self, tmp_path, domain_registry):
        clear_plan_cache()
        cfg = {
            "graph_id": "agent_rt",
            "metadata": {"num_loop_steps": 2},
            "nodes": [
                {"node_id": "r", "block_type": "agent/router"},
                {"node_id": "t", "block_type": "agent/tool_exec"},
            ],
            "edges": [
                {"source_node": "r", "source_port": "output", "target_node": "t", "target_port": "input"},
                {"source_node": "t", "source_port": "condition", "target_node": "r", "target_port": "input"},
            ],
            "exposed_inputs": [{"node_id": "r", "port_name": "input", "name": "q"}],
            "exposed_outputs": [{"node_id": "t", "port_name": "condition", "name": "a"}],
        }
        h = Hypergraph.from_config(cfg, registry=domain_registry)
        self._roundtrip_hg(tmp_path, domain_registry, h, {"q": "test"}, "agent_rt")

    def test_audio_roundtrip(self, tmp_path, domain_registry):
        clear_plan_cache()
        h = Hypergraph(graph_id="audio_rt")
        h.add_node_from_config("b", "audio/backbone", registry=domain_registry)
        h.add_node_from_config("s", "audio/solver", registry=domain_registry)
        h.add_node_from_config("d", "audio/decoder", registry=domain_registry)
        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("b", "pred", "s", "pred"))
        h.add_edge(Edge("s", "next_latent", "b", "latent"))
        h.add_edge(Edge("s", "next_timestep", "b", "timestep"))
        h.add_edge(Edge("s", "next_latent", "d", "input"))
        h.expose_input("b", "latent", "z")
        h.expose_input("b", "timestep", "t")
        h.expose_output("d", "output", "audio")
        h.metadata = {"num_loop_steps": 2}
        self._roundtrip_hg(tmp_path, domain_registry, h, {"z": "az", "t": "at"}, "audio_rt")

    def test_video_roundtrip(self, tmp_path, domain_registry):
        clear_plan_cache()
        h = Hypergraph(graph_id="video_rt")
        h.add_node_from_config("b", "video/backbone", registry=domain_registry)
        h.add_node_from_config("s", "video/solver", registry=domain_registry)
        h.add_node_from_config("d", "video/decoder", registry=domain_registry)
        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("b", "pred", "s", "pred"))
        h.add_edge(Edge("s", "next_latent", "b", "latent"))
        h.add_edge(Edge("s", "next_timestep", "b", "timestep"))
        h.add_edge(Edge("s", "next_latent", "d", "input"))
        h.expose_input("b", "latent", "z")
        h.expose_input("b", "timestep", "t")
        h.expose_output("d", "output", "video")
        h.metadata = {"num_loop_steps": 2}
        self._roundtrip_hg(tmp_path, domain_registry, h, {"z": "vz", "t": "vt"}, "video_rt")

    def test_rag_roundtrip(self, tmp_path, domain_registry):
        clear_plan_cache()
        h = Hypergraph(graph_id="rag_rt")
        h.graph_kind = "rag"
        h.add_node_from_config("r", "rag/retriever", registry=domain_registry)
        h.add_node_from_config("m", "rag/merger", registry=domain_registry)
        h.add_node_from_config("l", "llm/backbone", registry=domain_registry)
        from yggdrasill.engine.edge import Edge
        h.add_edge(Edge("r", "result", "m", "context"))
        h.add_edge(Edge("m", "condition", "l", "input"))
        h.expose_input("r", "query", "query")
        h.expose_input("m", "input", "orig")
        h.expose_output("l", "output", "answer")
        self._roundtrip_hg(
            tmp_path, domain_registry, h,
            {"query": "q", "orig": "o"}, "rag_rt",
        )

    def test_cross_domain_workflow_roundtrip(self, tmp_path, domain_registry):
        """Workflow save → load → run roundtrip for cross-domain (LLM → Diffusion)."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        llm_hg = Hypergraph(graph_id="llm")
        llm_hg.add_node_from_config("tok", "llm/tokenizer", registry=domain_registry)
        llm_hg.add_node_from_config("llm", "llm/backbone", registry=domain_registry)
        llm_hg.add_edge(Edge("tok", "output", "llm", "input"))
        llm_hg.expose_input("tok", "input", "input")
        llm_hg.expose_output("llm", "output", "output")

        diff_hg = Hypergraph(graph_id="diff")
        diff_hg.add_node_from_config("enc", "diffusion/vae_encoder", registry=domain_registry)
        diff_hg.add_node_from_config("dec", "diffusion/vae_decoder", registry=domain_registry)
        diff_hg.add_edge(Edge("enc", "output", "dec", "input"))
        diff_hg.expose_input("enc", "input", "input")
        diff_hg.expose_output("dec", "output", "output")

        w = Workflow(workflow_id="cross")
        w.add_node("llm_step", llm_hg)
        w.add_node("diff_step", diff_hg)
        w.add_edge("llm_step", "output", "diff_step", "input")
        w.expose_input("llm_step", "input", "prompt")
        w.expose_output("diff_step", "output", "image")

        inputs = {"prompt": "sunset"}
        out1 = w.run(inputs, validate_before=False)

        w.save(tmp_path / "cross_wf")
        w2 = Workflow.load(tmp_path / "cross_wf", registry=domain_registry)
        out2 = w2.run(inputs, validate_before=False)
        assert out1 == out2


# ============================================================================
# Extended domain edge cases
# ============================================================================


class _AllRolesUpscaler(AbstractInnerModule):
    """Dummy upscaler to complete the 7-role ensemble."""

    @property
    def block_type(self) -> str:
        return "diffusion/upscaler"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.ANY),
            Port("output", PortDirection.OUT, PortType.ANY),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": f"upscaled({inputs.get('input')})"}


class _IsAgentNode(AbstractBackbone):
    """Backbone node that exposes is_agent=True via attribute."""
    is_agent = True

    def __init__(self, node_id: str, **kw: Any) -> None:
        super().__init__(node_id=node_id, **kw)
        self._calls = 0

    @property
    def block_type(self) -> str:
        return "test/is_agent_backbone"

    def declare_ports(self) -> List[Port]:
        return [
            Port("input", PortDirection.IN, PortType.TEXT),
            Port("output", PortDirection.OUT, PortType.TEXT),
        ]

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._calls += 1
        if self._calls == 1:
            return {
                "output": "thinking",
                "tool_calls": [{"tool_id": "search", "args": {"q": "x"}}],
            }
        return {"output": "answer"}


class TestDiffusionAllRoles:
    """Verify all 7 abstract roles can coexist in one diffusion graph."""

    def test_all_seven_roles(self, domain_registry):
        clear_plan_cache()
        domain_registry.register("diffusion/upscaler", _AllRolesUpscaler)
        from yggdrasill.engine.edge import Edge

        h = Hypergraph(graph_id="all_roles")
        h.add_node_from_config("backbone", "diffusion/backbone_v2", registry=domain_registry)
        h.add_node_from_config("inner", "diffusion/solver_v2", registry=domain_registry)
        h.add_node_from_config("outer", "diffusion/noise_gen", registry=domain_registry)
        h.add_node_from_config("injector", "diffusion/lora", registry=domain_registry)
        h.add_node_from_config("conjector", "diffusion/clip_encoder", registry=domain_registry)
        h.add_node_from_config("helper", "diffusion/scheduler", registry=domain_registry)
        h.add_node_from_config("converter", "diffusion/vae_decoder", registry=domain_registry)

        h.add_edge(Edge("outer", "latent", "backbone", "latent"))
        h.add_edge(Edge("outer", "timestep", "backbone", "timestep"))
        h.add_edge(Edge("conjector", "condition", "injector", "condition"))
        h.add_edge(Edge("injector", "adapted", "backbone", "condition"))
        h.add_edge(Edge("backbone", "pred", "inner", "pred"))
        h.add_edge(Edge("inner", "next_latent", "converter", "input"))

        h.expose_input("outer", "input", "seed_input")
        h.expose_input("conjector", "input", "prompt")
        h.expose_input("helper", "query", "sched_query")
        h.expose_output("converter", "output", "image")
        h.expose_output("helper", "result", "schedule")

        out = h.run(
            {"seed_input": "s", "prompt": "a cat", "sched_query": "linear"},
            validate_before=False,
        )
        assert "image" in out
        assert "schedule" in out
        assert "decoded" in out["image"]


class TestDiffusionImg2Img:
    def test_img2img_pipeline(self, domain_registry):
        """Img2img: VAE encode → Backbone ↔ Solver (cycle) → VAE decode."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        h = Hypergraph(graph_id="img2img")
        h.add_node_from_config("enc", "diffusion/vae_encoder", registry=domain_registry)
        h.add_node_from_config("bb", "diffusion/backbone_v2", registry=domain_registry)
        h.add_node_from_config("solver", "diffusion/solver_v2", registry=domain_registry)
        h.add_node_from_config("dec", "diffusion/vae_decoder", registry=domain_registry)

        h.add_edge(Edge("enc", "output", "bb", "latent"))
        h.add_edge(Edge("bb", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "bb", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "bb", "timestep"))
        h.add_edge(Edge("solver", "next_latent", "dec", "input"))

        h.expose_input("enc", "input", "image")
        h.expose_input("bb", "timestep", "strength")
        h.expose_output("dec", "output", "result")

        out = h.run(
            {"image": "pixels", "strength": "0.7"},
            num_loop_steps=2, validate_before=False,
        )
        assert "result" in out
        assert "decoded" in out["result"]


class TestDiffusionNoiseGenOuterModule:
    def test_noise_gen_feeds_backbone(self, domain_registry):
        """OuterModule (noise gen) provides initial latent+timestep."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        h = Hypergraph()
        h.add_node_from_config("noise", "diffusion/noise_gen", registry=domain_registry)
        h.add_node_from_config("bb", "diffusion/backbone_v2", registry=domain_registry)

        h.add_edge(Edge("noise", "latent", "bb", "latent"))
        h.add_edge(Edge("noise", "timestep", "bb", "timestep"))

        h.expose_input("noise", "input", "seed")
        h.expose_output("bb", "pred", "prediction")

        out = h.run({"seed": "42"}, validate_before=False)
        assert "prediction" in out
        assert "noise_init" in out["prediction"]


class TestAgentIsAgentAttribute:
    def test_is_agent_attribute_runs_agent_loop(self, domain_registry):
        """Node with is_agent=True is auto-detected by planner without metadata."""
        clear_plan_cache()
        agent = _IsAgentNode("ag")
        h = Hypergraph()
        h.add_node("ag", agent)
        h.add_node_from_config("s", "tool/search", registry=domain_registry)
        h.expose_input("ag", "input", "q")
        h.expose_output("ag", "output", "a")
        h.metadata = {"tool_id_to_node_id": {"search": "s"}}

        plan = build_plan(h)
        assert plan[0][0] == "agent_loop"

        out = run(h, {"q": "hi"}, validate_before=False)
        assert out["a"] == "answer"


class TestAgentGraphKindDetection:
    def test_graph_kind_agent_auto_promotes_backbone(self, domain_registry):
        """graph_kind='agent' promotes backbone-role nodes to agent_loop."""
        clear_plan_cache()
        agent = _IsAgentNode("ag")
        agent.is_agent = False  # disable attribute, rely on graph_kind

        h = Hypergraph()
        h.add_node("ag", agent)
        h.add_node_from_config("s", "tool/search", registry=domain_registry)
        h.expose_input("ag", "input", "q")
        h.expose_output("ag", "output", "a")
        h.graph_kind = "agent"
        h.metadata = {"tool_id_to_node_id": {"search": "s"}}

        plan = build_plan(h)
        assert plan[0][0] == "agent_loop"


class TestAgentNoToolsAvailable:
    def test_agent_without_tools_runs_once(self, domain_registry):
        """Agent with tool_calls but no tool_map should still complete."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("agent", "agent/llm", registry=domain_registry)
        h.expose_input("agent", "input", "q")
        h.expose_output("agent", "output", "a")
        h.metadata = {"agent_node_ids": ["agent"]}

        out = run(h, {"q": "hi"}, validate_before=False)
        assert "a" in out


class TestAudioWithConditioning:
    def test_audio_clap_conditioning(self, domain_registry):
        """Audio pipeline with CLAP text conditioning."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        h = Hypergraph(graph_id="audio_cond")
        h.add_node_from_config("clap", "audio/clap", registry=domain_registry)
        h.add_node_from_config("bb", "audio/backbone", registry=domain_registry)
        h.add_node_from_config("solver", "audio/solver", registry=domain_registry)
        h.add_node_from_config("dec", "audio/decoder", registry=domain_registry)

        h.add_edge(Edge("clap", "condition", "bb", "condition"))
        h.add_edge(Edge("bb", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "bb", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "bb", "timestep"))
        h.add_edge(Edge("solver", "next_latent", "dec", "input"))

        h.expose_input("clap", "input", "text_prompt")
        h.expose_input("bb", "latent", "noise")
        h.expose_input("bb", "timestep", "t")
        h.expose_output("dec", "output", "audio")

        out = h.run(
            {"text_prompt": "birds singing", "noise": "az", "t": "T"},
            num_loop_steps=2, validate_before=False,
        )
        assert "audio" in out
        assert "audio_decode" in out["audio"]


class TestVideoWithConditioning:
    def test_video_with_image_conditioning(self, domain_registry):
        """Video pipeline with image input (SVD-like)."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        h = Hypergraph(graph_id="video_cond")
        enc = FakeVAEEncoder.__new__(FakeVAEEncoder)
        FakeVAEEncoder.__init__(enc, node_id="img_enc")
        h.add_node("img_enc", enc)
        h.add_node_from_config("bb", "video/backbone", registry=domain_registry)
        h.add_node_from_config("solver", "video/solver", registry=domain_registry)
        h.add_node_from_config("dec", "video/decoder", registry=domain_registry)

        h.add_edge(Edge("img_enc", "output", "bb", "condition"))
        h.add_edge(Edge("bb", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "bb", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "bb", "timestep"))
        h.add_edge(Edge("solver", "next_latent", "dec", "input"))

        h.expose_input("img_enc", "input", "reference_image")
        h.expose_input("bb", "latent", "noise")
        h.expose_input("bb", "timestep", "t")
        h.expose_output("dec", "output", "video")

        out = h.run(
            {"reference_image": "frame0", "noise": "vz", "t": "T"},
            num_loop_steps=2, validate_before=False,
        )
        assert "video" in out
        assert "video_decode" in out["video"]


class TestRAGTwoRetrievers:
    def test_rag_dual_retriever(self, domain_registry):
        """RAG with two retrievers feeding into merger via CONCAT aggregation."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        h = Hypergraph(graph_id="rag_dual")
        h.add_node_from_config("r1", "rag/retriever", registry=domain_registry)
        h.add_node_from_config("r2", "rag/retriever", registry=domain_registry)
        h.add_node_from_config("merger", "rag/merger", registry=domain_registry)
        h.add_node_from_config("llm", "llm/backbone", registry=domain_registry)

        h.add_edge(Edge("r1", "result", "merger", "context"))
        h.add_edge(Edge("r2", "result", "merger", "context"))
        h.add_edge(Edge("merger", "condition", "llm", "input"))

        h.expose_input("r1", "query", "q1")
        h.expose_input("r2", "query", "q2")
        h.expose_input("merger", "input", "original")
        h.expose_output("llm", "output", "answer")

        out = h.run(
            {"q1": "search1", "q2": "search2", "original": "question"},
            validate_before=False,
        )
        assert "answer" in out


class TestWorkflowThreeStages:
    def test_three_stage_workflow(self, domain_registry):
        """Workflow with three sequential hypergraph stages."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        def make_passthrough(gid):
            hg = Hypergraph(graph_id=gid)
            hg.add_node_from_config("enc", "diffusion/vae_encoder", registry=domain_registry)
            hg.add_node_from_config("dec", "diffusion/vae_decoder", registry=domain_registry)
            hg.add_edge(Edge("enc", "output", "dec", "input"))
            hg.expose_input("enc", "input", "input")
            hg.expose_output("dec", "output", "output")
            return hg

        w = Workflow(workflow_id="three_stage")
        for name in ("stage1", "stage2", "stage3"):
            w.add_node(name, make_passthrough(name))
        w.add_edge("stage1", "output", "stage2", "input")
        w.add_edge("stage2", "output", "stage3", "input")
        w.expose_input("stage1", "input", "in")
        w.expose_output("stage3", "output", "out")

        out = w.run({"in": "data"}, validate_before=False)
        assert "out" in out
        assert "decoded" in out["out"]


class TestLargeChain:
    def test_ten_node_chain(self, domain_registry):
        """Chain of 10 tokenizer nodes — verifies large DAG handling."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        h = Hypergraph(graph_id="chain10")
        for i in range(10):
            h.add_node_from_config(f"n{i}", "llm/tokenizer", registry=domain_registry)
        for i in range(9):
            h.add_edge(Edge(f"n{i}", "output", f"n{i+1}", "input"))
        h.expose_input("n0", "input", "start")
        h.expose_output("n9", "output", "end")

        plan = build_plan(h)
        assert len(plan) == 10
        ids = [s[1] for s in plan]
        assert ids == [f"n{i}" for i in range(10)]

        out = h.run({"start": "x"}, validate_before=False)
        assert "end" in out


class TestSingleNodeAgentLoop:
    def test_single_agent_node_no_tools(self, domain_registry):
        """Single agent_loop node with no tools — should still complete."""
        clear_plan_cache()
        h = Hypergraph()
        h.add_node_from_config("a", "agent/llm", registry=domain_registry)
        h.expose_input("a", "input", "q")
        h.expose_output("a", "output", "a")
        h.metadata = {"agent_node_ids": ["a"]}
        out = run(h, {"q": "hi"}, validate_before=False)
        assert "a" in out


class TestDiffusionInpainting:
    def test_inpainting_pipeline(self, domain_registry):
        """Inpainting: image + mask → encoder → Backbone ↔ Solver → decoder."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        h = Hypergraph(graph_id="inpaint")
        h.add_node_from_config("enc", "diffusion/vae_encoder", registry=domain_registry)
        h.add_node_from_config("clip", "diffusion/clip_encoder", registry=domain_registry)
        h.add_node_from_config("bb", "diffusion/backbone_v2", registry=domain_registry)
        h.add_node_from_config("solver", "diffusion/solver_v2", registry=domain_registry)
        h.add_node_from_config("dec", "diffusion/vae_decoder", registry=domain_registry)

        h.add_edge(Edge("enc", "output", "bb", "latent"))
        h.add_edge(Edge("clip", "condition", "bb", "condition"))
        h.add_edge(Edge("bb", "pred", "solver", "pred"))
        h.add_edge(Edge("solver", "next_latent", "bb", "latent"))
        h.add_edge(Edge("solver", "next_timestep", "bb", "timestep"))
        h.add_edge(Edge("solver", "next_latent", "dec", "input"))

        h.expose_input("enc", "input", "masked_image")
        h.expose_input("clip", "input", "prompt")
        h.expose_input("bb", "timestep", "t")
        h.expose_output("dec", "output", "result")

        out = h.run(
            {"masked_image": "img+mask", "prompt": "fill with flowers", "t": "T0"},
            num_loop_steps=2, validate_before=False,
        )
        assert "result" in out
        assert "decoded" in out["result"]


class TestMultimodalWorkflow:
    def test_audio_to_text_workflow(self, domain_registry):
        """Cross-modal: audio decoding → LLM transcription."""
        clear_plan_cache()

        audio_hg = Hypergraph(graph_id="audio_decode")
        audio_hg.add_node_from_config("dec", "audio/decoder", registry=domain_registry)
        audio_hg.expose_input("dec", "input", "input")
        audio_hg.expose_output("dec", "output", "output")

        llm_hg = Hypergraph(graph_id="transcribe")
        llm_hg.add_node_from_config("llm", "llm/backbone", registry=domain_registry)
        llm_hg.expose_input("llm", "input", "input")
        llm_hg.expose_output("llm", "output", "output")

        w = Workflow(workflow_id="audio_to_text")
        w.add_node("decode", audio_hg)
        w.add_node("transcribe", llm_hg)
        w.add_edge("decode", "output", "transcribe", "input")
        w.expose_input("decode", "input", "audio")
        w.expose_output("transcribe", "output", "text")

        out = w.run({"audio": "audio_data"}, validate_before=False)
        assert "text" in out
        assert "llm" in out["text"]


class TestDiffusionMemoryHelper:
    def test_diffusion_with_memory(self, domain_registry):
        """Pipeline using a Helper (memory) alongside backbone."""
        clear_plan_cache()
        from yggdrasill.engine.edge import Edge

        h = Hypergraph()
        h.add_node_from_config("mem", "agent/memory", registry=domain_registry)
        h.add_node_from_config("bb", "diffusion/backbone_v2", registry=domain_registry)
        h.add_node_from_config("dec", "diffusion/vae_decoder", registry=domain_registry)

        h.add_edge(Edge("mem", "result", "bb", "condition"))
        h.add_edge(Edge("bb", "pred", "dec", "input"))

        h.expose_input("mem", "query", "context")
        h.expose_input("bb", "latent", "noise")
        h.expose_input("bb", "timestep", "t")
        h.expose_output("dec", "output", "image")

        out = h.run(
            {"context": "remember style", "noise": "z", "t": "T"},
            validate_before=False,
        )
        assert "image" in out
        assert "decoded" in out["image"]


class TestWorkflowMetadataPreserved:
    def test_workflow_roundtrip_preserves_metadata(self, tmp_path, domain_registry):
        """Verify metadata survives workflow save/load."""
        clear_plan_cache()

        hg = Hypergraph(graph_id="inner_meta")
        hg.add_node_from_config("tok", "llm/tokenizer", registry=domain_registry)
        hg.expose_input("tok", "input", "input")
        hg.expose_output("tok", "output", "output")
        hg.metadata = {"domain": "llm", "version": 1}

        w = Workflow(workflow_id="meta_wf")
        w.add_node("step", hg)
        w.expose_input("step", "input", "in")
        w.expose_output("step", "output", "out")

        out1 = w.run({"in": "hello"}, validate_before=False)
        w.save(tmp_path / "meta_wf")
        w2 = Workflow.load(tmp_path / "meta_wf", registry=domain_registry)
        out2 = w2.run({"in": "hello"}, validate_before=False)
        assert out1 == out2

        inner = w2.get_node("step")
        assert inner.metadata.get("domain") == "llm"
        assert inner.metadata.get("version") == 1


class TestAgentLoopInWorkflow:
    def test_agent_subgraph_in_workflow(self, domain_registry):
        """Workflow containing an agent hypergraph as a sub-node."""
        clear_plan_cache()

        agent_hg = Hypergraph(graph_id="agent_sub")
        agent_hg.add_node_from_config("agent", "agent/llm", registry=domain_registry)
        agent_hg.add_node_from_config("s", "tool/search", registry=domain_registry)
        agent_hg.expose_input("agent", "input", "input")
        agent_hg.expose_output("agent", "output", "output")
        agent_hg.metadata = {
            "agent_node_ids": ["agent"],
            "tool_id_to_node_id": {"search": "s"},
        }

        llm_hg = Hypergraph(graph_id="summarize")
        llm_hg.add_node_from_config("llm", "llm/backbone", registry=domain_registry)
        llm_hg.expose_input("llm", "input", "input")
        llm_hg.expose_output("llm", "output", "output")

        w = Workflow(workflow_id="agent_wf")
        w.add_node("research", agent_hg)
        w.add_node("summarize", llm_hg)
        w.add_edge("research", "output", "summarize", "input")
        w.expose_input("research", "input", "question")
        w.expose_output("summarize", "output", "answer")

        out = w.run({"question": "what is AI?"}, validate_before=False)
        assert "answer" in out
        assert "llm" in out["answer"]
