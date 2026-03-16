# Yggdrasill

Hypergraph framework from block to universe.

A domain-agnostic engine for building, validating, planning, and executing
computational graphs — from single task-nodes through hypergraphs to
multi-level workflows. Designed for diffusion models, LLMs, agent loops,
and any directed computation pipeline.

## Installation

```bash
pip install -e .

# With development dependencies:
pip install -e ".[dev]"

# With YAML config support:
pip install -e ".[yaml]"

# With diffusion dependencies:
pip install -e ".[diffusion]"
```

## Running tests

```bash
pytest tests/
```

## Quick start

### Build a minimal hypergraph

```python
from yggdrasill.engine.structure import Hypergraph
from yggdrasill.engine.edge import Edge
import yggdrasill.task_nodes  # registers identity stubs

h = Hypergraph(graph_id="demo")
h.add_node_from_config("enc", "converter/identity")
h.add_node_from_config("bb", "backbone/identity")
h.add_edge(Edge("enc", "output", "bb", "latent"))
h.expose_input("enc", "input", "x")
h.expose_input("bb", "timestep", "t")
h.expose_output("bb", "pred", "y")

result = h.run({"x": "data", "t": 0})
print(result)  # {"y": "data"}
```

### Build a workflow (hypergraph of hypergraphs)

```python
from yggdrasill.workflow import Workflow

w = Workflow(workflow_id="pipeline")
w.add_node("step1", h1)  # h1 is a Hypergraph
w.add_node("step2", h2)
w.add_edge("step1", "y", "step2", "x")
w.infer_exposed_ports()

result = w.run({"x": "input"})
```

### Save and load

```python
h.save("/tmp/my_graph")
h2 = Hypergraph.load("/tmp/my_graph")
assert h2.run({"x": "data", "t": 0}) == result
```

## SDXL Examples

- `examples/diffusion/sdxl_text2img.py` builds an SDXL graph via `Hypergraph.from_template("sdxl_text2img", ...)`.
- `examples/diffusion/sdxl_img2img.py` builds an SDXL graph via `Hypergraph.from_template("sdxl_img2img", ...)`.
- `examples/diffusion/sdxl_base_refiner.py` builds the SDXL workflow via `Workflow.from_template("sdxl_base_refiner", ...)`.
- `examples/diffusion/sdxl_manual_graph.py` shows manual SDXL assembly through `ModelStore` and presets.

Shortest form:

```python
from yggdrasill.engine.structure import Hypergraph

graph = Hypergraph.from_template(
    "sdxl_text2img",
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    device="cuda",
)
```

Run any example with:

```bash
python examples/diffusion/sdxl_text2img.py
```

## Architecture

- **Foundation**: `AbstractBaseBlock` (data/computation) + `AbstractGraphNode` (graph position/ports) — dual inheritance, no wrappers
- **Engine**: Universal `Validator` → `Planner` → `Executor` pipeline operating on any structural protocol
- **Task Nodes**: Seven abstract roles (Backbone, Injector, Conjector, InnerModule, OuterModule, Helper, Converter) with canonical port contracts
- **Serialization**: Config (JSON/YAML) + Checkpoint (pickle) with block_id deduplication. See [SECURITY.md](SECURITY.md) for checkpoint safety.
- **Workflow**: Hypergraph-of-hypergraphs executed by the same engine

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0
