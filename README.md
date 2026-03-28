# YggDrasill

YggDrasill is a hypergraph framework that currently provides a real executable core for:

- blocks and nodes;
- task hypergraphs;
- workflows built from hypergraphs;
- config/checkpoint serialization;
- experimental diffusion integrations on top of the same engine.

The repository also contains a broader architectural canon that extends above the currently implemented runtime. Start here:

1. `documentation/IMPLEMENTATION_STATUS.md` for the honest current-state map.
2. `documentation/API_SURFACE.md` for stable vs experimental API boundaries.
3. `documentation/README.md` for full documentation navigation.
4. `philosophy/` for the philosophical foundation of the project.

## Current status

Implemented and test-backed today:

- `Hypergraph` as the main low-level execution surface;
- `Workflow` as a hypergraph of hypergraphs;
- validator / planner / executor;
- cycles, partial run, suspend-resume, and minimal agent loop;
- block and hypergraph serialization.

Not yet a finished production runtime:

- higher layers (`Stage`, `World`, `Universe`) described in the canon are not shipped as code yet;
- full parity with every ambition described in the domain canon;
- full stabilization of high-level diffusion ergonomics.

## Install

Core only:

```bash
pip install -e .
```

Core + development tools:

```bash
pip install -e ".[dev,yaml]"
```

Core + diffusion stack:

```bash
pip install -e ".[dev,yaml,diffusion]"
```

## Reproducibility strategy

This repository currently uses:

- explicit optional dependency groups in `pyproject.toml`;
- CI-validated install paths for core and diffusion subsets;
- source builds through `python -m build`.

For local reproducibility, create a clean virtual environment and install one of the supported extras combinations above instead of mixing ad-hoc packages into a long-lived environment.

## Supported usage layers

Stable low-level path:

- `yggdrasill.Hypergraph`
- `yggdrasill.workflow.workflow.Workflow`
- explicit graph assembly and execution
- `examples/core_hypergraph.py` as the simplest core-only starting point

Experimental / evolving path:

- diffusion builders and templates
- convenience auto-connect layers
- future higher ontology levels
- diffusion notebooks in `examples/`

## Examples

Core-first example:

- `examples/core_hypergraph.py` runs on the stable runtime surface and does not require diffusion dependencies.

Experimental notebooks:

- `examples/diffusion_sd15.ipynb`
- `examples/diffusion_sdxl.ipynb`
- `examples/diffusion_train_lora.ipynb`

These notebooks are diffusion-addon examples, not the canonical stable API path. They typically assume `torch`, `diffusers`, model downloads, and a suitable GPU environment.

See `examples/README.md` for the examples policy and environment guidance.

## Development

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

Build the package:

```bash
python -m build
```
