# Contributing to YggDrasill

Thank you for your interest in contributing.

## Development setup

```bash
git clone https://github.com/your-org/YggDrasill.git
cd YggDrasill
pip install -e ".[dev,yaml]"
# For diffusion tests/examples:
pip install -e ".[diffusion]"
```

## Running tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ -v --cov=yggdrasill --cov-report=term-missing
```

## Linting and formatting

```bash
ruff check yggdrasill/ tests/
ruff format yggdrasill/ tests/
```

## Code style

- Follow the existing style; the project uses ruff for linting.
- Use type hints for public APIs.
- Add tests for new features and bug fixes.
- Update documentation (docstrings, README) when changing behavior.

## Architecture overview

- **foundation/**: `AbstractBaseBlock`, `AbstractGraphNode`, ports, registry
- **engine/**: Validator, Planner, Executor, Hypergraph structure
- **task_nodes/**: Abstract roles and stubs
- **hypergraph/**: Serialization (config + checkpoint)
- **workflow/**: Hypergraph-of-hypergraphs
- **integrations/diffusers/**: SD 1.5, SDXL, Flux pipelines

## Submitting changes

1. Fork the repository.
2. Create a feature branch.
3. Make your changes with tests.
4. Ensure tests pass and lint is clean.
5. Open a pull request with a clear description.
