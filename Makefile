.PHONY: test lint format install-dev

install-dev:
	pip install -e ".[dev,yaml]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=yggdrasill --cov-report=term-missing

lint:
	ruff check yggdrasill/ tests/

format:
	ruff format yggdrasill/ tests/
	ruff check --fix yggdrasill/ tests/
