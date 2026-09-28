.PHONY: help install install-dev lint format test smoke continual meta transfer clean

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + Atari + TensorBoard
	$(PY) -m pip install -e ".[atari,tensorboard]"

install-dev:  ## Install everything incl. test/lint tooling and git hooks
	$(PY) -m pip install -e ".[all]"
	pre-commit install

lint:  ## Static checks
	ruff check .
	ruff format --check .

format:  ## Auto-format and fix lint
	ruff format .
	ruff check . --fix

test:  ## Unit + integration tests
	$(PY) -m pytest --cov --cov-report=term-missing

smoke:  ## End-to-end run of every entry point (~2 min, CPU)
	bash scripts/smoke_test.sh

continual:  ## Sine benchmark (transfer, MAML, EWC) with the reference config
	bash scripts/reproduce_continual.sh

meta:  ## Meta-RL study on PuckWorld (Reptile vs multitask vs random)
	bash scripts/reproduce_meta.sh puckworld 3

transfer:  ## Snake -> PuckWorld transfer study
	bash scripts/reproduce_transfer.sh snake puckworld 3

clean:  ## Remove caches (keeps outputs/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
