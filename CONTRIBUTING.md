# Contributing

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pre-commit install
```

## Workflow

1. Create a branch from `main`.
2. Make changes with tests. Every bug fix should come with a regression test.
3. Run `make lint test` (and `make smoke` for changes to entry points).
4. Open a pull request describing the change and, for experimental changes, the effect on
   the reference results.

## Conventions

- **Configs.** Every tunable lives in a config dataclass (`*/config.py`), which automatically
  exposes it as a CLI flag. Validate values in `__post_init__`.
- **Randomness.** Never use global RNG state inside library code. Derive seeds from the
  generator returned by `common.seed.set_seed`, and seed environments via `reset(seed=...)`.
- **Outputs.** Write artefacts only inside the run directory from `common.io.create_run_dir`
  and record provenance with `save_run_metadata`.
- **Reporting.** Aggregate across seeds with `common.stats` (mean ± 95% Student-t CI). Do not
  report single-seed RL results as findings.
- **Style.** Ruff (lint + format), type hints on public functions, docstrings that cite the
  method's source paper where applicable.

## Adding an environment

1. Implement a `gymnasium.Env` in `envs/` that seeds via `super().reset(seed=seed)` and uses
   `self.np_random`.
2. Register it in `envs/registry.py`.
3. Add it to the parametrised `check_env` test in `tests/test_envs.py`.
4. For meta-learning, add a `TaskDistribution` with disjoint train/test splits in `envs/tasks.py`.
