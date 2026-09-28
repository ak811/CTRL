#!/usr/bin/env bash
# Reproduce the sine-regression benchmark (transfer, MAML, EWC) reported in README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m continual.run --config configs/continual/default.yaml --run-name "${RUN_NAME:-reference}" "$@"
