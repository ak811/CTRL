#!/usr/bin/env bash
# Meta-RL study: Reptile vs. multi-task pretraining vs. random init on held-out tasks.
# Usage: scripts/reproduce_meta.sh [snake|puckworld|pong] [n_seeds]
set -euo pipefail
cd "$(dirname "$0")/.."
ENV="${1:-puckworld}"
SEEDS="${2:-3}"

for seed in $(seq 0 $((SEEDS - 1))); do
  python -m meta.train --config "configs/meta/${ENV}.yaml" --algorithm reptile   --seed "$seed"
  python -m meta.train --config "configs/meta/${ENV}.yaml" --algorithm multitask --seed "$seed"
done

ARGS=()
for seed in $(seq 0 $((SEEDS - 1))); do
  ARGS+=(--checkpoint "reptile=outputs/meta/${ENV}/reptile_seed${seed}/policy.pt")
  ARGS+=(--checkpoint "multitask=outputs/meta/${ENV}/multitask_seed${seed}/policy.pt")
done
python -m meta.evaluate "${ARGS[@]}" --include-random --tasks 32 --output "outputs/meta/${ENV}/comparison"
