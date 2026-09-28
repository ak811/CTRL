#!/usr/bin/env bash
# Cross-domain transfer study: SOURCE -> TARGET with scratch / fine-tune / frozen encoder.
# Usage: scripts/reproduce_transfer.sh [source] [target] [n_seeds]
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-snake}"
TGT="${2:-puckworld}"
SEEDS="${3:-3}"
SRC_RUN="outputs/transfer/${SRC}/source_for_${TGT}"

if [[ ! -f "${SRC_RUN}/final_model.zip" ]]; then
  python -m transfer.train --config "configs/transfer/${SRC}.yaml" --run-name "source_for_${TGT}"
fi

for seed in $(seq 0 $((SEEDS - 1))); do
  python -m transfer.train --config "configs/transfer/${TGT}.yaml" --seed "$seed" \
      --run-name "scratch_seed${seed}"
  python -m transfer.train --config "configs/transfer/${TGT}.yaml" --seed "$seed" \
      --init-from "${SRC_RUN}/final_model.zip" --run-name "finetune_seed${seed}"
  python -m transfer.train --config "configs/transfer/${TGT}.yaml" --seed "$seed" \
      --init-from "${SRC_RUN}/final_model.zip" --freeze-encoder --run-name "frozen_seed${seed}"
done

python -m transfer.analyze \
  --run "scratch=outputs/transfer/${TGT}/scratch_seed*" \
  --run "finetune=outputs/transfer/${TGT}/finetune_seed*" \
  --run "frozen=outputs/transfer/${TGT}/frozen_seed*" \
  --baseline scratch --title "${SRC} → ${TGT}" \
  --output "outputs/transfer/analysis_${SRC}_to_${TGT}"
