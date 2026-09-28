#!/usr/bin/env bash
# End-to-end smoke test of every entry point (~2 min on CPU). Used by CI.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-/tmp/ctrl_smoke}"
rm -rf "$OUT"
python -m continual.run --config configs/continual/quick.yaml --n-seeds 1 --output-root "$OUT"
python -m meta.train --config configs/meta/smoke.yaml --output-root "$OUT" --run-name reptile
python -m meta.train --config configs/meta/smoke.yaml --algorithm multitask --output-root "$OUT" --run-name mt
python -m meta.evaluate --checkpoint "r=$OUT/meta/snake/reptile/policy.pt" --include-random --tasks 1 --output "$OUT/meta/eval"
python -m transfer.train --config configs/transfer/smoke.yaml --output-root "$OUT" --run-name src
python -m transfer.train --config configs/transfer/smoke.yaml --env puckworld --output-root "$OUT" --run-name scratch_seed0
python -m transfer.train --config configs/transfer/smoke.yaml --env puckworld --output-root "$OUT" \
  --init-from "$OUT/transfer/snake/src/final_model.zip" --freeze-encoder --run-name frozen_seed0
python -m transfer.analyze --run "scratch=$OUT/transfer/puckworld/scratch_seed*" \
  --run "frozen=$OUT/transfer/puckworld/frozen_seed*" --baseline scratch --output "$OUT/transfer/analysis"
python -m transfer.evaluate --model "$OUT/transfer/puckworld/frozen_seed0/final_model.zip" --episodes 1
echo "Smoke test passed. Outputs in $OUT"
