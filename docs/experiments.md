# Experiment Protocols

Every entry point follows the same conventions:

- **Configuration precedence**: dataclass defaults < `--config file.yaml` < command-line flags. Every config field is exposed as a flag (`maml_inner_lr` → `--maml-inner-lr`, booleans as `--flag / --no-flag`). Unknown YAML keys are rejected.
- **Run directories**: `outputs/<paradigm>/<env>/<run-name>/` (default run name is derived from the method and seed). Existing directories are never overwritten; a numeric suffix is appended instead.
- **Provenance**: every run directory contains `config.yaml` (fully resolved) and `metadata.json` (timestamp, command line, git revision, Python / package versions, platform, CUDA availability).

Commands below assume the repository root as working directory (or an editable install, in which case the `ctrl-*` console scripts are equivalent to `python -m …`).

---

## 1. Sine-regression benchmark

```bash
python -m continual.run --config configs/continual/default.yaml --run-name reference
# subset of experiments / quick variant
python -m continual.run --config configs/continual/quick.yaml --experiments maml ewc
# domain-incremental EWC with online Fisher
python -m continual.run --experiments ewc --cl-scenario domain --ewc-mode online --ewc-online-gamma 0.9
```

| Experiment | Protocol | Seeds | Output |
|---|---|---|---|
| `transfer` | 4 (source, target) pairs × {scratch, frozen, finetune}, 5 000 Adam steps each | 3 | `transfer/transfer_curves.png`, `curves_*.csv`, `summary.json` |
| `maml` | 10 000 meta-iterations (second-order), evaluated on 100 held-out tasks with 10-shot support sets, 0–10 SGD steps | 3 | `maml/few_shot_adaptation.png`, `meta_training.png`, `qualitative.png`, `maml_init_seed*.pt` |
| `ewc` | 8 tasks × 3 000 steps, fine-tune vs EWC (λ = 100), task-incremental | 3 | `ewc/accuracy_matrix_{finetune,ewc}.png`, `task1_retention.png`, `R_*.csv` |

A consolidated `RESULTS.md` and `summary.json` are written to the run root. Runtime of the reference configuration: roughly 10–15 min on a CPU.

---

## 2. Meta-reinforcement learning

```bash
# Reptile and the multi-task control on PuckWorld, one seed each
python -m meta.train --config configs/meta/puckworld.yaml --algorithm reptile   --seed 0
python -m meta.train --config configs/meta/puckworld.yaml --algorithm multitask --seed 0

# Held-out comparison, pooling seeds by label
python -m meta.evaluate \
    --checkpoint reptile=outputs/meta/puckworld/reptile_seed0/policy.pt \
    --checkpoint multitask=outputs/meta/puckworld/multitask_seed0/policy.pt \
    --include-random --tasks 32 --adapt-steps 5 --output outputs/meta/puckworld/comparison

# Full study (3 seeds, Reptile vs multitask vs random)
bash scripts/reproduce_meta.sh puckworld 3
```

**Protocol.** Meta-training samples tasks from the *train* split only. Every `eval_every` iterations the current initialisation is adapted on `eval_tasks` *test-split* tasks and the post-adaptation return is logged (`history.csv`, `training.png`). At the end of training, the learned initialisation and a random initialisation are adapted on `final_eval_tasks` held-out tasks with identical episode seeds (`adaptation.png`, `adaptation.csv`, `adaptation_summary.json`).

| Config | Task family | Iterations | Notes |
|---|---|---|---|
| `configs/meta/snake.yaml` | 8×8 Snake, permuted controls | 500 | vector observations, MLP policy |
| `configs/meta/puckworld.yaml` | PuckWorld, permuted controls + dynamics | 500 | vector observations, MLP policy |
| `configs/meta/pong.yaml` | ALE Pong modes / difficulties | 200 | 4×40×40 stacks, CNN policy; expensive, GPU recommended |
| `configs/meta/smoke.yaml` | Snake | 2 | CI smoke test |

Checkpoints: `checkpoint.pt` (periodic snapshot of the meta-initialisation with its config and iteration) and `policy.pt` (final weights + config, consumed by `meta.evaluate`).

---

## 3. Transfer learning with PPO

```bash
# 1) Source policy
python -m transfer.train --config configs/transfer/snake.yaml --run-name source

# 2) Target: scratch, encoder transfer, frozen encoder
python -m transfer.train --config configs/transfer/puckworld.yaml --seed 0
python -m transfer.train --config configs/transfer/puckworld.yaml --seed 0 \
    --init-from outputs/transfer/snake/source/final_model.zip
python -m transfer.train --config configs/transfer/puckworld.yaml --seed 0 \
    --init-from outputs/transfer/snake/source/final_model.zip --freeze-encoder

# 3) Analysis across seeds
python -m transfer.analyze \
    --run "scratch=outputs/transfer/puckworld/scratch_seed*" \
    --run "encoder=outputs/transfer/puckworld/transfer-encoder_seed*" \
    --run "frozen=outputs/transfer/puckworld/transfer-encoder-frozen_seed*" \
    --baseline scratch --output outputs/transfer/analysis_snake_to_puckworld

# Roll out a trained policy (optionally recording video with the `video` extra)
python -m transfer.evaluate --model outputs/transfer/puckworld/scratch_seed0/final_model.zip --episodes 10

# Full study
bash scripts/reproduce_transfer.sh snake puckworld 3
bash scripts/reproduce_transfer.sh snake pong 3
```

Per-run outputs: `final_model.zip`, `best/best_model.zip`, `eval/evaluations.npz` (including the t = 0 evaluation), `episodes.csv`, `training_returns.png`, `transfer.json` (transferred / frozen tensor counts, trainable parameters), `summary.json`, and TensorBoard logs under `tb/` when the `tensorboard` extra is installed.

Analysis outputs: `learning_curves.png` (mean ± 95 % CI), `metrics.json` and `metrics.md` with jumpstart, asymptotic return, normalised AUC and transfer ratio.

| Config | Environment | Budget |
|---|---|---|
| `configs/transfer/pong.yaml` | `ALE/Pong-v5` (sticky actions), 3 actions | 2 M steps |
| `configs/transfer/snake.yaml` | Snake image env | 500 k steps |
| `configs/transfer/puckworld.yaml` | PuckWorld image env, continuous control, dense reward | 300 k steps |
| `configs/transfer/smoke.yaml` | any | 1 024 steps, CI smoke test |

---

## 4. Compute budget

| Study | Approx. cost |
|---|---|
| Sine benchmark (reference) | 10–15 CPU-minutes |
| Meta-RL, PuckWorld/Snake, 3 seeds × 2 algorithms | a few CPU-hours |
| Transfer, Snake → PuckWorld, 3 seeds × 3 protocols | several GPU-hours (or 1–2 CPU-days) |
| Transfer involving Pong | multiple GPU-days |

The RL studies have not been run at full scale in this repository; the scripts above are the canonical way to produce those numbers.
