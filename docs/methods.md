# Methods

This document specifies the algorithms, objectives and evaluation metrics implemented in CTRL. Notation follows the cited papers where possible; every equation below maps to a concrete function in the code base (module paths in brackets).

---

## 1. Task families

All three paradigms are evaluated on *task distributions* rather than single environments, so that "transfer", "adaptation" and "forgetting" are measured on tasks the learner has not seen before.

| Family | Observation (vector / image) | Task variation | Train / test split |
|---|---|---|---|
| **Snake** | 8×8 grid, encoded `empty 0 · body 0.5 · head 1 · food −1` / `(1, 84, 84)` uint8 | action permutation π ∈ S₄, food reward ∈ [0.5, 2] | 18 / 6 permutations (disjoint) |
| **PuckWorld** | `(pos, vel, target)` ∈ ℝ⁶ / `(1, 84, 84)` uint8, continuous thrust | action permutation, thrust ∈ [0.08, 0.25], damping ∈ [0.85, 0.95] | 18 / 6 permutations (disjoint) |
| **Pong** | 4×40×40 float stack / `(1, 84, 84)` uint8 | ALE game mode m ∈ {0, 1}, difficulty d | d ∈ {0, 1, 2} train, d = 3 test |
| **Sine** | x ∈ [−5, 5] | amplitude A ∈ [0.1, 5], phase φ ∈ [0, π] | fresh tasks sampled at evaluation |

[`envs/tasks.py`, `envs/registry.py`, `continual/data.py`]

Action permutations are a standard way to create a family of MDPs with *identical* state distributions but *different* optimal policies: a policy that memorises "press 0 to go up" fails, while one that has learned to *infer* the control mapping from a handful of episodes adapts quickly. Held-out permutations are never seen during meta-training.

---

## 2. Transfer learning (PPO)

### 2.1 Learner

Proximal Policy Optimisation (Schulman et al., 2017) with the clipped surrogate

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big],
\qquad r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)},
$$

generalised advantage estimation (λ = 0.95), value loss coefficient 0.5, entropy bonus 0.01 and a linearly annealed learning rate. Observations pass through `VecFrameStack(4, channels_first)` so every environment exposes a `(4, 84, 84)` tensor to a shared **Nature-CNN** encoder (Mnih et al., 2015) with a 512-d projection [`transfer/extractor.py`].

Because the encoder is architecture-identical across Snake, PuckWorld and Pong, its weights can be transplanted between tasks even though the action spaces differ (Discrete(4), Box(2), Discrete(3)).

### 2.2 Transfer protocols

Let θ = (θ_enc, θ_mlp, θ_π, θ_V) be the actor-critic parameters and θ^S the parameters learned on the source task.

| Protocol | Initialisation on target | Trainable |
|---|---|---|
| `scratch` | random | all |
| `transfer-encoder` | θ_enc ← θ^S_enc | all |
| `transfer-encoder-frozen` | θ_enc ← θ^S_enc | θ_mlp, θ_π, θ_V |
| `transfer-encoder_mlp` | θ_enc, θ_mlp ← θ^S | all |

Heads (θ_π, θ_V) are always re-initialised because the target action space generally differs. Weight transfer is strict: parameters are matched by name, shape mismatches raise, and an empty match raises [`transfer/weights.py`]. Frozen encoders are verified bit-identical after training in the test suite.

### 2.3 Metrics (Taylor & Stone, 2009)

With G(t) the mean deterministic evaluation return at environment step t (evaluated at t = 0 and every `eval_freq` steps):

- **Jumpstart** J = G(0) — the zero-shot benefit of the transferred representation.
- **Asymptotic performance** = mean of the last three evaluations.
- **Normalised AUC** = (1 / (t_max − t_0)) ∫ G(t) dt (trapezoidal), i.e. the mean height of the learning curve.
- **Transfer ratio** = (AUC − AUC_scratch) / |AUC_scratch|.

All metrics are computed per seed and aggregated as mean ± 95 % Student-t confidence interval [`transfer/analyze.py`, `common/stats.py`].

---

## 3. Meta-reinforcement learning

### 3.1 Inner loop — REINFORCE

For a task 𝒯 the inner learner performs k policy-gradient updates, each on `episodes_per_step` fresh episodes:

$$
\nabla_\theta J \approx \frac{1}{\sum_i T_i}\sum_{i}\sum_{t=0}^{T_i-1} \nabla_\theta \log \pi_\theta(a_t^i\mid s_t^i)\,\tilde G_t^i \;+\; \beta\,\nabla_\theta \mathcal H[\pi_\theta],
\qquad G_t = \sum_{t'\ge t}\gamma^{t'-t} r_{t'} ,
$$

where G̃ are reward-to-go returns standardised across the *whole batch* of episodes (population standard deviation, so a single one-step episode yields a zero, not a NaN, advantage). Gradients are clipped to norm 1 [`meta/reinforce.py`].

### 3.2 Outer loop — Reptile

Reptile (Nichol et al., 2018) is a first-order meta-learner. Per meta-iteration, sample a batch of n tasks, adapt a copy of φ on each for k steps to obtain θ̃_i, then move the initialisation towards their mean:

$$
\phi \leftarrow \phi + \epsilon_\tau \Big(\frac{1}{n}\sum_{i=1}^{n}\tilde\theta_i - \phi\Big),
\qquad \epsilon_\tau = \epsilon_0\,(1 - \tau/T_\text{meta}).
$$

For k > 1 the expected update contains a term that maximises the inner product between gradients of different minibatches of the same task — i.e. it directly optimises for *fast within-task generalisation*, which is exactly the property probed by held-out action permutations [`meta/reptile.py`, `meta/train.py`].

### 3.3 Baselines

- **Multi-task pretraining** (`--algorithm multitask`): a single policy trained with Adam on the pooled REINFORCE loss over the same task batches. It sees exactly the same data as Reptile but does not optimise for adaptability — the standard control for meta-learning claims.
- **Random initialisation**: no pretraining.

### 3.4 Evaluation

For each of `final_eval_tasks` *held-out* tasks, every initialisation is adapted with the *same* inner-loop budget and the same episode seeds, and the deterministic return is measured before adaptation and after each update. We report the adaptation curve (mean ± 95 % CI over tasks and seeds) and the post-adaptation return [`meta/evaluate.py`].

---

## 4. Sine-regression benchmark

A controlled supervised setting in which each paradigm can be evaluated precisely and cheaply. The regressor is a 1 → 40 → 40 → 1 ReLU MLP (optionally multi-headed) [`continual/models.py`].

### 4.1 Supervised transfer

For each (source, target) pair: train on the source for N steps, then train on the target under three regimes — `scratch`, `frozen` (trunk frozen, output head trained) and `finetune` (all parameters trained from source weights). Per-pair metrics are jumpstart MSE, final MSE (mean of last 5 % of steps) and the AUC of log₁₀ MSE [`continual/transfer.py`].

### 4.2 MAML

Model-Agnostic Meta-Learning (Finn et al., 2017), sine protocol: K = 10 support points, one inner SGD step with α = 0.01, meta-batch 10, Adam meta-optimiser (β = 10⁻³):

$$
\min_\phi\ \mathbb{E}_{\mathcal T}\Big[\mathcal L^{\text{query}}_{\mathcal T}\big(\phi - \alpha\nabla_\phi \mathcal L^{\text{support}}_{\mathcal T}(\phi)\big)\Big].
$$

The implementation is fully functional (`torch.func.functional_call`), so the meta-gradient flows through the inner update, including the second-order term ∇²ℒ; `maml_first_order: true` gives FOMAML by detaching the inner gradient [`continual/maml.py`].

> **Bug fixed.** The original implementation trained a `load_state_dict` copy of the meta-model in place and back-propagated the query loss into that copy, so the meta-model never received a gradient and `opt_meta.step()` was a no-op. A regression test asserts that every meta-parameter changes after one meta-step.

Three initialisations are compared on 100 held-out tasks with shared support/query sets: **MAML**, **pretrained** (multi-task regression on the same task distribution — the canonical non-meta baseline, which collapses towards the mean function ≈ 0) and **scratch**.

### 4.3 Continual learning with EWC

Tasks T₁ … T₈ are learned sequentially, 3 000 Adam steps each. Two scenarios are supported: **task-incremental** (one output head per task, task identity known at test time) and **domain-incremental** (single shared head).

Elastic Weight Consolidation (Kirkpatrick et al., 2017) adds a quadratic penalty anchored at the parameters θ*_j learned on each previous task:

$$
\mathcal L(\theta) = \mathcal L_{\mathcal T_i}(\theta) + \frac{\lambda}{2}\sum_{j<i}\sum_k F_{j,k}\,(\theta_k - \theta^*_{j,k})^2 .
$$

**Fisher information.** For a Gaussian likelihood y ~ 𝒩(f_θ(x), σ²), the Fisher is F = E_x[J(x)ᵀJ(x)]/σ² with J = ∂f_θ/∂θ. We compute its diagonal exactly from *per-sample* Jacobians via `vmap(grad(·))`, i.e. the **true** Fisher. The original code squared mini-batch loss gradients — the *empirical* Fisher, which depends on the labels and on the batch size and is known to be a biased estimator (Kunstner et al., 2019).

**Online EWC** (Schwarz et al., 2018) keeps a single running anchor and Fisher, F ← γF + F_i, which bounds memory to O(|θ|) regardless of the number of tasks.

> **Bug fixed.** The original `run_ewc` overwrote the regulariser after each task, so only the most recent task was ever protected.

### 4.4 Continual-learning metrics

With R ∈ ℝ^{T×T}, R_{i,j} = test MSE on task j after training through task i (lower is better) [`continual/metrics.py`]:

| Metric | Definition | Interpretation |
|---|---|---|
| Average final MSE | (1/T) Σ_j R_{T,j} | overall performance at the end |
| Average learning MSE | (1/T) Σ_j R_{j,j} | plasticity |
| Backward transfer (BWT) | (1/(T−1)) Σ_{j<T} (R_{T,j} − R_{j,j}) | > 0 ⇒ forgetting (Lopez-Paz & Ranzato, 2017) |
| Forgetting | (1/(T−1)) Σ_{j<T} (R_{T,j} − min_{l<T} R_{l,j}) | worst-case loss since best (Chaudhry et al., 2018) |

---

## 5. Statistics and reproducibility

- Every stochastic component draws from an explicitly seeded `numpy.random.Generator` or `torch` generator; environments are seeded through the Gymnasium `reset(seed=…)` API.
- Summary statistics are reported as mean ± half-width of the 95 % Student-t interval over seeds (or tasks × seeds where stated).
- Each run directory contains the fully resolved `config.yaml` and a `metadata.json` with git revision, package versions, device and host information.

---

## References

- Chaudhry, A. et al. (2018). *Riemannian Walk for Incremental Learning.* ECCV.
- Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML.
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS.
- Kunstner, F., Hennig, P., & Balles, L. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent.* NeurIPS.
- Lopez-Paz, D., & Ranzato, M. (2017). *Gradient Episodic Memory for Continual Learning.* NeurIPS.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Nichol, A., Achiam, J., & Schulman, J. (2018). *On First-Order Meta-Learning Algorithms.* arXiv:1803.02999.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schwarz, J. et al. (2018). *Progress & Compress: A scalable framework for continual learning.* ICML.
- Taylor, M. E., & Stone, P. (2009). *Transfer Learning for Reinforcement Learning Domains: A Survey.* JMLR.
- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning.
