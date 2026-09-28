"""Entry point for the sine-regression benchmark.

Examples
--------
    python -m continual.run                                   # all experiments, default config
    python -m continual.run --experiments maml ewc --n-seeds 5
    python -m continual.run --config configs/continual/quick.yaml   # ~1 min smoke test
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from common.config import parse_config
from common.io import create_run_dir, ensure_dir, save_run_metadata, write_csv, write_json
from common.logger import get_logger
from common.plotting import plot_curves, plot_heatmap, plt
from common.stats import summarize
from common.torch_utils import resolve_device
from continual.config import ContinualConfig
from continual.data import SineTask, sample_task
from continual.ewc import METHODS as CL_METHODS
from continual.ewc import run_sequence
from continual.maml import adapt_functional, run_maml_seed
from continual.metrics import continual_metrics
from continual.transfer import MODES as TRANSFER_MODES
from continual.transfer import run_transfer_pair, transfer_metrics

EXPERIMENTS = ("transfer", "maml", "ewc")
log = get_logger("continual")


# ------------------------------------------------------------------------ transfer
def experiment_transfer(cfg: ContinualConfig, device, out: Path) -> dict:
    out = ensure_dir(out / "transfer")
    curves: dict[str, list[np.ndarray]] = {m: [] for m in TRANSFER_MODES}
    per_run: dict[str, list[dict]] = {m: [] for m in TRANSFER_MODES}
    for seed in range(cfg.n_seeds):
        for src, tgt in cfg.transfer_pairs:
            res = run_transfer_pair(SineTask(*src), SineTask(*tgt), seed, cfg, device, out)
            for m in TRANSFER_MODES:
                curves[m].append(res[m].curve)
                per_run[m].append(transfer_metrics(res[m].curve))
        log.info("transfer: seed %d done", seed)

    plot_curves(
        {m: np.stack(c) for m, c in curves.items()},
        out / "transfer_curves.png",
        title="Sine transfer: target-task learning curves",
        xlabel="Gradient step on target task",
        ylabel="MSE (geometric mean, log scale)",
        logy=True,
        smoothing=25,
    )
    summary = {m: {k: summarize([r[k] for r in runs]) for k in runs[0]} for m, runs in per_run.items()}
    write_json(out / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------- MAML
def _plot_maml_qualitative(models, cfg: ContinualConfig, device, path: Path) -> None:
    rng = np.random.default_rng(12345)
    task = sample_task(rng, cfg.amplitude_range, cfg.phase_range)
    xs, ys = task.sample(cfg.maml_k_shot, rng, device, cfg.x_range)
    xg, yg = task.grid(cfg.eval_points, device, cfg.x_range)
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    for ax, (label, model) in zip(np.atleast_1d(axes), models.items(), strict=False):
        ax.plot(xg.cpu(), yg.cpu(), "k-", lw=2, label="ground truth")
        ax.scatter(xs.cpu(), ys.cpu(), c="k", marker="^", zorder=5, label=f"{cfg.maml_k_shot}-shot support")
        params = {n: p.detach() for n, p in model.named_parameters()}
        for steps, style in ((0, ":"), (1, "--"), (cfg.maml_eval_steps, "-")):
            p = {n: v.clone().requires_grad_(True) for n, v in params.items()}
            fast = adapt_functional(model, p, xs, ys, steps, cfg.maml_inner_lr, first_order=True)
            with torch.no_grad():
                pred = torch.func.functional_call(model, fast, (xg,))
            ax.plot(xg.cpu(), pred.cpu(), style, label=f"{steps} step(s)")
        ax.set_title(label)
        ax.set_xlabel("x")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f"Few-shot adaptation on an unseen task ({task.label})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def experiment_maml(cfg: ContinualConfig, device, out: Path) -> dict:
    out = ensure_dir(out / "maml")
    all_curves: dict[str, list[np.ndarray]] = {}
    histories = []
    models = None
    for seed in range(cfg.n_seeds):
        t0 = time.time()
        models, curves, history = run_maml_seed(cfg, seed, device, out)
        histories.append(history)
        for k, v in curves.items():
            all_curves.setdefault(k, []).append(v.mean(axis=0))  # mean over eval tasks
        log.info(
            "maml: seed %d done in %.0fs | post-%d-step MSE: %s",
            seed,
            time.time() - t0,
            cfg.maml_eval_steps,
            ", ".join(f"{k}={v.mean(0)[-1]:.3f}" for k, v in curves.items()),
        )

    stacked = {k: np.stack(v) for k, v in all_curves.items()}
    plot_curves(
        stacked,
        out / "few_shot_adaptation.png",
        title=f"{cfg.maml_k_shot}-shot sine regression on held-out tasks",
        xlabel="Gradient steps on support set",
        ylabel="Test MSE",
    )
    plot_curves(
        {"meta-train loss": np.stack(histories)},
        out / "meta_training.png",
        title="MAML meta-training (post-adaptation query loss)",
        xlabel="Meta-iteration",
        ylabel="MSE (log scale)",
        logy=True,
        smoothing=100,
    )
    if models is not None:
        _plot_maml_qualitative(models, cfg, device, out / "qualitative.png")
    summary = {
        k: {f"mse_after_{s}_steps": summarize(v[:, s]) for s in (0, 1, cfg.maml_eval_steps)}
        for k, v in stacked.items()
    }
    write_json(out / "summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- EWC
def experiment_ewc(cfg: ContinualConfig, device, out: Path) -> dict:
    out = ensure_dir(out / "ewc")
    n = len(cfg.task_params)
    matrices: dict[str, list[np.ndarray]] = {m: [] for m in CL_METHODS}
    traces: dict[str, list[np.ndarray]] = {m: [] for m in CL_METHODS}
    for seed in range(cfg.n_seeds):
        for method in CL_METHODS:
            res = run_sequence(cfg, method, seed, device)
            matrices[method].append(res["R"])
            traces[method].append(res["trace"])
            write_csv(out / f"R_{method}_seed{seed}.csv", res["R"].tolist())
        log.info("ewc: seed %d done", seed)

    labels = [f"T{i + 1}" for i in range(n)]
    summary: dict[str, dict] = {}
    for method in CL_METHODS:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_R = np.nanmean(np.stack(matrices[method]), axis=0)
        plot_heatmap(
            mean_R,
            out / f"accuracy_matrix_{method}.png",
            title=f"{method}: test MSE (mean over {cfg.n_seeds} seeds)",
            xlabels=labels,
            ylabels=[f"after {lbl}" for lbl in labels],
            xlabel="Evaluated task",
            ylabel="Training stage",
            cbar_label="MSE",
            log_color=True,
        )
        per_seed = [continual_metrics(R) for R in matrices[method]]
        summary[method] = {k: summarize([m[k] for m in per_seed]) for k in per_seed[0]}

    # Retention of task 1 across the sequence
    t1 = {m: np.stack([R[:, 0] for R in matrices[m]]) for m in CL_METHODS}
    plot_curves(
        t1,
        out / "task1_retention.png",
        x=np.arange(1, n + 1),
        title="Retention of task T1 during sequential training",
        xlabel="Tasks trained so far",
        ylabel="Test MSE on T1 (log scale)",
        logy=True,
    )
    write_json(out / "summary.json", summary)
    return summary


# -------------------------------------------------------------------------- report
def _write_report(results: dict, cfg: ContinualConfig, out: Path) -> None:
    lines = ["# Sine benchmark results", "", f"Seeds: {cfg.n_seeds} · values are mean ± 95% CI.", ""]

    def cell(s: dict) -> str:
        half = (s["ci_high"] - s["ci_low"]) / 2
        return f"{s['mean']:.4g} ± {half:.2g}"

    if "transfer" in results:
        lines += [
            "## Transfer (target task)",
            "",
            "| Mode | Jumpstart MSE | Final MSE | AUC log10 MSE |",
            "|---|---|---|---|",
        ]
        for m, s in results["transfer"].items():
            lines.append(
                f"| {m} | {cell(s['jumpstart_mse'])} | {cell(s['final_mse'])} | {cell(s['auc_log_mse'])} |"
            )
        lines.append("")
    if "maml" in results:
        k = cfg.maml_eval_steps
        lines += ["## Few-shot adaptation", "", f"| Init | MSE @0 | MSE @1 | MSE @{k} |", "|---|---|---|---|"]
        for m, s in results["maml"].items():
            lines.append(
                f"| {m} | {cell(s['mse_after_0_steps'])} | {cell(s['mse_after_1_steps'])} | {cell(s[f'mse_after_{k}_steps'])} |"
            )
        lines.append("")
    if "ewc" in results:
        lines += [
            f"## Continual learning ({cfg.cl_scenario}-incremental, λ={cfg.ewc_lambda})",
            "",
            "| Method | Avg final MSE | Avg learning MSE | BWT (↓) | Forgetting (↓) |",
            "|---|---|---|---|---|",
        ]
        for m, s in results["ewc"].items():
            lines.append(
                f"| {m} | {cell(s['avg_final_mse'])} | {cell(s['avg_learning_mse'])} | {cell(s['bwt'])} | {cell(s['forgetting'])} |"
            )
        lines.append("")
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _extra(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[*EXPERIMENTS, "all"],
        default=["all"],
        help="Subset of experiments to run.",
    )


def main(argv: Sequence[str] | None = None) -> Path:
    cfg, args = parse_config(ContinualConfig, "Sine-regression benchmark: transfer, MAML, EWC.", argv, _extra)
    selected = EXPERIMENTS if "all" in args.experiments else tuple(args.experiments)
    device = resolve_device(cfg.device)
    out = create_run_dir(cfg.output_root, "continual", run_name=cfg.run_name)
    save_run_metadata(out, cfg)
    get_logger("continual", out / "run.log")
    log.info("device=%s | experiments=%s | output=%s", device, ",".join(selected), out)

    runners = {"transfer": experiment_transfer, "maml": experiment_maml, "ewc": experiment_ewc}
    results = {}
    for name in selected:
        t0 = time.time()
        results[name] = runners[name](cfg, device, out)
        log.info("%s finished in %.1fs", name, time.time() - t0)

    write_json(out / "summary.json", results)
    _write_report(results, cfg, out)
    log.info("results written to %s", out)
    return out


if __name__ == "__main__":
    main()
