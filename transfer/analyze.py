"""Aggregate transfer runs into learning curves and standard transfer metrics.

Metrics (Taylor & Stone, 2009), computed per run from the evaluation curve and then
aggregated across seeds with 95% CIs:

* ``jumpstart``   - evaluation return at t = 0
* ``asymptotic``  - mean of the last three evaluations
* ``auc``         - area under the curve normalised by the time span (mean height)
* ``transfer_ratio`` - (AUC - AUC_baseline) / |AUC_baseline|   (relative to --baseline)

    python -m transfer.analyze \
        --run "scratch=outputs/transfer/puckworld/scratch_seed*" \
        --run "encoder=outputs/transfer/puckworld/transfer-encoder_seed*" \
        --run "frozen=outputs/transfer/puckworld/transfer-encoder-frozen_seed*" \
        --baseline scratch --output outputs/transfer/analysis_puckworld
"""

from __future__ import annotations

import argparse
import glob
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from common.io import ensure_dir, write_json
from common.logger import get_logger
from common.plotting import plot_curves
from common.stats import normalized_auc, summarize

log = get_logger("transfer.analyze")


def load_eval_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(run_dir) / "eval" / "evaluations.npz")
    return data["timesteps"].astype(np.float64), data["results"].mean(axis=1).astype(np.float64)


def run_metrics(t: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "jumpstart": float(y[0]),
        "asymptotic": float(y[-3:].mean()),
        "auc": normalized_auc(y, t),
    }


def _expand(spec: str) -> tuple[str, list[Path]]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--run must be label=glob")
    label, pattern = spec.split("=", 1)
    paths = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / "eval" / "evaluations.npz").exists())
    if not paths:
        raise SystemExit(f"No runs with eval/evaluations.npz match '{pattern}'")
    return label, paths


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Compare transfer runs (learning curves + transfer metrics)."
    )
    parser.add_argument("--run", action="append", required=True, help="label=glob (repeatable)")
    parser.add_argument("--baseline", default=None, help="Label used as the no-transfer reference.")
    parser.add_argument("--output", type=Path, default=Path("outputs/transfer/analysis"))
    parser.add_argument("--title", default="Transfer learning curves")
    args = parser.parse_args(argv)
    out = ensure_dir(args.output)

    groups = dict(_expand(s) for s in args.run)
    curves, grids, metrics = {}, {}, {}
    for label, paths in groups.items():
        loaded = [load_eval_curve(p) for p in paths]
        horizon = min(t[-1] for t, _ in loaded)
        grid = next(t for t, _ in loaded)
        grid = grid[grid <= horizon]
        curves[label] = np.stack([np.interp(grid, t, y) for t, y in loaded])
        grids[label] = grid
        metrics[label] = [run_metrics(grid, row) for row in curves[label]]
        log.info("%s: %d run(s)", label, len(paths))

    summary: dict[str, dict] = {
        label: {k: summarize([m[k] for m in ms]) for k in ms[0]} for label, ms in metrics.items()
    }
    if args.baseline:
        if args.baseline not in metrics:
            raise SystemExit(f"--baseline '{args.baseline}' is not one of the run labels")
        base_auc = summary[args.baseline]["auc"]["mean"]
        for label, ms in metrics.items():
            ratios = [(m["auc"] - base_auc) / (abs(base_auc) + 1e-8) for m in ms]
            summary[label]["transfer_ratio"] = summarize(ratios)

    plot_curves(
        curves,
        out / "learning_curves.png",
        x=grids,
        title=args.title,
        xlabel="Environment steps",
        ylabel="Evaluation return",
    )
    write_json(out / "metrics.json", summary)

    cols = ["jumpstart", "asymptotic", "auc"] + (["transfer_ratio"] if args.baseline else [])
    lines = ["| Condition | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for label, s in summary.items():
        cells = [f"{s[c]['mean']:.3f} ± {(s[c]['ci_high'] - s[c]['ci_low']) / 2:.3f}" for c in cols]
        lines.append(f"| {label} | {s['jumpstart']['n']} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (out / "metrics.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return summary


if __name__ == "__main__":
    main()
