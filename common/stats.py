"""Statistics for reporting results across seeds.

Confidence intervals use the Student-t distribution on the seed axis, which is the
appropriate small-sample choice for the 3-10 seeds typical of RL experiments.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

# Two-sided 95% Student-t critical values, df = 1..30.
_T95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def t_critical(n: int, confidence: float = 0.95) -> float:
    df = n - 1
    if df < 1:
        return float("nan")
    if abs(confidence - 0.95) < 1e-9 and df <= len(_T95):
        return _T95[df - 1]
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def mean_ci(
    x: np.ndarray, axis: int = 0, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mean, lower, upper)`` along ``axis``. With one sample the band collapses."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    mean = x.mean(axis=axis)
    if n < 2:
        return mean, mean.copy(), mean.copy()
    sem = x.std(axis=axis, ddof=1) / np.sqrt(n)
    half = t_critical(n, confidence) * sem
    return mean, mean - half, mean + half


def summarize(values: Any, confidence: float = 0.95) -> dict[str, float]:
    """Scalar summary (mean, std, CI, n) of a 1-D collection of per-seed values."""
    v = np.asarray(values, dtype=np.float64).ravel()
    mean, lo, hi = mean_ci(v, axis=0, confidence=confidence)
    return {
        "mean": float(mean),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(v.size),
    }


def format_mean_ci(values: Any, precision: int = 3) -> str:
    s = summarize(values)
    half = (s["ci_high"] - s["ci_low"]) / 2.0
    return f"{s['mean']:.{precision}f} ± {half:.{precision}f}"


def normalized_auc(y: np.ndarray, x: np.ndarray | None = None) -> float:
    """Area under a learning curve normalised by the x-range (i.e. the curve's mean height)."""
    y = np.asarray(y, dtype=np.float64)
    if y.size == 1:
        return float(y[0])
    x = np.arange(y.size, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    span = x[-1] - x[0]
    area = float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)
    return area / span if span > 0 else float(y.mean())
