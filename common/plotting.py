"""Publication-style plots with mean ± 95% CI bands across seeds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.stats import mean_ci  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "font.size": 10,
    }
)


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average that preserves length (uses a shrinking window at the start)."""
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or y.shape[-1] < 2:
        return y
    kernel_sum = np.cumsum(y, axis=-1)
    out = np.empty_like(y)
    for i in range(y.shape[-1]):
        lo = max(0, i - window + 1)
        total = kernel_sum[..., i] - (kernel_sum[..., lo - 1] if lo > 0 else 0.0)
        out[..., i] = total / (i - lo + 1)
    return out


def plot_curves(
    curves: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    x: np.ndarray | Mapping[str, np.ndarray] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    logy: bool = False,
    smoothing: int = 1,
    figsize: tuple[float, float] = (7.0, 4.2),
) -> Path:
    """Plot ``{label: array[n_seeds, T]}`` as mean lines with 95% CI bands.

    With ``logy=True`` statistics are computed in log10 space, i.e. the line is the geometric
    mean and the band a multiplicative 95% CI, so the band never crosses zero on a log axis.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, arr in curves.items():
        arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        arr = smooth(arr, smoothing)
        xs = x[label] if isinstance(x, Mapping) else x
        xs = np.arange(arr.shape[1]) if xs is None else np.asarray(xs)
        if logy:
            mean, lo, hi = (10.0**v for v in mean_ci(np.log10(np.maximum(arr, 1e-12)), axis=0))
        else:
            mean, lo, hi = mean_ci(arr, axis=0)
        (line,) = ax.plot(xs, mean, label=f"{label} (n={arr.shape[0]})", linewidth=1.6)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color(), linewidth=0)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heatmap(
    matrix: np.ndarray,
    path: str | Path,
    *,
    title: str = "",
    xlabels: Sequence[str] | None = None,
    ylabels: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    cbar_label: str = "",
    annotate: bool = True,
    log_color: bool = False,
) -> Path:
    """Heatmap with NaN cells left blank (e.g. the untrained upper triangle of a CL matrix)."""
    from matplotlib.colors import LogNorm

    m = np.asarray(matrix, dtype=np.float64)
    masked = np.ma.masked_invalid(m)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * m.shape[1], 0.8 + 0.6 * m.shape[0]))
    norm = None
    if log_color and np.nanmin(m) > 0:
        norm = LogNorm(vmin=np.nanmin(m), vmax=np.nanmax(m))
    im = ax.imshow(masked, cmap="viridis", aspect="auto", norm=norm)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=cbar_label)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), labels=list(xlabels), rotation=45, ha="right")
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), labels=list(ylabels))
    if annotate:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2g}", ha="center", va="center", fontsize=7, color="w")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
