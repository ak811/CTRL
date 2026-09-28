"""Training callbacks: per-episode statistics logged to CSV and plotted."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from common.io import write_csv
from common.plotting import plt, smooth


class EpisodeStatsCallback(BaseCallback):
    """Collects ``(timestep, return, length)`` for every finished training episode.

    Relies on the ``episode`` entry that ``VecMonitor`` injects into ``infos``.
    Writes ``episodes.csv`` and ``training_returns.png`` at the end of training.
    """

    def __init__(self, out_dir: str | Path, smoothing: int = 50, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.smoothing = smoothing
        self.records: list[tuple[int, float, int]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.records.append((int(self.num_timesteps), float(ep["r"]), int(ep["l"])))
        return True

    def _on_training_end(self) -> None:
        if not self.records:
            return
        write_csv(self.out_dir / "episodes.csv", self.records, ["timestep", "return", "length"])
        arr = np.asarray(self.records, dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(arr[:, 0], arr[:, 1], alpha=0.25, lw=0.8, label="episode return")
        ax.plot(arr[:, 0], smooth(arr[:, 1], self.smoothing), lw=1.8, label=f"moving avg ({self.smoothing})")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Return")
        ax.set_title("Training returns")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.out_dir / "training_returns.png")
        plt.close(fig)


# Backward-compatible name.
PlottingCallback = EpisodeStatsCallback
