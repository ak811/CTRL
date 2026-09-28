"""Continual-learning metrics for error-based (lower-is-better) accuracy matrices.

With ``R[i, j]`` the test error on task ``j`` after training through task ``i``
(Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018):

* ``avg_final_mse``   = mean_j R[T-1, j]
* ``avg_learning_mse``= mean_j R[j, j]                (plasticity)
* ``bwt``             = mean_{j<T-1} R[T-1, j] - R[j, j]   (> 0 => forgetting)
* ``forgetting``      = mean_{j<T-1} R[T-1, j] - min_{j<=l<T-1} R[l, j]
"""

from __future__ import annotations

import numpy as np


def continual_metrics(R: np.ndarray) -> dict[str, float]:
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix")
    T = R.shape[0]
    final = R[-1]
    diag = np.diag(R)
    out = {
        "avg_final_mse": float(np.nanmean(final)),
        "avg_learning_mse": float(np.nanmean(diag)),
    }
    if T > 1:
        out["bwt"] = float(np.mean(final[:-1] - diag[:-1]))
        best = np.array([np.nanmin(R[j : T - 1, j]) for j in range(T - 1)])
        out["forgetting"] = float(np.mean(final[:-1] - best))
    else:
        out["bwt"] = 0.0
        out["forgetting"] = 0.0
    return out
