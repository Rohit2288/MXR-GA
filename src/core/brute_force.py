"""Exhaustive-enumeration exact reference (tasks 0.6, 0.7).

Independent of every solver: enumerates all 2^n subsets with numpy doubling
(exact int64 arithmetic), finds a subset maximizing mu, and rescored it via
the single scorer. Used as the ground-truth reference in the solver contract
tests and the Suite U re-verification script. Feasible up to n ~ 25.
"""

from __future__ import annotations

import numpy as np

from src.core.evaluate import Evaluation, evaluate
from src.core.instance import Instance

MAX_N = 25  # 2^25 int64 sums = 256 MiB; hard stop to avoid accidental blowups


def all_subset_sums(items) -> np.ndarray:
    """sums[k] = sum of items[i] for each bit i set in k (exact, int64)."""
    sums = np.zeros(1, dtype=np.int64)
    for a in items:
        sums = np.concatenate([sums, sums + np.int64(a)])
    return sums


def _mu_vectorized(sums: np.ndarray, T_L: float, T_M: float, T_R: float) -> np.ndarray:
    x = sums.astype(np.float64)
    mu = np.zeros_like(x)
    dL, dR = T_M - T_L, T_R - T_M
    left = (x >= T_L) & (x < T_M)
    if dL > 0:
        mu[left] = (x[left] - T_L) / dL
    else:
        mu[left] = 1.0
    right = (x > T_M) & (x <= T_R)
    if dR > 0:
        mu[right] = (T_R - x[right]) / dR
    else:
        mu[right] = 1.0
    mu[x == T_M] = 1.0
    return mu


def exact_optimum(instance: Instance) -> Evaluation:
    """Return an Evaluation of a subset achieving the true maximum mu.

    The optimal subset is located by exhaustive enumeration, then rescored by
    the single scorer (so the reported mu is scalar-arithmetic, not the
    vectorized intermediate).
    """
    n = instance.n
    if n > MAX_N:
        raise ValueError(f"n={n} exceeds enumeration limit {MAX_N}")
    sums = all_subset_sums(instance.items)
    t = instance.target
    mu = _mu_vectorized(sums, t.T_L, t.T_M, t.T_R)
    k = int(np.argmax(mu))
    subset = [i for i in range(n) if (k >> i) & 1]
    ev = evaluate(instance, subset)
    # Cross-check the two mu computations agree at the optimum.
    assert abs(ev.mu - float(mu[k])) < 1e-12
    return ev
