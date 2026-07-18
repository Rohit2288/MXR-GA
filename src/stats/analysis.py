"""Statistical analysis pipeline (task 2.3; R4.2).

Per suite: Friedman omnibus test over the instance x solver mean-mu matrix;
if significant, pairwise Wilcoxon signed-rank tests against a reference
solver (or all pairs) with Bonferroni-Holm correction; Cliff's delta effect
sizes with standard magnitude labels. Effect sizes are reported alongside
every p-value — significance from trivial deltas at large n must not be
leaned on (working principle 4; R4 flagged this).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta of x over y in [-1, 1] (positive: x tends larger)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # O(n log n) via ranks: delta = 2*U/(n_x*n_y) - 1
    u = sps.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def cliffs_magnitude(delta: float) -> str:
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def holm_correction(pvals: Sequence[float]) -> List[float]:
    """Bonferroni-Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


@dataclass(frozen=True)
class PairwiseResult:
    solver_a: str
    solver_b: str
    p_raw: float
    p_holm: float
    delta: float           # Cliff's delta of a over b (paired samples)
    magnitude: str
    median_diff: float
    n_pairs: int


@dataclass(frozen=True)
class SuiteAnalysis:
    friedman_chi2: Optional[float]
    friedman_p: Optional[float]
    n_instances: int
    solvers: Tuple[str, ...]
    mean_ranks: Dict[str, float]
    pairwise: List[PairwiseResult]


def analyze_suite(matrix: Dict[str, Dict[str, float]],
                  reference: Optional[str] = None) -> SuiteAnalysis:
    """matrix[solver][instance_id] = per-instance score (e.g. seed-mean mu).

    Only instances present for EVERY solver are used (paired design). If
    `reference` is given, pairwise tests are reference-vs-others; otherwise
    all pairs.
    """
    solvers = tuple(sorted(matrix))
    common = sorted(set.intersection(*(set(matrix[s]) for s in solvers)))
    if len(common) < 2 or len(solvers) < 2:
        raise ValueError("need >=2 solvers and >=2 common instances")
    data = np.array([[matrix[s][i] for i in common] for s in solvers])

    # Friedman needs >=3 treatments.
    if len(solvers) >= 3:
        chi2, p = sps.friedmanchisquare(*data)
        friedman_chi2, friedman_p = float(chi2), float(p)
    else:
        friedman_chi2 = friedman_p = None

    # Mean ranks (higher mu -> better -> lower rank number).
    ranks = np.apply_along_axis(
        lambda col: sps.rankdata(-col), 0, data)
    mean_ranks = {s: float(ranks[k].mean()) for k, s in enumerate(solvers)}

    if reference is not None:
        if reference not in solvers:
            raise ValueError(f"reference {reference!r} not in matrix")
        pairs = [(reference, s) for s in solvers if s != reference]
    else:
        pairs = [(a, b) for k, a in enumerate(solvers)
                 for b in solvers[k + 1:]]

    raw = []
    for a, b in pairs:
        xa = np.array([matrix[a][i] for i in common])
        xb = np.array([matrix[b][i] for i in common])
        diff = xa - xb
        if np.allclose(diff, 0):
            p_raw = 1.0
        else:
            p_raw = float(sps.wilcoxon(xa, xb, zero_method="wilcox",
                                       method="auto").pvalue)
        raw.append((a, b, p_raw, xa, xb, diff))

    p_holm = holm_correction([r[2] for r in raw])
    pairwise = [
        PairwiseResult(
            solver_a=a, solver_b=b, p_raw=p_raw, p_holm=ph,
            delta=cliffs_delta(xa, xb),
            magnitude=cliffs_magnitude(cliffs_delta(xa, xb)),
            median_diff=float(np.median(diff)),
            n_pairs=len(common),
        )
        for (a, b, p_raw, xa, xb, diff), ph in zip(raw, p_holm)
    ]
    return SuiteAnalysis(
        friedman_chi2=friedman_chi2, friedman_p=friedman_p,
        n_instances=len(common), solvers=solvers,
        mean_ranks=mean_ranks, pairwise=pairwise,
    )


def win_tie_loss(matrix: Dict[str, Dict[str, float]],
                 tol: float = 1e-12) -> Dict[Tuple[str, str], Tuple[int, int, int]]:
    """Pairwise (wins, ties, losses) of row solver vs column solver."""
    solvers = tuple(sorted(matrix))
    common = sorted(set.intersection(*(set(matrix[s]) for s in solvers)))
    out = {}
    for a in solvers:
        for b in solvers:
            if a == b:
                continue
            w = t = l = 0
            for i in common:
                d = matrix[a][i] - matrix[b][i]
                if d > tol:
                    w += 1
                elif d < -tol:
                    l += 1
                else:
                    t += 1
            out[(a, b)] = (w, t, l)
    return out
