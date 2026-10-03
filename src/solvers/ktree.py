"""Generalized-birthday (k-tree) baseline for dense subset sum (Phase 12).

Risk audit (2026-10-03): the dense / high-density subset sum literature
(Wagner 2002, k-tree; Lyubashevsky 2005; Shallue 2008; Dinur 2018 framework)
targets exactly the regime claimed for ma_xr (many items, large values). A
reviewer could reasonably ask for it, so it is implemented here in its
integer form (cf. "On Wagner's k-tree algorithm over integers", 2024).

One attempt:
  1. Draw m = 2^t disjoint random groups of g items; each group yields a
     list of all 2^g subset sums (with masks).
  2. Fix the remaining items by randomised greedy so that the residual
     target r = T_M - sum(fixed) sits near the centre of what the groups
     can reach (sum of half the group totals).
  3. Merge lists pairwise up a binary tree. At each node only the M pair
     sums closest to that node's share of r are kept (share proportional to
     the node's total), so the window around the target shrinks by about a
     factor M per level -- the integer form of Wagner's bit-zeroing.
  4. At the root, the pair sum closest to r (asymmetric distance) gives an
     explicit subset, scored through the protocol (certified).
Attempts repeat with fresh groups until the protocol stops the run.

Parameters: g = 16 (lists of 65,536), M = 65,536 survivors per node,
t = min(4, floor(log2(n / g))) levels (up to 16 groups = 256 items).
One counted evaluation per attempt.
"""

from __future__ import annotations

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver
from src.solvers.exact_lns import _half_sums


def _keep_closest(a: np.ndarray, b: np.ndarray, c: float, m: int):
    """Indices (i, j) of the <= m pairs whose a[i] + b[j] is closest to c.
    a, b sorted ascending. Window found by bisection on its half-width.
    The centre is clamped into the attainable range, and materialisation is
    capped (rows subsampled) so a degenerate window can never exhaust memory."""
    c = float(min(max(c, a[0] + b[0]), a[-1] + b[-1]))

    def count(w):
        lo = np.searchsorted(b, c - w - a, side="left")
        hi = np.searchsorted(b, c + w - a, side="right")
        return lo, hi, int((hi - lo).sum())
    # Estimate the window from a random sample of pair sums (quantile of
    # |a_i + b_j - c| at the target survivor fraction), then adjust with a
    # few exact counts: typically 1-3 searchsorted passes instead of ~35.
    rng = np.random.default_rng(len(a) * 1_000_003 + len(b))
    samp = 200_000
    d = np.abs(a[rng.integers(len(a), size=samp)]
               + b[rng.integers(len(b), size=samp)] - c)
    frac = min(1.0, 1.5 * m / (float(len(a)) * len(b)))
    w = float(np.quantile(d, frac)) + 0.5
    lo, hi, n = count(w)
    for _ in range(40):
        if n >= m:
            break
        w = w * 2.0 + 1.0
        lo, hi, n = count(w)
    lens = hi - lo
    cap = 8 * m
    if n > cap:                      # degenerate (many equal sums): thin rows
        scale = cap / n
        lens = np.floor(lens * scale).astype(np.int64)
    ii = np.repeat(np.arange(len(a)), lens)
    starts = np.repeat(lo, lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    jj = starts + offs
    if len(ii) > m:
        sel = np.argsort(np.abs(a[ii] + b[jj] - c), kind="stable")[:m]
        ii, jj = ii[sel], jj[sel]
    return ii, jj


class KTree(Solver):
    name = "ktree"

    def __init__(self, g: int = 16, keep: int = 65536, max_levels: int = 4):
        self.g, self.keep, self.max_levels = g, keep, max_levels
        self.last_stats: dict = {}

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        n = instance.n
        items = np.asarray(instance.items, dtype=np.int64)
        t_m = int(round(instance.target.T_M))
        d_l = instance.target.delta_L or 1.0
        d_r = instance.target.delta_R or 1.0
        g = min(self.g, n)
        levels = int(min(self.max_levels, max(0, np.floor(np.log2(max(1, n // g))))))
        m = 2 ** levels
        stats = self.last_stats = {"attempts": 0, "levels": levels}

        while True:
            perm = rng.permutation(n)
            groups = [perm[i * g:(i + 1) * g] for i in range(m)]
            rest = perm[m * g:]
            totals = np.array([int(items[gr].sum()) for gr in groups], dtype=np.float64)
            centre = 0.5 * totals.sum()
            # Randomised greedy on the remaining items: fill up to T - centre.
            fixed, s = [], 0
            for i in rng.permutation(rest):
                a = int(items[i])
                if s + a <= t_m - centre:
                    fixed.append(int(i))
                    s += a
            r = t_m - s
            # Leaves: (sorted sums, order -> mask index) per group.
            nodes = []
            for gi, gr in enumerate(groups):
                sums = _half_sums(items[gr])
                order = np.argsort(sums, kind="stable")
                nodes.append({"sums": sums[order], "share": totals[gi],
                              "leaf": (gi, order)})
            tree = []                      # per level: list of (left, right, ii, jj)
            while len(nodes) > 1:
                nxt, lvl = [], []
                for k in range(0, len(nodes), 2):
                    A, B = nodes[k], nodes[k + 1]
                    share = A["share"] + B["share"]
                    c = r * share / totals.sum()
                    ii, jj = _keep_closest(A["sums"], B["sums"], c, self.keep)
                    ps = A["sums"][ii] + B["sums"][jj]
                    order = np.argsort(ps, kind="stable")
                    nxt.append({"sums": ps[order], "share": share,
                                "kids": (A, B, ii[order], jj[order])})
                nodes = nxt
            root = nodes[0]
            under = r - root["sums"]
            dist = np.where(under >= 0, under / d_l, -under / d_r)
            best = int(np.argmin(dist))
            # Reconstruct the chosen leaf masks.
            chosen = list(fixed)

            def walk(node, idx):
                if "leaf" in node:
                    gi, order = node["leaf"]
                    mask = int(order[idx])
                    gr = groups[gi]
                    chosen.extend(int(gr[b]) for b in range(len(gr)) if (mask >> b) & 1)
                    return
                A, B, ii, jj = node["kids"]
                walk(A, int(ii[idx]))
                walk(B, int(jj[idx]))
            walk(root, best)
            stats["attempts"] += 1
            objective.evaluate(sorted(chosen))
