"""Exact-neighbourhood search for FSSP (round-3 redesign, pilot stage).

FSSP reduces to closest subset sum under the asymmetric distance
d(S) = (T_M - S)/dL below T_M and (S - T_M)/dR above it, with
mu = max(0, 1 - d) (the reduction proposition of the manuscript). This
module exploits that directly: fix every item except a free set F of k
items, and the best assignment of F is the subset of F whose sum is
closest, under d, to the residual r = T_M - sum(fixed). That closest-sum
problem is solved EXACTLY by meet-in-the-middle over F in O(2^(k/2) k)
time, independent of item magnitudes. One such move inspects all 2^k
assignments of F, so a single step searches a neighbourhood far larger
than any bit-flip operator, and it keeps working where a pseudo-polynomial
DP table does not fit in memory.

Three solvers share the move:

  * LNSMITM    single trajectory; each step frees k random items.
  * MAXR       memetic algorithm. Recombination frees the items on which
               two parents DISAGREE and solves that sub-problem exactly
               (optimal recombination over the disagreement set, with the
               agreed items inherited); an exact random-k move follows.
  * MAXR(recomb="random") the same population and selection, but the
               recombination's free set is drawn at random. Isolates
               whether the population structure itself carries information
               (the "why a GA at all" control, R4).

Selection is by d (the protocol's crisp guide, src/protocol/guide.py);
reported quality is always the protocol-tracked, harness-rescored mu.

Accounting: every repaired subset is ONE counted evaluation; the 2^k
inspected sums are internal, exactly as in the exact solvers and the
DP-window repair. Because of that, comparisons involving these solvers
use the protocol's wall-clock budget (ProtocolConfig.max_wall_s), under
which uncounted internal work is paid for in time.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


def _half_sums(vals: np.ndarray) -> np.ndarray:
    """All 2^len(vals) subset sums; index bit j <-> vals[j] included."""
    s = np.zeros(1, dtype=np.int64)
    for v in vals:
        s = np.concatenate([s, s + v])
    return s


def closest_subset(vals: np.ndarray, r: int, d_l: float, d_r: float
                   ) -> Tuple[float, np.ndarray]:
    """Exact: the subset of `vals` minimising d(sum - r) (asymmetric).

    Returns (d*, boolean inclusion mask over vals). d is unimodal in the
    sum, so for each left-half sum only the two right-half sums bracketing
    r - left can be optimal; checking both is exact.
    """
    k = len(vals)
    h = k // 2
    left = _half_sums(vals[:h])
    right = _half_sums(vals[h:])
    order = np.argsort(right, kind="stable")
    rs = right[order]
    pos = np.searchsorted(rs, r - left)
    best_d, best = np.inf, (0, 0)
    for cand in (np.clip(pos - 1, 0, len(rs) - 1),
                 np.clip(pos, 0, len(rs) - 1)):
        under = r - (left + rs[cand])          # > 0: total below the residual
        d = np.where(under >= 0, under / d_l, -under / d_r)
        j = int(np.argmin(d))
        if d[j] < best_d:
            best_d, best = float(d[j]), (j, int(order[cand[j]]))
    ml, mr = best
    mask = np.zeros(k, dtype=bool)
    for b in range(h):
        mask[b] = (ml >> b) & 1
    for b in range(k - h):
        mask[h + b] = (mr >> b) & 1
    return best_d, mask


class _Core:
    """Shared state for one run: items, distance, the exact move."""

    def __init__(self, instance: Instance, objective: BudgetedObjective,
                 rng: np.random.Generator, k: int, subsolver: str = "mitm",
                 symmetric: bool = False):
        self.n = instance.n
        self.items = np.asarray(instance.items, dtype=np.int64)
        t = instance.target
        self.t_m = int(round(t.T_M))
        self.d_l = t.delta_L if t.delta_L > 0 else 1.0
        self.d_r = t.delta_R if t.delta_R > 0 else 1.0
        if symmetric:
            # Ablation "crisp closest sum": ignore the fuzzy target's asymmetric
            # tolerances and minimise |S - T_M|. Reported quality is still mu.
            self.d_l = self.d_r = 1.0
        self.objective = objective
        self.rng = rng
        self.k = min(k, self.n)
        self.moves = 0
        self.subsolver = subsolver

    def dist(self, s: int) -> float:
        u = self.t_m - s
        return u / self.d_l if u >= 0 else -u / self.d_r

    def commit(self, chrom: np.ndarray) -> float:
        """One counted evaluation of chrom; returns its distance d."""
        self.objective.evaluate(np.flatnonzero(chrom).tolist())
        return self.dist(int(self.items[chrom].sum()))

    def random_init(self) -> np.ndarray:
        """Ablation "no greedy start": Bernoulli(T_M / sum) inclusion."""
        p = min(1.0, self.t_m / max(1, int(self.items.sum())))
        return self.rng.random(self.n) < p

    def greedy(self) -> np.ndarray:
        """Randomised greedy: random order, add while the sum stays <= T_M."""
        chrom = np.zeros(self.n, dtype=bool)
        s = 0
        for i in self.rng.permutation(self.n):
            a = int(self.items[i])
            if s + a <= self.t_m:
                chrom[i] = True
                s += a
        return chrom

    def exact_move(self, chrom: np.ndarray, free: np.ndarray) -> float:
        """Reassign `free` optimally given the rest of chrom; one evaluation.

        The current assignment of `free` is among the candidates, so the
        result is never worse; ties move to a different assignment, which
        lets the search drift across neutral plateaus."""
        free = self.rng.permutation(free)
        fixed = chrom.copy()
        fixed[free] = False
        r = self.t_m - int(self.items[fixed].sum())
        if self.subsolver == "mitm":
            _, mask = closest_subset(self.items[free], r, self.d_l, self.d_r)
        else:
            mask = self._cpsat_closest(free, r, chrom[free])
        chrom[free] = mask
        self.moves += 1
        return self.commit(chrom)

    def _cpsat_closest(self, free: np.ndarray, r: int,
                       current: np.ndarray) -> np.ndarray:
        """CMSA-style ablation: the same sub-problem handed to CP-SAT
        (min weighted |sum - r|, current assignment as hint), time-capped
        per move. Kept only if not worse than the current assignment."""
        from ortools.sat.python import cp_model
        vals = [int(v) for v in self.items[free]]
        wl = max(1, int(round(self.d_r)))      # weights: dist/dL ~ wl*under
        wr = max(1, int(round(self.d_l)))      # cross-multiplied by dL*dR
        m = cp_model.CpModel()
        x = [m.new_bool_var(f"x{i}") for i in range(len(vals))]
        hi = sum(vals) + abs(r)
        under = m.new_int_var(0, hi, "u")
        over = m.new_int_var(0, hi, "o")
        m.add(sum(v * xi for v, xi in zip(vals, x)) + under - over == r)
        m.minimize(wl * under + wr * over)
        for xi, c in zip(x, current):
            m.add_hint(xi, bool(c))
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        solver.parameters.max_time_in_seconds = max(
            0.01, min(0.2, self.objective.time_left()))
        solver.parameters.random_seed = int(self.rng.integers(1, 2**31 - 1))
        st = solver.solve(m)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return current.copy()
        mask = np.array([bool(solver.value(xi)) for xi in x])

        def dist(msk):
            u = r - int(np.asarray(vals)[msk].sum()) if msk.any() else r
            return u / self.d_l if u >= 0 else -u / self.d_r
        return mask if dist(mask) <= dist(current) else current.copy()

    def random_free(self, exclude: Optional[np.ndarray] = None,
                    size: Optional[int] = None) -> np.ndarray:
        size = self.k if size is None else size
        if exclude is None or len(exclude) == 0:
            return self.rng.choice(self.n, size=size, replace=False)
        pool = np.setdiff1d(np.arange(self.n), exclude, assume_unique=True)
        size = min(size, len(pool))
        return self.rng.choice(pool, size=size, replace=False)


class LNSMITM(Solver):
    """Single-trajectory exact large-neighbourhood search."""

    name = "lns_mitm"

    def __init__(self, k: int = 32, name: Optional[str] = None, init: str = "greedy"):
        self.k = k
        self.init = init
        if name:
            self.name = name
        self.last_stats: dict = {}

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        core = _Core(instance, objective, rng, self.k)
        self.last_stats = {"moves": 0}
        chrom = core.greedy() if self.init == "greedy" else core.random_init()
        core.commit(chrom)
        while True:
            core.exact_move(chrom, core.random_free())
            self.last_stats["moves"] = core.moves


class MAXR(Solver):
    """Memetic algorithm with exact recombination and exact moves."""

    def __init__(self, k: int = 32, pop_size: int = 16,
                 recomb: str = "exact", name: Optional[str] = None,
                 subsolver: str = "mitm", init: str = "greedy",
                 post_move: bool = True, symmetric: bool = False,
                 init_move: bool = True, selection: str = "tournament"):
        if recomb not in ("exact", "random"):
            raise ValueError(recomb)
        if selection not in ("tournament", "uniform"):
            raise ValueError(selection)
        # Post-hoc component checks: init_move=False skips the exact move on each
        # initial subset; selection="uniform" picks parents uniformly at random.
        self.init_move, self.selection = init_move, selection
        self.k, self.pop_size, self.recomb = k, pop_size, recomb
        self.subsolver = subsolver
        self.init, self.post_move, self.symmetric = init, post_move, symmetric
        default = "ma_xr" if recomb == "exact" else "ma_rand"
        if subsolver != "mitm":
            default += f"_{subsolver}"
        self.name = name or default
        self.last_stats: dict = {}

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        core = _Core(instance, objective, rng, self.k, self.subsolver,
                     symmetric=self.symmetric)
        stats = self.last_stats = {"moves": 0, "generations": 0,
                                   "accepted": 0, "mean_diff": 0.0}
        pop: List[np.ndarray] = []
        fit: List[float] = []
        seen = set()
        for _ in range(self.pop_size):
            c = core.greedy() if self.init == "greedy" else core.random_init()
            if self.init_move:
                f = core.exact_move(c, core.random_free())
            else:
                f = core.commit(c)
            key = c.tobytes()
            if key in seen:
                continue
            seen.add(key)
            pop.append(c)
            fit.append(f)

        def pick() -> int:
            if self.selection == "uniform":
                return int(rng.integers(len(pop)))
            a, b = rng.integers(len(pop), size=2)
            return int(a) if fit[a] <= fit[b] else int(b)

        diff_total = 0
        while True:
            i, j = pick(), pick()
            p1, p2 = pop[i], pop[j]
            child = p1.copy()
            if self.recomb == "exact":
                diff = np.flatnonzero(p1 != p2)
                diff_total += len(diff)
                if len(diff) > core.k:
                    free = rng.choice(diff, size=core.k, replace=False)
                else:
                    extra = core.random_free(exclude=diff,
                                             size=core.k - len(diff))
                    free = np.concatenate([diff, extra]).astype(np.int64)
            else:
                free = core.random_free()
            f = core.exact_move(child, free)
            if self.post_move:
                f = core.exact_move(child, core.random_free())
            stats["generations"] += 1
            stats["moves"] = core.moves
            stats["mean_diff"] = diff_total / stats["generations"]
            key = child.tobytes()
            if key in seen:
                continue
            worst = int(np.argmax(fit))
            if f <= fit[worst]:
                seen.discard(pop[worst].tobytes())
                pop[worst], fit[worst] = child, f
                seen.add(key)
                stats["accepted"] += 1
