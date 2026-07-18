"""FH-GA ported to the Phase-0 interface (task 1.1).

Port of legacy/benchmark-fsspb/algorithm/fhga.py with all five mechanisms
(hybrid initialization, FB-UX, impact-aware mutation, gradient-guided local
search, Mamdani controller) plus optional WBF-Repair. Configurations:
FHGA(variant="fast") — no repair (w_max=0); FHGA(variant="thorough") — w_max=12.

Deliberate deviations from legacy (all protocol-driven, per R4.3 / R2.5):
  * Fitness comes ONLY from the protocol's BudgetedObjective; the solver never
    computes reported quality itself. One evaluation is charged per candidate
    subset scored (legacy charged one per incremental bit-flip re-evaluation,
    so mutation here costs one eval per mutated child, not one per flipped bit).
  * No private termination: legacy's eps_stop early-exit (mu >= 1-1e-3) and
    max_generations are removed. The run ends only when the shared budget or
    the shared plateau criterion fires. Stagnation restarts remain (a search
    mechanism, not termination).
  * The mutation annealing schedule phi (legacy: generation / max_generations)
    is driven by consumed budget fraction, since generations are unbounded.
  * RNG is the protocol-supplied numpy Generator.

Ablation variants (Phase 3) are expressed via config overrides exactly as in
legacy run_ablation_v2.py: no_init (init fracs 0), no_ls (p_ls=0), no_repair
(w_max=0), no_controller (pc_low=pc_high=0.8, pm mults 1.0), no_impact
(gamma=0), no_fb_xover (uniform_xover=True).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


@dataclass(frozen=True)
class FHGAConfig:
    pop_size: Optional[int] = None       # None -> max(50, 4n)
    elite: int = 2
    tournament_size: int = 3
    p_c0: float = 0.8
    p_m0: Optional[float] = None         # None -> 1/n
    gamma: float = 0.5
    p_ls: float = 0.3
    p_kick: float = 0.1
    ls_max_iters: int = 20
    init_greedy_frac: float = 0.30
    init_alpha_frac: float = 0.20
    pc_low: float = 0.5
    pc_high: float = 0.95
    pm_low_mult: float = 0.5
    pm_high_mult: float = 5.0
    w_max: int = 12                      # 0 disables WBF-Repair (Fast config)
    G_stag: int = 20
    fitness_change_eps: float = 1e-6
    uniform_xover: bool = False          # True = plain uniform crossover (ablation)
    ls_mode: str = "gradient"            # "gradient" | "best_flip" | "swap" (R2.3 controls)

    def materialize(self, n: int) -> "FHGAConfig":
        return replace(
            self,
            pop_size=self.pop_size if self.pop_size is not None else max(50, 4 * n),
            p_m0=self.p_m0 if self.p_m0 is not None else 1.0 / n,
        )


class _Ind:
    __slots__ = ("chrom", "f")

    def __init__(self, chrom: List[int], f: float):
        self.chrom = chrom
        self.f = f

    def clone(self) -> "_Ind":
        return _Ind(list(self.chrom), self.f)


class FHGA(Solver):
    def __init__(self, variant: str = "thorough", config: Optional[FHGAConfig] = None,
                 name: Optional[str] = None):
        if variant not in ("fast", "thorough"):
            raise ValueError(f"unknown variant {variant!r}")
        cfg = config or FHGAConfig()
        if variant == "fast":
            cfg = replace(cfg, w_max=0)
        self.cfg = cfg
        self.name = name or f"fhga_{variant}"
        self.last_stats: dict = {}

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        run = _Run(instance, objective, rng, self.cfg.materialize(instance.n),
                   guide=self._make_guide(instance))
        self.last_stats = run.stats
        run.run()

    def _make_guide(self, instance: Instance):
        """Internal search-guidance signal g(S, mu) -> float.

        Default: the membership mu itself. FHGACrisp overrides this with an
        unclamped crisp distance (task 1.9). The guide steers selection and
        local search ONLY; reported quality is always the protocol-tracked,
        harness-rescored mu, identically for every variant."""
        return None


class _Run:
    """State of one FH-GA run. All mu values come from objective.evaluate."""

    def __init__(self, instance: Instance, objective: BudgetedObjective,
                 rng: np.random.Generator, cfg: FHGAConfig, guide=None):
        self.items = list(instance.items)
        self.n = instance.n
        self.target = instance.target
        self.objective = objective
        self.rng = rng
        self.cfg = cfg
        self.guide = guide  # None -> fitness is mu; else g(S, mu) -> float
        min_spread = max(min(self.target.delta_L, self.target.delta_R), 1e-9)
        self.rho = [a / min_spread for a in self.items]
        self.rho_max = max(self.rho)
        self.p_c = cfg.p_c0
        self.p_m = cfg.p_m0
        self.stats = {
            "generations": 0, "restarts": 0, "wbf_repair_count": 0,
            "wbf_repair_time_s": 0.0, "ls_invocations": 0, "apex_crossings": 0,
        }

    # -- evaluation ---------------------------------------------------------

    def _eval_chrom(self, chrom: List[int]) -> _Ind:
        idx = [i for i, b in enumerate(chrom) if b]
        mu = self.objective.evaluate(idx)
        if self.guide is None:
            return _Ind(chrom, mu)
        return _Ind(chrom, self.guide(sum(self.items[i] for i in idx), mu))

    def _fit(self, idx: List[int]) -> float:
        mu = self.objective.evaluate(idx)
        if self.guide is None:
            return mu
        return self.guide(sum(self.items[i] for i in idx), mu)

    def _sum(self, chrom: List[int]) -> int:
        return sum(a for a, b in zip(self.items, chrom) if b)

    def _phi(self) -> float:
        return self.objective.evals_used / self.objective.config.max_evals

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        cfg = self.cfg
        population = self._hybrid_init()
        best_ever = max(population, key=lambda ind: ind.f).clone()
        stagnation = 0
        prev_best_f = best_ever.f
        while True:
            self.stats["generations"] += 1
            Q: List[_Ind] = []
            while len(Q) < cfg.pop_size:
                p1 = self._tournament(population)
                p2 = self._tournament(population)
                c1, c2 = self._crossover(p1, p2)
                self._impact_mutate(c1)
                self._impact_mutate(c2)
                Q.append(c1)
                if len(Q) < cfg.pop_size:
                    Q.append(c2)
            for ind in Q:
                if self.rng.random() < cfg.p_ls:
                    self._gradient_local_search(ind)
            if cfg.w_max > 0:
                for ind in Q:
                    self._wbf_repair(ind)  # no-op when the sum is in-support
            population = self._elitist_replace(population, Q)
            gen_best = max(population, key=lambda ind: ind.f)
            if gen_best.f > best_ever.f:
                best_ever = gen_best.clone()
            diversity = self._diversity(population)
            self._fuzzy_controller(
                diversity, max(0.0, min(1.0, best_ever.f)))
            if best_ever.f - prev_best_f < cfg.fitness_change_eps:
                stagnation += 1
            else:
                stagnation = 0
            prev_best_f = best_ever.f
            if stagnation > cfg.G_stag:
                population = self._restart(population)
                stagnation = 0
                self.stats["restarts"] += 1

    # -- initialization -----------------------------------------------------

    def _hybrid_init(self) -> List[_Ind]:
        cfg = self.cfg
        N = cfg.pop_size
        N_g = int(cfg.init_greedy_frac * N)
        N_a = int(cfg.init_alpha_frac * N)
        P: List[_Ind] = []
        for _ in range(N_g):
            jitter = self.rng.normal(0.0, 0.05) * self.target.T_M
            tau = max(self.target.T_L,
                      min(self.target.T_R, self.target.T_M + jitter))
            P.append(self._eval_chrom(self._greedy_fill(tau)))
        for _ in range(N_a):
            alpha = self.rng.uniform(0.2, 0.9)
            L_a, R_a = self.target.alpha_cut(alpha)
            P.append(self._eval_chrom(self._iss_greedy(L_a, R_a)))
        total = sum(self.items)
        density = min(1.0, self.target.T_M / max(total, 1e-9))
        while len(P) < N:
            chrom = [1 if self.rng.random() < density else 0
                     for _ in range(self.n)]
            P.append(self._eval_chrom(chrom))
        return P

    def _greedy_fill(self, tau: float) -> List[int]:
        center = tau / self.n
        order = sorted(range(self.n), key=lambda i: abs(self.items[i] - center))
        chrom = [0] * self.n
        s = 0.0
        for i in order:
            if s + self.items[i] <= self.target.T_R:
                chrom[i] = 1
                s += self.items[i]
            if s >= tau and self.rng.random() < 0.5:
                break
        return chrom

    def _iss_greedy(self, L: float, R: float) -> List[int]:
        order = sorted(range(self.n), key=lambda i: self.items[i], reverse=True)
        chrom = [0] * self.n
        s = 0.0
        for i in order:
            if s + self.items[i] <= R:
                chrom[i] = 1
                s += self.items[i]
            if s >= L and self.rng.random() < 0.3:
                break
        return chrom

    # -- variation ----------------------------------------------------------

    def _tournament(self, P: List[_Ind]) -> _Ind:
        k = min(self.cfg.tournament_size, len(P))
        contenders = self.rng.choice(len(P), size=k, replace=False)
        return max((P[int(i)] for i in contenders), key=lambda ind: ind.f)

    def _crossover(self, p1: _Ind, p2: _Ind):
        if self.rng.random() > self.p_c:
            return p1.clone(), p2.clone()
        if self.cfg.uniform_xover:
            w = 0.5
        else:
            w = p1.f / (p1.f + p2.f + 1e-6)
        c1 = [0] * self.n
        c2 = [0] * self.n
        for i in range(self.n):
            if self.rng.random() < w:
                c1[i] = p1.chrom[i]; c2[i] = p2.chrom[i]
            else:
                c1[i] = p2.chrom[i]; c2[i] = p1.chrom[i]
        return self._eval_chrom(c1), self._eval_chrom(c2)

    def _impact_mutate(self, ind: _Ind) -> None:
        phi = self._phi()
        pm = self.p_m
        gamma = self.cfg.gamma
        flipped = False
        for i in range(self.n):
            rho_norm = self.rho[i] / self.rho_max
            beta = (1.0 - phi) * rho_norm + phi * (1.0 - rho_norm)
            q = pm * (1.0 + gamma * (beta - 0.5))
            q = max(0.0, min(2.0 * pm, q))
            if self.rng.random() < q:
                ind.chrom[i] = 1 - ind.chrom[i]
                flipped = True
        if flipped:
            ind.f = self._fit([i for i, b in enumerate(ind.chrom) if b])

    # -- local search -------------------------------------------------------

    def _flip_candidate(self, ind: _Ind, i: int) -> List[int]:
        return [j for j, b in enumerate(ind.chrom) if (b if j != i else not b)]

    def _gradient_local_search(self, ind: _Ind) -> None:
        cfg = self.cfg
        self.stats["ls_invocations"] += 1
        T_M = self.target.T_M
        mode = cfg.ls_mode  # "gradient" | "best_flip" | "swap" (R2.3)
        for _ in range(cfg.ls_max_iters):
            if mode in ("gradient", "best_flip"):
                S = self._sum(ind.chrom)
                best_gain = 0.0
                best_i = -1
                best_f = ind.f
                for i in range(self.n):
                    sign = 1 if ind.chrom[i] == 0 else -1
                    new_S = S + sign * self.items[i]
                    if (S < T_M < new_S) or (S > T_M > new_S):
                        self.stats["apex_crossings"] += 1
                    new_f = self._fit(self._flip_candidate(ind, i))
                    if new_f - ind.f > best_gain + 1e-12:
                        best_gain = new_f - ind.f
                        best_i = i
                        best_f = new_f
                if best_i >= 0:
                    ind.chrom[best_i] = 1 - ind.chrom[best_i]
                    ind.f = best_f
                    continue
                if mode == "best_flip":
                    break
            ones = [i for i, b in enumerate(ind.chrom) if b == 1]
            zeros = [i for i, b in enumerate(ind.chrom) if b == 0]
            K = min(20, self.n)
            ones_sorted = sorted(ones, key=lambda i: self.items[i])[:K]
            zeros_sorted = sorted(zeros, key=lambda j: self.items[j])[:K]
            best_gain = 0.0
            best_pair = (-1, -1)
            best_f = ind.f
            for i in ones_sorted:
                for j in zeros_sorted:
                    cand = [k for k, b in enumerate(ind.chrom)
                            if (b and k != i) or k == j]
                    new_f = self._fit(cand)
                    if new_f - ind.f > best_gain + 1e-12:
                        best_gain = new_f - ind.f
                        best_pair = (i, j)
                        best_f = new_f
            if best_pair[0] >= 0:
                i, j = best_pair
                ind.chrom[i] = 0
                ind.chrom[j] = 1
                ind.f = best_f
                continue
            if mode == "swap":
                break
            if self.rng.random() < cfg.p_kick:
                i = int(self.rng.integers(0, self.n))
                j = int(self.rng.integers(0, self.n))
                if i == j:
                    j = (j + 1) % self.n
                cand_chrom = list(ind.chrom)
                cand_chrom[i] = 1 - cand_chrom[i]
                cand_chrom[j] = 1 - cand_chrom[j]
                new_f = self._fit([k for k, b in enumerate(cand_chrom) if b])
                if new_f > ind.f + 1e-12:
                    ind.chrom = cand_chrom
                    ind.f = new_f
                    continue
            break

    # -- WBF repair ---------------------------------------------------------

    def _wbf_repair(self, ind: _Ind) -> None:
        S = self._sum(ind.chrom)
        if self.target.T_L <= S <= self.target.T_R:
            return
        w = min(self.cfg.w_max, self.n)
        if w == 0:
            return
        t0 = time.perf_counter()
        self.stats["wbf_repair_count"] += 1
        window = sorted(range(self.n), key=lambda i: self.rho[i],
                        reverse=True)[:w]
        window_set = set(window)
        fixed = [i for i in range(self.n)
                 if i not in window_set and ind.chrom[i] == 1]
        best_f = ind.f
        best_mask = -1
        try:
            for mask in range(1 << w):
                cand = fixed + [window[k] for k in range(w) if mask & (1 << k)]
                f = self._fit(cand)
                if f > best_f + 1e-12:
                    best_f = f
                    best_mask = mask
        finally:
            if best_mask >= 0:
                for k, idx in enumerate(window):
                    ind.chrom[idx] = 1 if (best_mask & (1 << k)) else 0
                ind.f = best_f
            self.stats["wbf_repair_time_s"] += time.perf_counter() - t0

    # -- replacement / controller / diversity / restart ---------------------

    def _elitist_replace(self, parents: List[_Ind], offspring: List[_Ind]):
        N, elite = self.cfg.pop_size, self.cfg.elite
        parents_sorted = sorted(parents, key=lambda ind: ind.f, reverse=True)
        offspring_sorted = sorted(offspring, key=lambda ind: ind.f, reverse=True)
        E = [ind.clone() for ind in parents_sorted[:elite]]
        return E + [ind.clone() for ind in offspring_sorted[:max(0, N - elite)]]

    def _fuzzy_controller(self, D: float, mu_star: float) -> None:
        def tri_low(x): return max(0.0, min(1.0, 1.0 - x / 0.5))

        def tri_med(x):
            if x < 0.5:
                return max(0.0, x / 0.5)
            return max(0.0, 2.0 - x / 0.5)

        def tri_high(x): return max(0.0, min(1.0, (x - 0.5) / 0.5))

        D_low, D_med, D_high = tri_low(D), tri_med(D), tri_high(D)
        f_low, f_med, f_high = tri_low(mu_star), tri_med(mu_star), tri_high(mu_star)
        pc_strengths = {"LOW": 0.0, "MED": 0.0, "HIGH": 0.0}
        pm_strengths = {"LOW": 0.0, "MED": 0.0, "HIGH": 0.0}

        def fire(d, fterm, pc_label, pm_label):
            alpha = min(d, fterm)
            pc_strengths[pc_label] = max(pc_strengths[pc_label], alpha)
            pm_strengths[pm_label] = max(pm_strengths[pm_label], alpha)

        fire(D_low, f_low, "MED", "HIGH")
        fire(D_low, f_med, "LOW", "HIGH")
        fire(D_low, f_high, "LOW", "MED")
        fire(D_med, f_low, "HIGH", "MED")
        fire(D_med, f_med, "MED", "MED")
        fire(D_med, f_high, "MED", "LOW")
        fire(D_high, f_low, "HIGH", "LOW")
        fire(D_high, f_med, "HIGH", "LOW")
        fire(D_high, f_high, "MED", "LOW")
        cfg = self.cfg
        pc_centers = {"LOW": cfg.pc_low,
                      "MED": 0.5 * (cfg.pc_low + cfg.pc_high),
                      "HIGH": cfg.pc_high}
        pm_low_val = cfg.pm_low_mult / self.n
        pm_high_val = cfg.pm_high_mult / self.n
        pm_centers = {"LOW": pm_low_val,
                      "MED": 0.5 * (pm_low_val + pm_high_val),
                      "HIGH": pm_high_val}
        pc_den = sum(pc_strengths.values())
        pm_den = sum(pm_strengths.values())
        new_pc = (sum(pc_strengths[k] * pc_centers[k] for k in pc_strengths)
                  / pc_den if pc_den > 1e-9 else self.p_c)
        new_pm = (sum(pm_strengths[k] * pm_centers[k] for k in pm_strengths)
                  / pm_den if pm_den > 1e-9 else self.p_m)
        self.p_c = 0.6 * self.p_c + 0.4 * new_pc
        self.p_m = 0.6 * self.p_m + 0.4 * new_pm

    def _diversity(self, P: List[_Ind]) -> float:
        N = len(P)
        if N < 2:
            return 0.0
        n_samples = max(10, int(np.sqrt(N)))
        total = 0.0
        for _ in range(n_samples):
            i, j = self.rng.choice(N, size=2, replace=False)
            xi, xj = P[int(i)].chrom, P[int(j)].chrom
            total += sum(1 for a, b in zip(xi, xj) if a != b) / self.n
        return total / n_samples

    def _restart(self, population: List[_Ind]) -> List[_Ind]:
        sorted_pop = sorted(population, key=lambda ind: ind.f, reverse=True)
        keep = [sorted_pop[0].clone(),
                (sorted_pop[1] if len(sorted_pop) > 1 else sorted_pop[0]).clone()]
        fresh = self._hybrid_init()
        self.p_c = self.cfg.p_c0
        self.p_m = self.cfg.p_m0
        return (keep + fresh)[: self.cfg.pop_size]
