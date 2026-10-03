"""Round-3 shared run definitions: one solver factory, one protocol.

Every round-3 driver builds solvers and protocols ONLY through this module,
so "identical settings for every solver" is a property of the code.

Protocol (wall-clock binding; see ProtocolConfig):
  * budget by instance size, identical for every solver:
      n <= 100: 2 s;  n <= 500: 5 s;  n > 500: 10 s
    (drivers may override with a fixed budget, e.g. the regime map);
  * evaluation cap high enough never to bind; schedules follow progress();
  * stop at the KNOWN optimum where one is certified (see known_optimum):
    after reaching the best attainable mu, more search cannot change the
    result, so stopping forgoes nothing and is applied to every solver.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.core.instance import Instance
from src.protocol.budget import ProtocolConfig

REPO = Path(__file__).resolve().parents[2]

V2 = dict(repair_mode="dp_window", uniform_xover=True, gamma=0.0,
          pc_low=0.8, pc_high=0.8, pm_low_mult=1.0, pm_high_mult=1.0)

NEW = ["ma_xr", "ma_rand", "lns_mitm"]
# Seeded like NEW when requested explicitly; not in the default list.
ABLATION = ["ma_xr_cpsat"]
BASE_STOCHASTIC = ["fhga_v2crisp", "ils_crisp", "sim_anneal_crisp",
                   "vanilla_ga_crisp", "random_search", "ktree"]
SINGLE = ["dp_exact", "mitm", "cpsat", "flsss", "lattice", "kk", "ckk",
          "alpha_cut", "greedy_center"]
STOCHASTIC = NEW + BASE_STOCHASTIC


def wall_budget(n: int) -> float:
    return 2.0 if n <= 100 else (5.0 if n <= 500 else 10.0)


def protocol(wall_s: float, stop_at_mu: Optional[float]) -> ProtocolConfig:
    return ProtocolConfig(max_evals=10_000_000, plateau_tol=1e-9,
                          plateau_patience=None, max_wall_s=wall_s,
                          stop_at_mu=stop_at_mu)


@lru_cache(maxsize=1)
def _suite_u_optima() -> dict:
    path = REPO / "results" / "suite_u_ground_truth" / "ground_truth.csv"
    with open(path) as fh:
        return {r["instance_id"]: float(r["exact_mu"]) for r in csv.DictReader(fh)}


@lru_cache(maxsize=1)
def _dp_optima() -> dict:
    """DP-proven optima from the round-2 main comparison (dp_exact rows with
    status ok). The DP is exact, so these are certified optima."""
    path = REPO / "results" / "final" / "combined.csv"
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["solver"] == "dp_exact" and r["status"] == "ok":
                out[r["instance_id"]] = float(r["mu"])
    return out


def known_optimum(inst: Instance) -> Optional[float]:
    """Certified optimum, or None. Sources, in order: a witness with mu = 1
    (planted / reachable apex, mu cannot exceed 1); Suite U exhaustive
    enumeration; the exact DP. Rounded down by 1e-12 so float noise in the
    scorer can never make an attained optimum look unattained."""
    if float(inst.witness.get("mu_value", 0.0)) >= 1.0:
        return 1.0 - 1e-12
    u = _suite_u_optima().get(inst.instance_id)
    if u is not None:
        return u - 1e-12 if u > 0 else None   # mu* = 0: nothing to stop on
    d = _dp_optima().get(inst.instance_id)
    if d is not None and d > 0:
        return d - 1e-12
    return None


def make_solver(name: str, wall_s: float, flsss_eps: float = 0.01, **kw):
    from src.solvers.alpha_cut import AlphaCut
    from src.solvers.baselines import GreedyCenter, RandomSearch, VanillaGA
    from src.solvers.cpsat import CPSAT
    from src.solvers.dp_bitset import DPBitset
    from src.solvers.exact_lns import LNSMITM, MAXR
    from src.solvers.fhga import FHGAConfig
    from src.solvers.fhga_crisp import FHGACrisp
    from src.solvers.flsss_solver import FLSSSSolver
    from src.solvers.ils import ILS
    from src.solvers.mitm import MITM
    from src.solvers.sim_anneal import SimAnneal
    k = kw.get("k", 32)
    pop = kw.get("pop", 16)
    table = {
        "ma_xr": lambda: MAXR(k=k, pop_size=pop, recomb="exact"),
        "ma_rand": lambda: MAXR(k=k, pop_size=pop, recomb="random"),
        "lns_mitm": lambda: LNSMITM(k=k),
        # CMSA-style ablation: same exact recombination, CP-SAT sub-solver.
        "ma_xr_cpsat": lambda: MAXR(k=k, pop_size=pop, recomb="exact",
                                    subsolver="cpsat"),
        "kk": lambda: __import__("src.solvers.kk", fromlist=["CKK"]).CKK(complete=False),
        "ckk": lambda: __import__("src.solvers.kk", fromlist=["CKK"]).CKK(complete=True),
        "fhga_v2crisp": lambda: FHGACrisp(variant="thorough",
                                          config=FHGAConfig(**V2),
                                          name="fhga_v2crisp"),
        "ils_crisp": lambda: ILS(guide_mode="crisp"),
        "sim_anneal_crisp": lambda: SimAnneal(guide_mode="crisp"),
        "vanilla_ga_crisp": lambda: VanillaGA(guide_mode="crisp"),
        "random_search": RandomSearch,
        "dp_exact": DPBitset,
        "mitm": lambda: MITM(max_n=kw.get("max_n", 44)),
        "cpsat": lambda: CPSAT(max_time_s=wall_s,
                               num_workers=kw.get("workers", 1)),
        "flsss": lambda: FLSSSSolver(
            eps=flsss_eps,
            tlimit_per_probe=wall_s / (2 + math.ceil(math.log2(1 / flsss_eps))),
            total_timeout=wall_s + 5),
        "alpha_cut": AlphaCut,
        "ktree": lambda: __import__("src.solvers.ktree", fromlist=["KTree"]).KTree(),
        "lattice": lambda: __import__("src.solvers.lattice", fromlist=["Lattice"]).Lattice(
            max_n=kw.get("max_n", 150)),
        "greedy_center": GreedyCenter,
    }
    return table[name]()
