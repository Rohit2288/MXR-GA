"""Solver registry (task 0.6).

`default_solvers()` returns one instance of every solver in its
paper-default configuration. The contract tests in
tests/test_solver_contract.py run against everything listed here, so each
Phase-1 solver must be added as it lands.
"""

from __future__ import annotations

from typing import List

from src.solvers.base import Solver


def default_solvers() -> List[Solver]:
    from src.solvers.alpha_cut import AlphaCut
    from src.solvers.baselines import GreedyCenter, RandomSearch, VanillaGA
    from src.solvers.cpsat import CPSAT
    from src.solvers.dp_bitset import DPBitset
    from src.solvers.exact_lns import LNSMITM, MAXR
    from src.solvers.fhga import FHGA, FHGAConfig
    from src.solvers.fhga_crisp import FHGACrisp
    from src.solvers.ils import ILS
    from src.solvers.mitm import MITM
    from src.solvers.sim_anneal import SimAnneal

    solvers: List[Solver] = [
        FHGA(variant="fast"),
        FHGA(variant="thorough"),
        DPBitset(),
        MITM(),
        CPSAT(max_time_s=5.0),
        SimAnneal(),
        ILS(),
        # R4.3 fairness (task 2.6): the same solvers steered by the shared
        # crisp guide, so they are not stranded on the mu = 0 plateau while
        # FH-GA escapes it by construction. See experiments/18.
        SimAnneal(guide_mode="crisp"),
        ILS(guide_mode="crisp"),
        VanillaGA(),
        VanillaGA(guide_mode="crisp"),
        RandomSearch(),
        GreedyCenter(),
        AlphaCut(),
        FHGACrisp(variant="fast"),
        FHGACrisp(variant="thorough"),
        # R2.3 local-search controls (task 1.10): FH-GA with the gradient LS
        # replaced by standard best-improvement / swap-only local search.
        FHGA(variant="fast", config=FHGAConfig(ls_mode="best_flip"),
             name="fhga_ls_bestflip"),
        FHGA(variant="fast", config=FHGAConfig(ls_mode="swap"),
             name="fhga_ls_swap"),
        # Round-3 redesign (pilot): exact-neighbourhood search and the
        # memetic algorithm with exact recombination, plus its control.
        LNSMITM(),
        MAXR(recomb="exact"),
        MAXR(recomb="random"),
        MAXR(recomb="exact", subsolver="cpsat", k=12),
        __import__("src.solvers.kk", fromlist=["CKK"]).CKK(complete=False),
        __import__("src.solvers.kk", fromlist=["CKK"]).CKK(complete=True),
    ]
    return solvers
