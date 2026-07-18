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
        VanillaGA(),
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
    ]
    return solvers
