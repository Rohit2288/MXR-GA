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
    from src.solvers.fhga import FHGA
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
    ]
    return solvers
