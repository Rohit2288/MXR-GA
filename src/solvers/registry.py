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
    solvers: List[Solver] = []
    # Phase 1: append solvers here as they are implemented, e.g.
    # from src.solvers.fhga import FHGA
    # solvers.append(FHGA(config="fast"))
    return solvers
