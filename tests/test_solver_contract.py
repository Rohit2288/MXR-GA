"""Solver contract tests (task 0.6) — cited in the reviewer response.

For EVERY solver in the registry, on small Suite U instances:
  (a) the returned subset is a valid index set,
  (b) its recomputed mu never exceeds the exhaustive-enumeration optimum,
  (c) evals_used <= budget,
  (d) the run is deterministic given the seed.

The registry is populated as Phase-1 solvers land; these tests apply to each
automatically.
"""

import numpy as np
import pytest

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.solvers.registry import default_solvers

SOLVERS = default_solvers()
BUDGET = ProtocolConfig(max_evals=2_000)


def small_suite_u_instances(k=3, max_n=18):
    paths = [p for p in iter_instance_paths(FSSP2_DIR) if "suite_U" in str(p)]
    out = []
    for p in paths:
        inst = load_instance(p)
        if inst.n <= max_n:
            out.append(inst)
        if len(out) == k:
            break
    return out


INSTANCES = small_suite_u_instances()

if not SOLVERS:
    pytest.skip("solver registry empty until Phase 1", allow_module_level=True)


@pytest.fixture(scope="module")
def optima():
    return {inst.instance_id: exact_optimum(inst).mu for inst in INSTANCES}


@pytest.mark.parametrize("solver", SOLVERS, ids=lambda s: s.name)
@pytest.mark.parametrize("inst", INSTANCES, ids=lambda i: i.instance_id)
class TestSolverContract:
    def test_valid_subset_and_budget(self, solver, inst):
        res = solver.solve(inst, BUDGET, np.random.default_rng(7))
        evaluate(inst, res.subset_indices)  # (a) raises if invalid
        assert res.evals_used <= BUDGET.max_evals  # (c)

    def test_mu_never_exceeds_exact_optimum(self, solver, inst, optima):
        res = solver.solve(inst, BUDGET, np.random.default_rng(7))
        mu = evaluate(inst, res.subset_indices).mu
        assert mu <= optima[inst.instance_id] + 1e-12  # (b)

    def test_determinism_given_seed(self, solver, inst):
        if not solver.deterministic:
            pytest.skip(f"{solver.name} is wall-clock-limited (anytime); "
                        "incumbent is not a pure function of the seed")
        r1 = solver.solve(inst, BUDGET, np.random.default_rng(123))
        r2 = solver.solve(inst, BUDGET, np.random.default_rng(123))
        assert r1.subset_indices == r2.subset_indices  # (d)
        assert r1.evals_used == r2.evals_used
