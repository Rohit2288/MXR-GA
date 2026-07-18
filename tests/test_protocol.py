"""Tests for the solver interface + protocol (tasks 0.4, 0.5).

Uses a minimal in-test random sampler so protocol behavior is tested
independently of the real solvers (which arrive in Phase 1).
"""

import csv
import json

import numpy as np
import pytest

from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import (BudgetedObjective, BudgetExhausted,
                                 PlateauReached, ProtocolConfig)
from src.protocol.runner import run_experiment
from src.solvers.base import Solver


class DummyRandomSolver(Solver):
    """Uniform random subsets forever; termination must come from protocol."""

    name = "dummy_random"

    def search(self, instance, objective, rng):
        while True:
            mask = rng.random(instance.n) < 0.5
            objective.evaluate([int(i) for i in np.flatnonzero(mask)])


@pytest.fixture(scope="module")
def inst():
    return load_instance(next(iter(iter_instance_paths(FSSP2_DIR))))


class TestBudgetedObjective:
    def test_budget_exhaustion_and_count(self, inst):
        cfg = ProtocolConfig(max_evals=50)
        obj = BudgetedObjective(inst, cfg)
        with pytest.raises(BudgetExhausted):
            for _ in range(10_000):
                obj.evaluate([0])
        assert obj.evals_used == 50
        assert obj.stop_reason == "budget_exhausted"

    def test_evaluate_after_stop_refused(self, inst):
        obj = BudgetedObjective(inst, ProtocolConfig(max_evals=1))
        with pytest.raises(BudgetExhausted):
            obj.evaluate([0])
        evals = obj.evals_used
        with pytest.raises(BudgetExhausted):
            obj.evaluate([1])
        assert obj.evals_used == evals  # not counted

    def test_plateau_fires_without_improvement(self, inst):
        cfg = ProtocolConfig(max_evals=10_000, plateau_patience=20)
        obj = BudgetedObjective(inst, cfg)
        with pytest.raises(PlateauReached):
            for _ in range(10_000):
                obj.evaluate([0])  # same subset -> no improvement
        assert obj.evals_used <= 21 + 1
        assert obj.stop_reason == "plateau"

    def test_best_tracking_matches_rescoring(self, inst):
        cfg = ProtocolConfig(max_evals=200)
        obj = BudgetedObjective(inst, cfg)
        rng = np.random.default_rng(0)
        with pytest.raises(BudgetExhausted):
            while True:
                mask = rng.random(inst.n) < 0.5
                obj.evaluate([int(i) for i in np.flatnonzero(mask)])
        assert obj.best_subset is not None
        assert evaluate(inst, obj.best_subset).mu == pytest.approx(obj.best_mu)


class TestSolverInterface:
    def test_result_has_no_quality_field(self, inst):
        res = DummyRandomSolver().solve(
            inst, ProtocolConfig(max_evals=100), np.random.default_rng(1))
        assert set(res.__dataclass_fields__) == {
            "subset_indices", "evals_used", "wall_time", "stop_reason"}

    def test_evals_within_budget(self, inst):
        res = DummyRandomSolver().solve(
            inst, ProtocolConfig(max_evals=100), np.random.default_rng(1))
        assert res.evals_used <= 100
        assert res.stop_reason == "budget_exhausted"

    def test_determinism_given_seed(self, inst):
        cfg = ProtocolConfig(max_evals=300)
        r1 = DummyRandomSolver().solve(inst, cfg, np.random.default_rng(42))
        r2 = DummyRandomSolver().solve(inst, cfg, np.random.default_rng(42))
        assert r1.subset_indices == r2.subset_indices
        assert r1.evals_used == r2.evals_used

    def test_returned_subset_is_valid(self, inst):
        res = DummyRandomSolver().solve(
            inst, ProtocolConfig(max_evals=100), np.random.default_rng(3))
        evaluate(inst, res.subset_indices)  # raises if invalid


class TestRunner:
    def test_writes_csv_and_manifest(self, inst, tmp_path):
        cfg = ProtocolConfig(max_evals=50)
        csv_path = run_experiment(
            [DummyRandomSolver()], [inst], seeds=[1, 2], config=cfg,
            out_dir=tmp_path, script_name="test_protocol.py")
        rows = list(csv.DictReader(open(csv_path)))
        assert len(rows) == 2
        for row in rows:
            # mu in the CSV must equal a fresh rescoring of nothing but the
            # protocol-tracked subset — here just check bounds and budget.
            assert 0.0 <= float(row["mu"]) <= 1.0
            assert int(row["evals_used"]) <= cfg.max_evals
        manifest = json.load(open(tmp_path / "manifest.json"))
        assert manifest["seeds"] == [1, 2]
        assert manifest["protocol"]["max_evals"] == 50
        assert manifest["git_commit"] != ""
