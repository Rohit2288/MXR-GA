"""Tests for the DP-window repair variant (task 3.3)."""

import numpy as np
import pytest

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.solvers.fhga import FHGA, FHGAConfig

CFG = ProtocolConfig(max_evals=5_000)


def small_u_instances(k=3, max_n=18):
    out = []
    for p in iter_instance_paths(FSSP2_DIR):
        if "suite_U" in str(p):
            inst = load_instance(p)
            if inst.n <= max_n:
                out.append(inst)
            if len(out) == k:
                break
    return out


def make_solver():
    return FHGA(variant="thorough",
                config=FHGAConfig(repair_mode="dp_window"),
                name="fhga_dpwin")


@pytest.mark.parametrize("inst", small_u_instances(),
                         ids=lambda i: i.instance_id)
class TestDPWindowRepair:
    def test_contract(self, inst):
        s = make_solver()
        res = s.solve(inst, CFG, np.random.default_rng(5))
        mu = evaluate(inst, res.subset_indices).mu
        assert mu <= exact_optimum(inst).mu + 1e-12
        assert res.evals_used <= CFG.max_evals
        assert s.last_stats["wbf_repair_count"] == 0  # WBF branch never taken

    def test_determinism(self, inst):
        r1 = make_solver().solve(inst, CFG, np.random.default_rng(9))
        r2 = make_solver().solve(inst, CFG, np.random.default_rng(9))
        assert r1.subset_indices == r2.subset_indices


def test_dp_window_uses_far_fewer_evals_per_repair():
    """The point of task 3.3: same window, ~1 counted eval per repair
    instead of 2^w."""
    inst = small_u_instances(k=1)[0]
    dpwin = make_solver()
    wbf = FHGA(variant="thorough")
    dpwin.solve(inst, CFG, np.random.default_rng(3))
    wbf.solve(inst, CFG, np.random.default_rng(3))
    if wbf.last_stats["wbf_repair_count"] and dpwin.last_stats["dpwin_repair_count"]:
        assert dpwin.last_stats["dpwin_repair_count"] > 0
        # WBF burns 2^12 = 4096 evals per firing; DP-window at most 1.
        assert (wbf.last_stats["wbf_repair_count"] * 4096
                > dpwin.last_stats["dpwin_repair_count"] * 10)
