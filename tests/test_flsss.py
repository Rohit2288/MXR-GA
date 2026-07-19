"""Tests for the FLSSS baseline wrapper (task 1.11)."""

import shutil

import numpy as np
import pytest

from src.core.brute_force import exact_optimum
from src.core.evaluate import evaluate
from src.core.instance import FSSP2_DIR, iter_instance_paths, load_instance
from src.protocol.budget import ProtocolConfig
from src.solvers.flsss_solver import FLSSSSolver

if not (shutil.which("Rscript") or __import__("pathlib").Path(
        "/opt/homebrew/bin/Rscript").exists()):
    pytest.skip("Rscript not available", allow_module_level=True)

CFG = ProtocolConfig(max_evals=100)


def small_u_instances(k=3):
    out = []
    for p in iter_instance_paths(FSSP2_DIR):
        if "suite_U" in str(p):
            inst = load_instance(p)
            if inst.n <= 18:
                out.append(inst)
            if len(out) == k:
                break
    return out


@pytest.mark.parametrize("inst", small_u_instances(),
                         ids=lambda i: i.instance_id)
def test_certified_and_bounded(inst):
    res = FLSSSSolver().solve(inst, CFG, np.random.default_rng(0))
    mu = evaluate(inst, res.subset_indices).mu  # raises if invalid subset
    assert mu <= exact_optimum(inst).mu + 1e-12
    assert res.evals_used <= CFG.max_evals
