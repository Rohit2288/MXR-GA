"""Synthetic-fixture tests for the queued report generators (09, 12).

These scripts run unattended in the overnight queue after long sweeps; a
crash there wastes a day. Each test builds a tiny synthetic results tree,
points the module's path constants at it, and runs main() end to end.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def load_module(script):
    spec = importlib.util.spec_from_file_location(
        script.replace(".py", ""), REPO / "experiments" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synth_rows(solver, suite, n_inst, seeds, rng, mu_shift=0.0, extra=None):
    rows = []
    for i in range(n_inst):
        base = rng.uniform(0.3, 0.9)
        for seed in range(seeds):
            rows.append({
                "solver": solver, "instance_id": f"{suite}-inst{i}",
                "suite": suite, "n": 20, "distribution": "uniform",
                "seed": seed, "status": "ok",
                "mu": min(1.0, max(0.0, base + mu_shift
                                   + rng.normal(0, 0.01))),
                "subset_sum": 100, "subset_size": 5,
                "evals_used": 1000, "wall_time": rng.uniform(0.1, 0.5),
                "stop_reason": "plateau", "budget": 50000,
                "extra": json.dumps(extra or {}),
            })
    return rows


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_ablation_report_end_to_end(tmp_path, rng, capsys):
    m = load_module("09_ablation_report.py")
    rows = []
    wbf_extra = {"wbf_repair_count": 3, "wbf_repair_time_s": 0.2}
    dpw_extra = {"dpwin_repair_count": 3, "dpwin_repair_time_s": 0.01}
    for suite in ("A", "U"):
        rows += synth_rows("fhga_thorough", suite, 12, 5, rng,
                           extra=wbf_extra)
        rows += synth_rows("fhga_fast", suite, 12, 5, rng, -0.02)
        for v in ("no_init", "no_ls", "no_controller", "no_impact",
                  "no_fb_xover"):
            rows += synth_rows(v, suite, 12, 5, rng, -0.01)
        rows += synth_rows("dp_window", suite, 12, 5, rng, 0.0,
                           extra=dpw_extra)
    df = pd.DataFrame(rows)
    ab = df[~df["solver"].isin(["fhga_thorough", "fhga_fast"])]
    main = df[df["solver"].isin(["fhga_thorough", "fhga_fast"])]

    (tmp_path / "ablation").mkdir()
    (tmp_path / "main").mkdir()
    ab.to_csv(tmp_path / "ablation" / "results.csv", index=False)
    main.to_csv(tmp_path / "main" / "results.csv", index=False)

    m.ABLATION = tmp_path / "ablation" / "results.csv"
    m.MAIN = tmp_path / "main" / "results.csv"
    m.OUT = tmp_path / "out"
    m.FRAG = tmp_path / "frag"
    m.FRAG.mkdir()
    assert m.main() == 0
    assert (m.OUT / "ablation_summary.csv").exists()
    assert (m.OUT / "ablation_tests.csv").exists()
    frag = (m.FRAG / "table_ablation.tex").read_text()
    assert r"\begin{table}" in frag and r"\bottomrule" in frag
    tests = pd.read_csv(m.OUT / "ablation_tests.csv")
    assert set(tests["suite"]) == {"A", "U"}
    assert (tests["p_holm"] >= tests["p_raw"] - 1e-15).all()


def test_controls_report_end_to_end(tmp_path, rng):
    m = load_module("12_controls_report.py")
    rows = []
    for suite in ("T", "A", "U"):
        rows += synth_rows("fhga_fast", suite, 10, 5, rng)
        rows += synth_rows("fhga_thorough", suite, 10, 5, rng)
        rows += synth_rows("fhga_crisp_fast", suite, 10, 5, rng, -0.03)
        rows += synth_rows("fhga_crisp_thorough", suite, 10, 5, rng, -0.03)
        rows += synth_rows("fhga_ls_bestflip", suite, 10, 5, rng, -0.005)
        rows += synth_rows("fhga_ls_swap", suite, 10, 5, rng, -0.01)
    df = pd.DataFrame(rows)
    ctrl = df[df["solver"].str.contains("crisp|ls_")]
    main = df[df["solver"].isin(["fhga_fast", "fhga_thorough"])]

    (tmp_path / "controls").mkdir()
    (tmp_path / "main").mkdir()
    ctrl.to_csv(tmp_path / "controls" / "results.csv", index=False)
    main.to_csv(tmp_path / "main" / "results.csv", index=False)

    m.CONTROLS = tmp_path / "controls" / "results.csv"
    m.MAIN = tmp_path / "main" / "results.csv"
    m.OUT = tmp_path / "out"
    m.FRAG = tmp_path / "frag"
    m.FRAG.mkdir()
    assert m.main() == 0
    res = pd.read_csv(m.OUT / "controls_tests.csv")
    # 4 pairs x 3 suites
    assert len(res) == 12
    assert set(res["task"].astype(str)) == {"4.1", "4.2"}
    frag = (m.FRAG / "table_controls.tex").read_text()
    assert r"\begin{table}" in frag and r"\bottomrule" in frag
