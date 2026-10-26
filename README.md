# MXR-GA — code, benchmarks and raw results (release 2.0.0)

Companion release for the manuscript *MXR-GA: A Genetic Algorithm with Exact Meet-in-the-Middle
Recombination for the Fuzzy Subset Sum Problem* (R. Sharma, A. Kumar, D. Kumar). It contains the
solver code, every benchmark corpus with its certified optima, the raw per-run results of every
experiment reported in the paper, and the scripts that regenerate every table and figure.
Release 1.x of this record (concept DOI 10.5281/zenodo.20312050) holds the earlier FH-GA work.

## Layout

| Path | Content |
|---|---|
| `src/core/` | fuzzy target, instance loader, `evaluate.py` — the single scorer: every reported μ is recomputed from the returned subset's true sum |
| `src/solvers/` | MXR-GA (`exact_lns.py`, class `MAXR`; final configuration `ma_final` in `src/protocol/r3.py`) and all baselines: DP, meet-in-the-middle, CP-SAT, lattice reduction (fpylll), k-tree, KK/CKK, FLSSS, α-cut FPTAS, SA, ILS, GA, FH-GA, random search, greedy |
| `src/protocol/` | shared wall-clock budget and stop rule (`budget.py`), solver registry and certified optima (`r3.py`) |
| `benchmarks/` | generated corpora (regime map, structured A/B, asymmetric, hard evaluation, scale, beyond-MITM, reconciliation, tuning set); each has a `manifest.json` with generator, commit and seed base |
| `legacy/benchmark-fsspb/instances_out*` | Benchmark-FSSP-1 and -2 instances (canonical, reused unchanged) |
| `experiments/` | `35_r3_runs.py` (run driver), `40_tables_figures.py` (all tables and figures), `38_headline_numbers.py`, `41_fig_recombination.py` (Fig. 1, brute-force checked), `42_subsolver_microbench.py`, `run_p21_reruns.sh` |
| `results/<run set>/` | `results.csv` (one row per run: solver, instance, seed, status, μ, certified optimum, subset sum, evaluations, wall time, stop reason, trace) and `manifest.json` (commit, date, settings, power state) |
| `paper_fragments/r3/` | generated LaTeX tables, figures and the CSV behind each table |
| `tests/` | correctness tests (`pytest tests`) |
| `REVISION_PLAN.md` | the project plan, including every pre-registered criterion; its commit history dates each criterion before the corresponding runs |

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests                     # correctness tests
.venv/bin/python experiments/40_tables_figures.py        # every table, figure and table CSV -> paper_fragments/r3/
.venv/bin/python experiments/38_headline_numbers.py      # results/headline_numbers.json
```

Re-running experiments uses `experiments/35_r3_runs.py` (see the commands in `experiments/run_p21_reruns.sh`).
Budgets are wall-clock, so timings depend on the machine: the paper's runs used the four performance
cores of an Apple M1 on mains power with at most four concurrent runs. The driver refuses to start, and
stops, on battery or in macOS Low Power Mode, and records the power state in each manifest.

The run sets behind the paper are those read by `experiments/40_tables_figures.py`: p13_regime,
p13_regime_lattice, p13_scale, p13_structured, p14_structured_rep, p15_asym, p15_fssp2_runtime,
p16_asym, p16_fssp2, p16_gateG, p16_hard, p16_reconcile, p16_regime, p16_scale, p16_structured,
p16_structured_rep, p16_tuning, p17_popsens, p18_beyond, p18_fssp2_final, p18_fssp2_kt_ils,
p18_fssp2_meta, p18_fssp2_single, p18_hard, p18_hard_eval, p18_reconcile, p18_regime, p18_scale,
p18_speed120, p20_components, p20_fssp1, p20_lattice_uncapped, p20_structured_seed0, p21_cpsat_sub,
p21_gateG_rep, plus p21_cpsat_sub_cap2s, p20_subsolver_microbench and the r3_lattice60, r3_long60 and
r3_cpsat_closest_* checks cited in the text. Where a rerun replaces part of an older run set, the
rerun is loaded first and supersedes it (rule documented in `load()`).

## Provenance notes

* **Superseded runs are kept, marked `INVALID*.md`.** Runs made while the machine was CPU-throttled
  (battery / Low Power Mode, detected from evaluation throughput) or asleep were repeated
  (`results/p19_*`, `results/p20_gateG_rep`, and the seed-0 part of `p13_structured`, the throttled part of
  `p18_fssp1`). The CP-SAT sub-solver rows of `p13/p14/p15/p16_ablation` and `p20_cpsat_cap2s` are
  invalid because the model overflowed CP-SAT's 64-bit bound (MODEL_INVALID); the fixed sub-solver runs
  are `p21_cpsat_sub*` (regression test `tests/test_cpsat_subsolver_valid.py`).
* **Commit hashes.** This repository's history is the project history restricted to code, data and
  results (manuscript drafts and confidential peer-review material are not included), so commit hashes
  differ from the original ones. Each commit message ends with `Original-commit: <hash>`; the
  `git_commit` field in every `manifest.json` refers to that original hash
  (find it with `git log --all --grep=<hash>`). Commit dates are unchanged.

## License

MIT (see `LICENSE`). Please cite via `CITATION.cff`.
