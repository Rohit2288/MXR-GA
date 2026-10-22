#!/bin/bash
# Phase 21 reruns on mains power (power guard in experiments/35_r3_runs.py). Resumable: re-run to continue.
#   A. CP-SAT sub-solver ablation with the overflow-fixed model, 0.2 s per move, seeds 0-4 (Table 10).
#   B. Replication of the ablation with disjoint seeds 5-9, all four variants (Appendix C).
#   C. CP-SAT sub-solver with a 2 s per-move limit, seeds 0-4 (robustness of A).
#   D. Sub-solver micro-benchmark (one k = 36 move, meet-in-the-middle vs CP-SAT).
#   E. Structured corpus A competitors, seed 0 (replaces partly throttled p13_structured seed-0 runs).
#   F. Benchmark-FSSP-1 (replaces partly throttled p18_fssp1).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R="$PY experiments/35_r3_runs.py --workers 4"
step() { echo "== $(date '+%F %T') $1"; }
step A; $R --corpus benchmarks/fsspb_hard_eval --out-dir results/p21_cpsat_sub --wall 30 --seeds-new 5 --seeds-base 10 \
  --solvers ma_final_cpsat || exit $?
step B; $R --corpus benchmarks/fsspb_hard_eval --out-dir results/p21_gateG_rep --wall 30 --seeds-new 5 --seeds-base 10 \
  --seed-offset 5 --solvers ma_final ma_final_rand lns_final ma_final_cpsat || exit $?
step C; $R --corpus benchmarks/fsspb_hard_eval --out-dir results/p21_cpsat_sub_cap2s --wall 30 --seeds-new 5 --seeds-base 10 \
  --solvers "ma_final_cpsat@cap_ms=2000" || exit $?
step D; $PY experiments/42_subsolver_microbench.py --n-inst 40 --cap 30 --workers 4 || exit $?
step E; $R --corpus benchmarks/fsspb_structured --out-dir results/p20_structured_seed0 --wall 30 --seeds-new 1 --seeds-base 1 \
  --solvers ktree ckk kk flsss flsss_exact cpsat "lattice@max_n=250" || exit $?
step F; $R --corpus fssp1 --out-dir results/p20_fssp1 --seeds-new 3 --seeds-base 3 \
  --solvers ma_final ktree random_search dp_exact flsss || exit $?
step done
