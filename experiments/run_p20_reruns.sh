#!/bin/bash
# Phase 20 reruns on mains power (Low Power Mode off, machine awake). Each step refuses to start, or stops,
# if the power state is wrong; finished rows are kept, so re-running this script resumes where it stopped.
#   1. Post-hoc component check (initial exact move, tournament selection), tuning set, tuning protocol.
#   2. Lattice reduction without a size cap on the hard evaluation corpus.
#   3. CP-SAT sub-solver with a 2 s per-move limit (10x the default) on the hard evaluation corpus.
#   4. Sub-solver micro-benchmark (one k = 36 move: meet-in-the-middle vs CP-SAT).
#   5. Structured corpus A competitors, seed 0 (the seed-0 runs of p13_structured partly ran throttled).
#   6. Benchmark-FSSP-1 (parts of p18_fssp1 ran throttled).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R="$PY experiments/35_r3_runs.py --workers 4"
step() { echo "== $(date '+%F %T') $1"; }
step 1; $R --corpus benchmarks/tuning_set --out-dir results/p20_components --wall 30 --seeds-new 3 --seeds-base 10 \
  --solvers "ma_xr@k=36,init=0,move=0" "ma_xr@k=36,init=0,move=0,imove=0" "ma_xr@k=36,init=0,move=0,sel=0" || exit $?
step 2; $R --corpus benchmarks/fsspb_hard_eval --out-dir results/p20_lattice_uncapped --wall 30 --seeds-new 5 --seeds-base 5 \
  --solvers "lattice@max_n=1100" || exit $?
step 3; $R --corpus benchmarks/fsspb_hard_eval --out-dir results/p20_cpsat_cap2s --wall 30 --seeds-new 5 --seeds-base 5 \
  --solvers "ma_final_cpsat@cap_ms=2000" || exit $?
step 4; $PY experiments/42_subsolver_microbench.py --n-inst 40 --cap 30 --workers 4 || exit $?
step 5; $R --corpus benchmarks/fsspb_structured --out-dir results/p20_structured_seed0 --wall 30 --seeds-new 1 --seeds-base 1 \
  --solvers ktree ckk kk flsss flsss_exact cpsat "lattice@max_n=250" || exit $?
step 6; $R --corpus fssp1 --out-dir results/p20_fssp1 --seeds-new 3 --seeds-base 3 \
  --solvers ma_final ktree random_search dp_exact flsss || exit $?
step done
