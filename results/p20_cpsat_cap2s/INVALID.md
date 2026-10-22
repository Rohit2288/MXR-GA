# Not used

The CP-SAT sub-solver model overflowed CP-SAT's 64-bit objective bound on these instances (MODEL_INVALID), so every call returned the current assignment without solving. Fixed in src/solvers/exact_lns.py (weights reduced by their gcd, pre-scaling if still needed, statuses recorded); superseded by results/p20_cpsat_sub_cap2s.
