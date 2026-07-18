# REVISION_PLAN.md — FH-GA Major Revision (clean-slate rebuild)

IDs: R1.x / R2.x / R3.x (major), R3.mx (minor) / R4.x = reviewer comments; E = editor. Verbatim text in `reviewer_comments.md`.

**Session protocol:** read this file at session start; continue from the first unchecked task; check off (`[x]`) when done; commit with the task ID in the message. `DECISION(Rohit)` tasks require author input — never resolve unilaterally.

---

## Phase 0 — Core architecture

- [x] 0.1 `src/core/fuzzy.py`: TriangularFuzzyNumber with μ and α-cut. Port from `legacy/.../algorithm/fuzzy.py`, add unit tests (μ at T^L/T^M/T^R, outside support = 0).
- [x] 0.2 `src/core/instance.py`: loader + jsonschema validation for the legacy instance JSONs (both `instances_out/` and `instances_out_v2/`). Inventory script: counts per suite/category; verify against the counts the paper claims. Flag any discrepancy as DECISION(Rohit).
- [x] 0.3 `src/core/evaluate.py`: THE single scorer. Input: instance + subset indices → validates indices, sums actual items, returns μ. Nothing else in the codebase computes reported quality.
- [ ] 0.4 Solver interface (`src/solvers/base.py`): every solver implements `solve(instance, budget, rng) -> SolverResult(subset_indices, evals_used, wall_time)`. Quality is NEVER in the result.
- [ ] 0.5 `src/protocol/budget.py`: shared evaluation-budget counter + shared plateau early-exit (identical tolerance and patience for every solver). `src/protocol/runner.py`: multi-seed runner writing CSVs + `manifest.json` (git commit, seeds, date).
- [ ] 0.6 `tests/`: (a) subset validity for all solvers, (b) μ(any solver) ≤ μ(DP-exact) on small instances, (c) evals ≤ budget, (d) determinism given seed.
- [ ] 0.7 Fresh Suite U ground-truth verification: new enumeration script re-derives exact optima for all 384 Suite U instances; report min/max/mean/median n + full distribution + feasibility argument + numerical precision. Output a LaTeX table fragment. (R3.2)

## Phase 1 — Solvers (all through the Phase-0 interface)

- [ ] 1.1 Port FH-GA (`src/solvers/fhga.py`) from `legacy/.../algorithm/fhga.py`: Fast and Thorough configs, all five mechanisms + WBF-Repair, adapted to the interface and protocol.
- [ ] 1.2 `dp_bitset.py` — exact DP (start from legacy `dp_exact.py`; add subset reconstruction; document n/target feasibility limits). (R2.7, R3.3)
- [ ] 1.3 `mitm.py` — meet-in-the-middle, exact for small/medium n; document the n ceiling. (R2.7, R3.3)
- [ ] 1.4 `cpsat.py` — OR-Tools CP-SAT, piecewise-linear μ maximization, time-limited, returns incumbent subset. (R2.7, R3.3)
- [ ] 1.5 `sim_anneal.py` — simulated annealing; tuning on a held-out slice, documented. (R2.7)
- [ ] 1.6 `ils.py` — iterated local search (flip/swap neighborhood + perturbation). (R2.7)
- [ ] 1.7 `alpha_cut.py` — REWRITE with subset certification: DP with parent pointers over scaled sums, reconstruct explicit subset, harness scores true unscaled sum. Verify Suite-U overshoot is now 0% by construction; record before/after. Write the formal spec (α-grid, interval construction, rounding, reconstruction) as a LaTeX fragment for the paper. (R2.5, R3.1)
- [ ] 1.8 Port `vanilla_ga.py`, `random_search.py`, `greedy.py` to the interface (equal budgets, shared early-exit — fixes legacy asymmetries). (R4.3)
- [ ] 1.9 `fhga_crisp.py` — FH-GA with identical operators but a crisp asymmetric weighted-distance objective. The fuzzy-vs-crisp control. (R4.1, R2.1)
- [ ] 1.10 Local-search controls: standard best-improvement and swap-based local search variants to compare against gradient-guided LS under equal budgets. (R2.3)
- [ ] 1.11 DECISION(Rohit): a published SoTA memetic/evolutionary knapsack or subset-sum solver with available code to include verbatim (R1.4). If impractical, document justification; DP/MITM/CP-SAT stand as the strong exact baselines.

## Phase 2 — Main comparison (R2.6, R3.4, R4.2, E)

- [ ] 2.1 Run ALL solvers on ALL FSSP-2 sub-suites (T, A, U). Per-suite reporting. ≥20 seeds for stochastic solvers; mean, SD, worst case.
- [ ] 2.2 Run FSSP-1 once as a sanity check only; it will be demoted to a brief subsection/appendix in the paper. (R4.2)
- [ ] 2.3 Statistics per sub-suite on the new data: Friedman → Wilcoxon + Bonferroni–Holm → Cliff's delta; interpret effect sizes honestly. (R4.2)
- [ ] 2.4 Full-precision reporting: no "≈1.0"; appropriate significant digits + SD + worst case. LaTeX table fragments. (R3.m2)
- [ ] 2.5 Win/tie/loss pairwise table regenerated from new data.

## Phase 3 — Ablation (R2.8, R3.5, R4.1, E)

- [ ] 3.1 Full 5-mechanism ablation on Suites A and U, ≥20 seeds, mean/SD/worst-case, CIs, Wilcoxon per mechanism.
- [ ] 3.2 WBF-Repair cost-benefit: quality gain vs runtime per suite; profile and explain the runtime scaling pattern (early stopping? implementation?). (R2.9)
- [ ] 3.3 DP-window repair variant: same window as WBF-Repair but localized DP instead of brute force — answers R4.1's "why a GA at all?". Report honestly whichever way it goes.
- [ ] 3.4 DECISION(Rohit): if mechanisms still show no signal on hard suites, the paper narrative restructures around what DOES matter. Discuss in chat before writing fragments.

## Phase 4 — Fuzzy-vs-crisp and local-search controls

- [ ] 4.1 Run fhga_crisp vs FH-GA on FSSP-2; equal everything. (R4.1)
- [ ] 4.2 Run gradient-guided LS vs best-improvement vs swap LS. If no measurable advantage, the "gradient" claim is renamed/downgraded to a sign/direction rule. (R2.3)
- [ ] 4.3 DECISION(Rohit): final fuzzy-vs-crisp positioning once data is in. If crisp matches, value proposition = bounded objective + asymmetric tolerances + linguistic control, not search advantage.
- [ ] 4.4 Problem-positioning fragment: FSSP vs closest-subset-sum, interval SSP, tolerance-based SSP, fuzzy knapsack — what is and is not new. (R2.1)

## Phase 5 — Runtime & instrumentation (R2.9, R4.3)

- [ ] 5.1 Runtime table from the fair protocol: wall-clock AND evals consumed, every solver, every suite.
- [ ] 5.2 Re-test the "fastest at large n" claim under fair termination; keep only if it survives.

## Phase 6 — Paper fragments (for Overleaf)

All outputs land in `paper_fragments/` as compilable LaTeX snippets + figure files, for Rohit to paste/upload into Overleaf.

- [ ] 6.1 Pseudocode replacing Figure 2 (master loop) — algorithm environment. (R1.1)
- [ ] 6.2 Appendix pseudocode for HybridInitialise, FB-UX, ImpactAwareMutate, GradientLocalSearch, WBF-Repair, Mamdani controller: inputs, outputs, parameters, boundary conditions, termination — implementable independently. (R3.6)
- [ ] 6.3 Early Fast/Thorough design-objectives paragraph for §1 + pointers to validating experiments. (R1.3)
- [ ] 6.4 De-guarantee pass list: every "always/guaranteed/correctness-critical" location in the manuscript with replacement wording. (R2.4, R3.m7)
- [ ] 6.5 Parameter-justification subsection: derive from FSSP structure where possible; otherwise tuned defaults + sensitivity evidence with ranges, sampling, repetitions, error bars, interactions. (R2.2, R3.m3)
- [ ] 6.6 Rewritten Related Work: critical analysis across fuzzy subset-sum, closest-subset-sum, fuzzy knapsack, tolerance-based SSP, memetic optimization; comparison table of objectives/complexity/output form/scale. Condensed Table 1. (R2.10, R3.7, R4.4)
- [ ] 6.7 Bibliography fixes: the "???" Springer entry, Reference [3], full verification of all entries. (R2.10, R3.m8)
- [ ] 6.8 All figures regenerated from new CSVs: Figure 5 redrawn cleanly; Figure 6 replaced by log-scale residual (1−μ) plot; each figure audited against its CSV. (R1.2, R4.4)
- [ ] 6.9 New results tables (main comparison, ablation, runtime, stats) as fragments. (R3.m2)
- [ ] 6.10 Condensation guidance: list of cuttable/appendix-movable passages (benchmark-generation detail, inconclusive ablation prose, conclusion overlap). (R3.m5, R3.m6, R4.4)
- [ ] 6.11 Acronym first-use audit: FSSP, SSP, FPTAS, FB-UX, WBF-Repair. (R3.m1)

## Phase 7 — Submission deliverables

- [ ] 7.1 Point-by-point response letter: every comment from all four reviewers + editor, quoted, with the response and the manuscript location of the change. Draft here; final wording with Rohit in chat.
- [ ] 7.2 Updated cover letter.
- [ ] 7.3 DECISION(Rohit): does GPEM require a tracked-changes/diff manuscript? If yes, produce via latexdiff against the submitted version.
- [ ] 7.4 Final audit: every reviewer comment maps to a concrete change or a justified rebuttal. Produce a mapping table for the response letter.

---

## Open decisions queue (Rohit)

1. (1.11) Published SoTA solver, or justify exact methods as strong baselines.
2. (3.4) Narrative restructuring pending hard-suite ablation results.
3. (4.3) Fuzzy-vs-crisp positioning pending control results.
4. (0.2) Any instance-count discrepancies found by the inventory.
5. (7.3) GPEM tracked-changes requirement.
