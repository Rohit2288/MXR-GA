# REVISION_PLAN.md — FH-GA Major Revision (clean-slate rebuild)

IDs: R1.x / R2.x / R3.x (major), R3.mx (minor) / R4.x = reviewer comments; E = editor. Verbatim text in `reviewer_comments.md`.

**Session protocol:** read this file at session start; continue from the first unchecked task; check off (`[x]`) when done; commit with the task ID in the message. `DECISION(Rohit)` tasks require author input — never resolve unilaterally.

---

## Phase 0 — Core architecture

- [x] 0.1 `src/core/fuzzy.py`: TriangularFuzzyNumber with μ and α-cut. Port from `legacy/.../algorithm/fuzzy.py`, add unit tests (μ at T^L/T^M/T^R, outside support = 0).
- [x] 0.2 `src/core/instance.py`: loader + jsonschema validation for the legacy instance JSONs (both `instances_out/` and `instances_out_v2/`). Inventory script: counts per suite/category; verify against the counts the paper claims. Flag any discrepancy as DECISION(Rohit).
- [x] 0.3 `src/core/evaluate.py`: THE single scorer. Input: instance + subset indices → validates indices, sums actual items, returns μ. Nothing else in the codebase computes reported quality.
- [x] 0.4 Solver interface (`src/solvers/base.py`): every solver implements `solve(instance, budget, rng) -> SolverResult(subset_indices, evals_used, wall_time)`. Quality is NEVER in the result.
- [x] 0.5 `src/protocol/budget.py`: shared evaluation-budget counter + shared plateau early-exit (identical tolerance and patience for every solver). `src/protocol/runner.py`: multi-seed runner writing CSVs + `manifest.json` (git commit, seeds, date).
- [x] 0.6 `tests/`: (a) subset validity for all solvers, (b) μ(any solver) ≤ μ(DP-exact) on small instances, (c) evals ≤ budget, (d) determinism given seed.
- [x] 0.7 Fresh Suite U ground-truth verification: new enumeration script re-derives exact optima for all 384 Suite U instances; report min/max/mean/median n + full distribution + feasibility argument + numerical precision. Output a LaTeX table fragment. (R3.2)

## Phase 1 — Solvers (all through the Phase-0 interface)

- [x] 1.1 Port FH-GA (`src/solvers/fhga.py`) from `legacy/.../algorithm/fhga.py`: Fast and Thorough configs, all five mechanisms + WBF-Repair, adapted to the interface and protocol.
- [x] 1.2 `dp_bitset.py` — exact DP (start from legacy `dp_exact.py`; add subset reconstruction; document n/target feasibility limits). (R2.7, R3.3)
- [x] 1.3 `mitm.py` — meet-in-the-middle, exact for small/medium n; document the n ceiling. (R2.7, R3.3)
- [x] 1.4 `cpsat.py` — OR-Tools CP-SAT, piecewise-linear μ maximization, time-limited, returns incumbent subset. (R2.7, R3.3)
- [x] 1.5 `sim_anneal.py` — simulated annealing; tuning on a held-out slice, documented. (R2.7)
- [x] 1.6 `ils.py` — iterated local search (flip/swap neighborhood + perturbation). (R2.7)
- [x] 1.7 `alpha_cut.py` — REWRITE with subset certification: DP with parent pointers over scaled sums, reconstruct explicit subset, harness scores true unscaled sum. Verify Suite-U overshoot is now 0% by construction; record before/after. Write the formal spec (α-grid, interval construction, rounding, reconstruction) as a LaTeX fragment for the paper. (R2.5, R3.1)
- [x] 1.8 Port `vanilla_ga.py`, `random_search.py`, `greedy.py` to the interface (equal budgets, shared early-exit — fixes legacy asymmetries). (R4.3)
- [x] 1.9 `fhga_crisp.py` — FH-GA with identical operators but a crisp asymmetric weighted-distance objective. The fuzzy-vs-crisp control. (R4.1, R2.1)
- [x] 1.10 Local-search controls: standard best-improvement and swap-based local search variants to compare against gradient-guided LS under equal budgets. (R2.3)
- [x] 1.11 DECISION(Rohit) RESOLVED — FLSSS integrated: a published SoTA memetic/evolutionary knapsack or subset-sum solver with available code to include verbatim (R1.4). If impractical, document justification; DP/MITM/CP-SAT stand as the strong exact baselines.

## Phase 2 — Main comparison (R2.6, R3.4, R4.2, E)

- [x] 2.1 Run ALL solvers on ALL FSSP-2 sub-suites (T, A, U). Per-suite reporting. ≥20 seeds for stochastic solvers; mean, SD, worst case.
- [x] 2.2 Run FSSP-1 once as a sanity check only; it will be demoted to a brief subsection/appendix in the paper. (R4.2)
- [x] 2.3 Statistics per sub-suite on the new data: Friedman → Wilcoxon + Bonferroni–Holm → Cliff's delta; interpret effect sizes honestly. (R4.2)
- [x] 2.4 Full-precision reporting: no "≈1.0"; appropriate significant digits + SD + worst case. LaTeX table fragments. (R3.m2)
- [x] 2.5 Win/tie/loss pairwise table regenerated from new data.
- [x] 2.6 Baseline fairness (R2.7, R4.3) — COMPLETE, all three sub-suites,
      34,640 runs. SA and ILS were steered by mu, which is flat at zero
      outside the support, so they stalled on a zero plateau while FH-GA
      escapes it by construction. Shared guide in `src/protocol/guide.py`,
      guided variants added, SA re-tuned for the guided scale. Corrections:
      Suite T ILS 0.1983 -> 0.9235, SA 0.3583 -> 0.9404; Suite A ILS
      0.6319 -> 0.8874; Suite U ILS 0.7001 -> 0.7377, SA 0.3732 -> 0.6213.
      Against the repaired baselines FH-GA's advantage is small on T
      (delta=0.234 vs guided ILS) and negligible on A and U; FLSSS edges it
      on both. One large effect survives: FH-GA over random search on Suite T
      (delta=+0.508). See `paper_fragments/fairness_and_coverage_memo.md`.
- [x] 2.7 Coverage-aware comparison (R4.3, R3.4): `table_main_comparison.tex`
      compared dp_exact's mean over the 144 Suite A instances it solves against
      the heuristics' means over all 360. `table_coverage.tex` now reports
      coverage, own-mean and common-subset mean separately. See
      `paper_fragments/fairness_and_coverage_memo.md`.

## Phase 3 — Ablation (R2.8, R3.5, R4.1, E)

- [x] 3.1 Full 5-mechanism ablation on Suites A and U, ≥20 seeds, mean/SD/worst-case, CIs, Wilcoxon per mechanism.
- [x] 3.2 WBF-Repair cost-benefit: quality gain vs runtime per suite; profile and explain the runtime scaling pattern (early stopping? implementation?). (R2.9)
- [x] 3.3 DP-window repair variant: same window as WBF-Repair but localized DP instead of brute force — answers R4.1's "why a GA at all?". Report honestly whichever way it goes.
- [x] 3.4 DECISION(Rohit) RESOLVED (delegated 2026-07-20; see narrative_restructure.md FINAL VERDICT): narrative restructured around certified protocol + honest hierarchy + v2-crisp simplification.

## Phase 4 — Fuzzy-vs-crisp and local-search controls

- [x] 4.1 Run fhga_crisp vs FH-GA on FSSP-2; equal everything. (R4.1)
- [x] 4.2 Run gradient-guided LS vs best-improvement vs swap LS. If no measurable advantage, the "gradient" claim is renamed/downgraded to a sign/direction rule. (R2.3)
- [x] 4.3 DECISION(Rohit) RESOLVED (delegated; verdict in narrative memo): final fuzzy-vs-crisp positioning once data is in. If crisp matches, value proposition = bounded objective + asymmetric tolerances + linguistic control, not search advantage.
- [x] 4.4 Problem-positioning fragment: FSSP vs closest-subset-sum, interval SSP, tolerance-based SSP, fuzzy knapsack — what is and is not new. (R2.1)

## Phase 5 — Runtime & instrumentation (R2.9, R4.3)

- [x] 5.1 Runtime table from the fair protocol: wall-clock AND evals consumed, every solver, every suite.
- [x] 5.2 Re-test the "fastest at large n" claim under fair termination; keep only if it survives.

## Phase 6 — Paper fragments (for Overleaf)

All outputs land in `paper_fragments/` as compilable LaTeX snippets + figure files, for Rohit to paste/upload into Overleaf.

- [x] 6.1 Pseudocode replacing Figure 2 (master loop) — algorithm environment. (R1.1)
- [x] 6.2 Appendix pseudocode for HybridInitialise, FB-UX, ImpactAwareMutate, GradientLocalSearch, WBF-Repair, Mamdani controller: inputs, outputs, parameters, boundary conditions, termination — implementable independently. (R3.6)
- [x] 6.3 Early Fast/Thorough design-objectives paragraph for §1 + pointers to validating experiments. (R1.3)
- [x] 6.4 De-guarantee pass list: every "always/guaranteed/correctness-critical" location in the manuscript with replacement wording. (R2.4, R3.m7)
- [x] 6.5 Parameter-justification subsection: derive from FSSP structure where possible; otherwise tuned defaults + sensitivity evidence with ranges, sampling, repetitions, error bars, interactions. (R2.2, R3.m3)
- [x] 6.5a Sensitivity sweep (`experiments/08_sensitivity.py`): one-at-a-time ranges over FH-GA parameters + p_c×p_m interaction grid, stratified A/U slice, ≥10 seeds, error bars; data source for 6.5. (R2.2, R3.m3)
- [x] 6.6 Rewritten Related Work: critical analysis across fuzzy subset-sum, closest-subset-sum, fuzzy knapsack, tolerance-based SSP, memetic optimization; comparison table of objectives/complexity/output form/scale. Condensed Table 1. (R2.10, R3.7, R4.4)
- [x] 6.7 Bibliography fixes: the "???" Springer entry, Reference [3], full verification of all entries. (R2.10, R3.m8)
- [x] 6.8 All figures regenerated from new CSVs: Figure 5 redrawn cleanly; Figure 6 replaced by log-scale residual (1−μ) plot; each figure audited against its CSV. (R1.2, R4.4)
- [x] 6.9 New results tables (main comparison, ablation, runtime, stats) as fragments. (R3.m2)
- [x] 6.10 Condensation guidance: list of cuttable/appendix-movable passages (benchmark-generation detail, inconclusive ablation prose, conclusion overlap). (R3.m5, R3.m6, R4.4)
- [x] 6.11 Acronym first-use audit: FSSP, SSP, FPTAS, FB-UX, WBF-Repair. (R3.m1)

## Phase 7 — Submission deliverables

- [x] 7.1 Point-by-point response letter — DRAFTED against final numbers (2026-08-01), all four reviewers + editor, every comment quoted with response and a "Change:" location. Section numbers still bound as (§?) pending Overleaf. Includes a "Summary of claims withdrawn" block — ROHIT to approve keeping it.
- [x] 7.2 Updated cover letter.
- [x] 7.3 RESOLVED (2026-08-01, from the journal's own guidelines): GPEM's
      submission guidelines require "all relevant editable source files at
      every submission and revision" but contain NO tracked-changes or
      marked-up-manuscript requirement. Decision: produce the latexdiff PDF
      anyway and upload it as a supplementary "marked-up manuscript" item —
      it is optional, costs nothing (`scripts/make_latexdiff.sh` is vendored
      and tested), and directly serves the reviewers' request to see where
      each change landed. Rohit: if the decision e-mail from Editorial
      Manager states otherwise, that instruction overrides this.
- [x] 7.4 Final audit: every reviewer comment maps to a concrete change or a justified rebuttal. Produce a mapping table for the response letter.

---

## Open decisions queue (Rohit)

1. (1.11) RESOLVED: FLSSS (Option A) — integrate as published tolerance-based solver (Rohit, 2026-07-19).
2. (3.4) RESOLVED & COMPLETE: v2-crisp is the recommended config, 99.9% exact on U, beats Thorough p=2e-22 (2026-07-20).
3. (4.3) RESOLVED: crisp guide wins as search signal; fuzzy = modeling interface; v2crisp adopts crisp guide (2026-07-20).
4. (0.2) Any instance-count discrepancies found by the inventory.
5. (7.3) RESOLVED: not required by GPEM; produce and attach anyway (2026-08-01).
6. (6.7) RESOLVED: `ref_zakharov` to be removed (Rohit, 2026-07-19).
7. (2.6/2.7) RESOLVED (Rohit delegated, 2026-08-08): adopt the honest
   reframe — paper leads on the phenomenon+theory+three-domain+protocol
   contribution; FH-GA demoted to a strong instance; the three over-claims
   withdrawn. Single source of truth: `paper_fragments/positioning_final.md`.

## Phase 8 — Contribution-strengthening additions (R2) and manuscript build

- [x] 2.17 Third domain (fuzzy multiple-subset-sum, k=2 coupled targets):
      exact 2-D DP verified vs brute force; 108-instance corpus; phenomenon
      reproduces (plateau mass 0.964; crisp-guide +0.109 over mu-guide,
      +0.273 over blind), concentrated where plateau deepest per theory.
      Third structurally-distinct domain confirming the clipped-objective
      principle (experiments/31, src/core/mssp.py). (R2 contribution)

- [x] 2.8 Structural difficulty axis Gamma = mu* - best-of-K-random, computed
      per instance with no reference to any solver (experiments/20, 21).
      56.2% of instances with a known optimum are already solved by blind
      sampling. FH-GA over random search by Gamma quartile: +0.0055 / +0.0021
      / +0.0107 / +0.1905. (R2.6, R4.2)
- [x] 2.9 Plateau mass quantified: predicts the guided-instrumentation gain
      (SA rho=+0.603, p=5e-87). Generalises the finding to clipped objectives
      beyond FSSP. (R2.7, R4.3)
- [x] 2.10 Full 2^5 factorial, 30,720 runs, main effects + all 10 two-way
      interactions. Only hybrid init has a positive main effect (+0.0736);
      FB-UX and impact-aware mutation are negatively signed; one negligible
      interaction. Best config in the design space is init+impact+ctrl,
      not the proposed algorithm. (R2.2)
- [x] 2.11 LS-conditional-on-repair check: the factorial's null LS main effect
      and the leave-one-out's large one are reconciled — LS contributes only
      when repair is active (+0.0099 p=0.028 vs -0.0010 p=0.89). Reported,
      not smoothed. (R2.2)
- [x] 2.12 Formal reduction proposition mu = max(0, 1-d) with reductions both
      ways; FSSP conceded as closest-subset-sum with asymmetric penalties.
      (R2.1)
- [x] 7.5 Revised manuscript BUILT (scripts/build_manuscript.py) from the
      frozen submitted .tex plus generated fragments; marked-up version via
      latexdiff. Two deliverables: manuscript/fhga_manuscript_revised.tex and
      manuscript/fhga_manuscript_diff.tex. See manuscript/README.md.
- [ ] 7.6 Confirmatory runs before submission: ./scripts/run_queue10.sh
      stage 1 (Suite U at full scale, ~1.5 h) then rebuild. Stage 2 (Suite A,
      ~12 h) optional.
- [~] 7.7 Overleaf: bind section numbers into the response letter, apply the
      journal template, apply condensation_guidance.md, final proofread,
      upload figures.

## Author decisions (2026-08-03, Rohit)

1. Paper framing: BENCHMARK + PROTOCOL paper. FH-GA is the worked case study;
   contributions reordered to benchmark (with difficulty axis), protocol,
   clipped-objective analysis.
2. Recommended configuration: UNCHANGED (v2-crisp) pending queue10 stage 1.
   The factorial winner init+impact+ctrl came from a 96-instance slice with
   the top four configs inside 0.004.
3. Response letter: KEEP the "Summary of claims withdrawn" block, all seven,
   in the current direct wording.
4. Sequencing: confirmatory runs before final numbers are fixed. The
   manuscript is built so that re-running the generators refreshes every
   table automatically.

---

# ROUND 3 — New submission after round-2 reject (decision 2026-10-02)

Round-2 comments are appended verbatim to `reviewer_comments.md` (IDs R1b/R2b/R3b/R4b/AEb).
Blocking demand: AEb.1 + R4b.1 — show WHEN and WHY the proposed method should be used.
Everything else (R1b.1-4, R2b.1-5, R3b, R4b.2, AEb.2) is presentation/consistency work.
Rohit (2026-10-02): new submission, so the algorithm itself may be redesigned; run the pilot.

## Phase 9 — Decision pilot (pre-registered before results)

Data check that motivated it (results/final/combined.csv): fhga_v2crisp is strictly best
(excl. DP) on 0.8% of Suite T and 2.2% of Suite A instances; DP solves all of T in ~4 ms;
on Suite H FLSSS (0.9997) and guided ILS (0.86) beat it (0.75).

- [x] 9.1 Protocol: optional wall-clock budget, protocol-owned stop at mu = 1, and
      `progress()` for budget-fraction schedules (eval-budgeted runs unchanged; 352 tests pass).
- [x] 9.2 Redesigned algorithm prototype (`src/solvers/exact_lns.py`), built on the FSSP ->
      closest-subset-sum reduction: an EXACT move that frees k items and solves the residual
      closest-sum problem by meet-in-the-middle (2^k assignments per move, magnitude-independent).
      `ma_xr` = memetic algorithm whose recombination solves the parents' disagreement set exactly;
      controls `ma_rand` (same population, random free set) and `lns_mitm` (single trajectory).
- [x] 9.3 Regime-map suite R (`experiments/33`, 132 instances, planted mu* = 1): dense block
      n in {100,200,500,1000} x bits {16,24,32,40} x precision gap {8,16,exact}; critical block
      n ~ bits, n in {32,40,46}. Finding during design: with integer items, n >> bits makes every
      instance dense (~2^(n-bits) subsets per attainable sum); needles exist only near n ~ bits.
- [x] 9.4 Run pilot (`experiments/34`): 11 solvers, 10 s wall-clock each, 3 seeds (stochastic). 3,300 runs, 2026-10-02. Raw: results/regime_pilot/results.csv; per-instance: per_instance.csv.
- [x] 9.5 Analysed against the gates (outcome below). Rohit to confirm next step.

### Pilot outcome (2026-10-02)

Heuristic regime = DP infeasible and n > 44: 95 of 132 instances (83 dense, 12 critical).

- Gate N (as pre-registered, success = mu exactly 1): MET.
  dense: ma_xr 0.839 vs best existing FLSSS 0.325 (+51.4 pts, Cliff delta on mean mu +0.471);
  critical: ma_rand 0.472 vs every existing solver 0.000 (+47.2 pts, delta +0.667).
  CAVEAT, found in the fairness check: FLSSS is configured with eps = 0.01, so its alpha search
  stops at about mu >= 0.99. With success counted as mu >= 0.99, dense is ma_xr 0.940 vs FLSSS
  0.880 (+6 pts only); critical is ma_xr 0.778 / ma_rand 0.833 vs FLSSS 0.417 (+36 to +42 pts).
  So the large dense margin is partly an artefact of the FLSSS precision setting. The strong
  claim currently rests on the critical block plus the 40-bit dense cells. The rest of the
  dense block is "matches or slightly beats FLSSS, and far beats GA/ILS/SA/CP-SAT/FH-GA v2".
- Gate G: NOT MET. ma_xr vs ma_rand: +1.0 pts success, time-to-optimum delta -0.051;
  vs lns_mitm: +4.3 pts, delta -0.146 (threshold 0.147). ma_rand leads on critical (0.824 vs 0.796).
  Post-hoc only (not a gate): on the 40-bit dense cells, the hardest region, ma_xr hits mu = 1 on
  0.444 of runs vs ma_rand 0.347 and lns_mitm 0.278. That is small-sample and is a hypothesis
  for a confirmatory test, not a result.
- Small n / small magnitude (37 instances where DP or MITM runs): the exact methods solve what
  they can run on; the new solvers solve all 37 as well (k >= n makes one move exact).
- Old FH-GA v2-crisp: 0.070 success in the heuristic regime. The redesign is necessary in any case.

Rohit approved 9.6 + 9.7 (2026-10-02: "run both and help me get paper published").

- [x] 9.6 FLSSS fairness rerun. Diagnosis changed on inspection: the wrapper is not just imprecise,
      it has a BUG. The alpha = 1 probe sends ME = 0, for which FLSSS returns no solution ("[1] FALSE"),
      so FLSSS could never return an exact hit except via a lucky bisection probe. Fix in
      src/solvers/flsss_probe.R: probe the integer window [ceil L, floor R] with a +0.5 margin, which
      admits exactly the same attainable sums. Verified: 3 previously missed instances now reach mu = 1.
      eps stays 0.01 (with mu* = 1 planted, the alpha = 1 probe decides; fewer probes give it more time).
      NOTE for the paper: FLSSS numbers in the round-2 manuscript came from the buggy wrapper and may
      understate FLSSS. Every FLSSS number in the new submission is re-run with the fixed wrapper.
      Rerun: fixed FLSSS on all 132 suite-R instances, 10 s, the pilot protocol ->
      results/regime_pilot/flsss_fixed.csv. Gate N is re-evaluated with the same rule
      (success = mu exactly 1, existing-solver set as before, with fixed FLSSS replacing the old one).
- [x] 9.7 Confirmatory Gate G test (PRE-REGISTERED here, before the run).
      Corpus: benchmarks/fsspb_regime_confirm, 88 FRESH instances (seed base 990000, no overlap with
      the pilot): dense n in {100,200,500,1000} x bits 40 x gap {8,16,exact}; critical (n,bits) in
      {(46,38),(46,46),(46,47),(48,40),(48,47)} x gap {16,exact}; 4 instances per cell.
      Solvers: ma_xr, ma_rand, lns_mitm (5 seeds each), fixed FLSSS (1 run). 30 s wall-clock each,
      same protocol (stop at mu = 1).
      Gate G (unchanged thresholds): ma_xr beats BOTH ma_rand and lns_mitm by >= 10 points in success
      rate (mu = 1, per-instance seed means, averaged over instances), OR by Cliff's delta <= -0.147 in
      time-to-optimum (unsolved = +inf) on the instances where they differ. Both comparisons must pass.
      Secondary (reported, not a gate): Wilcoxon signed-rank on per-instance success, Holm over the 2 pairs.
      Result -> results/regime_confirm/results.csv.

### 9.7 outcome (2026-10-02): Gate G MET, both comparisons, both criteria
88 fresh hard instances, 5 seeds, 30 s (results/regime_confirm/results.csv, 1,408 runs).
- ma_xr vs ma_rand: success 0.577 vs 0.461 (+11.6 pts); time-to-optimum Cliff delta -0.197
  (79 differing instances); Wilcoxon on per-instance success p = 1.7e-4 (Holm 1.7e-4).
- ma_xr vs lns_mitm: 0.577 vs 0.423 (+15.5 pts); delta -0.349 (75 instances); p = 7.3e-5 (Holm 1.5e-4).
- Where it comes from: DENSE 40-bit cells, ma_xr 0.638 vs ma_rand 0.429 / lns_mitm 0.454. On the
  critical cells ma_xr = ma_rand (0.505 vs 0.500), and lattice reduction solves those anyway (Phase 11).
  So the population with exact recombination earns its place exactly in the regime the method is for.
- FLSSS (fixed wrapper): 0.000 on this corpus.
=> Path: GPEM new submission. The method is a memetic algorithm with exact recombination; the
   "why a GA" evidence is this pre-registered test.

### 9.6 outcome (2026-10-02): Gate N still MET with the fixed FLSSS
Fixed FLSSS (results/regime_pilot/flsss_fixed.csv) vs the best new solver, heuristic regime:
- dense (83 inst): ma_xr 0.839 vs FLSSS 0.554 (was 0.325 with the bug): +28.5 pts, Cliff delta(mu) +0.222.
- critical (12 inst): ma_rand 0.472 vs FLSSS 0.083: +38.9 pts, delta +0.576.
- By magnitude (dense), success: 16 and 24 bits, all four solvers 1.000 (FLSSS 0.75-1.4 s median vs
  0.01 s); 32 bits, new 1.000 vs FLSSS 0.458; 40 bits, new 0.28-0.44 vs FLSSS 0.000.
  This is the regime map's "when": the advantage appears once items reach about 32 bits, and
  grows with magnitude.
- Where DP or MITM can run, fixed FLSSS reaches 0.514; the new solvers reach 1.000.


### Pre-registered decision gates (written before any pilot result was seen)

Primary metric per (solver, instance): success = reaches mu = 1 (the planted optimum);
secondary: mean mu and median time-to-optimum. "Heuristic regime" = cells where DP is
infeasible AND n > 44 (MITM infeasible).

- Gate N (niche exists): in the heuristic regime, the best proposed solver (ma_xr or lns_mitm)
  exceeds the best existing solver (FLSSS, CP-SAT, guided ILS, guided SA, fhga_v2crisp,
  random search) in success rate by >= 20 percentage points, OR in mean mu with Cliff's
  delta >= 0.33 (medium), on at least one coherent block of cells (not cherry-picked
  single cells).
- Gate G (population justified): ma_xr beats BOTH ma_rand and lns_mitm in success rate
  (>= 10 points) or in time-to-optimum (Cliff's delta >= 0.147, small) on the cells
  where they differ. If not met, the algorithm is reported as an exact-move LNS, not a GA.

Paths:
- N and G met    -> GPEM resubmission: memetic algorithm with exact recombination, regime
                    map as the "when/why" evidence, plateau theory as the "why".
- N met, G not   -> the method is an LNS matheuristic; GPEM fit is weak -> discuss venue
                    with Rohit (e.g. a heuristics / OR / fuzzy-systems journal).
- N not met      -> no honest "use FH-GA when..." claim; retarget as a benchmark +
                    protocol + plateau-theory paper to a better-fitting venue.

## Phase 10 — Tests the round-2 comments need (queued 2026-10-02, `scripts/run_r3_queue.sh`)

Rohit (2026-10-02): "please do tests now needed for current comments". Runs start automatically
after 9.7, so no two experiments share wall-clock time. Shared definitions: `src/protocol/r3.py`
(one solver factory, one protocol); driver `experiments/35_r3_runs.py`.

Protocol: wall-clock budget by size, identical for all solvers (n<=100: 2 s; <=500: 5 s; >500: 10 s;
regime map and sensitivity: fixed 10 s); stop at the certified optimum where known (witness mu = 1,
Suite U enumeration, exact DP): 650/866 FSSP-2 and 97/97 Suite H instances. Seeds: 20 for the new
solvers and (raised 2026-10-02, Rohit: "very strong defense") 20 for stochastic baselines on
Suite H, FSSP-2 and reconciliation; 5/5 on the regime map and the Phase-11 stress corpora. FLSSS uses the fixed wrapper.
New comparator for R4b.1: `vanilla_ga_crisp`, a conventional GA steered by the shared crisp guide.

- [x] 10.3 Regime map, fresh corpus `benchmarks/fsspb_regime_final` (198 inst, seed base 550000,
      3 per cell), all 14 solvers, 5 seeds, 10 s. -> results/r3_regime. (AEb.1, R4b.1 when/why)
- [x] 10.2 Suite H, all 97 instances (round-2 table reported 73: resolved by running all).
      -> results/r3_hard. (R4b.1 hard instances)
- [x] 10.1 Benchmark-FSSP-2 main comparison (T/A/U, 866), all solvers. Statistics with the
      recommended configuration as the reference arm (R1b.1): Friedman -> Wilcoxon + Holm ->
      Cliff's delta, per suite; success rate vs certified optimum; time-to-optimum. -> results/r3_main.
- [ ] 10.4 Sensitivity of the only remaining parameters: k in {16,24,32,36} x pop in {8,16,32} (ma_xr),
      k sweep for lns_mitm, on the 9.7 hard corpus, 3 seeds, 10 s. -> results/r3_sensitivity. (R2b.1)
- [ ] 10.5 Analysis + tables/figures for the new manuscript (regime heat-map first).

## Phase 11 — Strong-defence tests (Rohit 2026-10-02: "do all possible tests, very strong defense")

Each test pre-empts a specific reviewer objection. All of them are in `scripts/run_r3_queue.sh`
after Phase 10. New code: anytime trace in the protocol (every improvement: time, evals, mu);
CP-SAT worker option; MITM cap option; lattice baseline `src/solvers/lattice.py`
(CJLOSS embedding, LLL then BKZ 10/20/30, randomised restarts, n <= 150).

FINDING while building it (before any queued run): lattice reduction solves the critical-block
instances (n = 46, 38-46 bits) in 0.06-0.8 s, where ma_xr reached about 0.5 in the pilot. So the
critical / low-density regime belongs to lattice reduction, and the regime map must say so.
On dense 40-bit n = 100 it finds nothing in 5 s. Lattice is now in every queued corpus.

- [x] 11.1 Beyond MITM's memory reach: n in {56,64,80}, 46-47 bits (24 inst), 30 s, 5 seeds.
      Objection: "your hard cells are just MITM / lattice territory".
- [x] 11.2 Scale: n in {2000,5000,10000} x {32,39} bits (54 inst), 30 s. Objection: "only at
      your sizes". GA baselines are excluded at this size (population lists of 4n x n do not fit
      in memory per worker); this is stated, not hidden.
- [x] 11.3 Payment reconciliation (48 synthetic instances, amounts in cents, 5-40 invoices per
      payment, asymmetric fee tolerance): `experiments/36_generate_reconcile.py`. Objection:
      "is any of this practical?" Also motivates the asymmetric fuzzy target.
- [x] 11.4 Handicap checks on the 9.7 hard corpus: (a) CP-SAT with 8 workers, run alone;
      (b) MITM with max_n = 50; (c) 60 s budgets for the new solvers, FLSSS, CP-SAT and lattice.
      Objection: "you crippled the baselines / the budget was too short".
- [ ] 11.5 Anytime curves from the traces (all Phase 10-11 runs). Objection: "end-of-budget
      snapshots hide the dynamics".
- [x] 11.6 KK / CKK baselines (`src/solvers/kk.py`; Karmarkar-Karp differencing and Korf's complete
      anytime version, via the reduction subset sum -> partition with d = |S - 2T|; parity union-find
      reconstructs explicit subsets; CKK exact on 60/60 brute-force cases). Required by the novelty check
      (paper_fragments/novelty_check.md): KK is the standard heuristic for many large numbers. In the
      default solver list for every stage not yet started; `scripts/run_r3_queue2.sh` adds them to the
      regime map, beyond, scale and hard corpora (plus lattice on the hard corpus).
- [x] 11.7 CMSA-style ablation `ma_xr_cpsat`: the same exact recombination with CP-SAT as the
      sub-solver (0.2 s cap per move) vs ma_xr, on the hard corpus, 5 seeds, 30 s. Answers "why MITM
      rather than an ILP/CP sub-solver, as in CMSA or kernel search?".
- [x] 11.8 Lattice without the n <= 150 cap on the regime map (`scripts/run_r3_queue3.sh`). Reason: in the
      10.3 regime map, lattice solves 0.89 of dense n = 100 cells vs ma_xr 0.80, and shows 0 at n >= 200
      only because of the cap. Before claiming dense n >= 200, it must be shown that lattice fails there.

### 10.3 outcome (2026-10-02), success = mu exactly 1, 5 seeds, 10 s, 198 instances (no KK/CKK yet: queue 2)
- Critical block (n ~ bits): lattice 1.00 in every bit class; ma_xr 1.00 / 1.00 / 0.88 / 0.97 / 0.52 / 0.13
  at 24 / 32 / 38 / 40 / 46 / 47 bits. => lattice reduction is the tool there.
- Dense block: 16 and 24 bits, ma_xr = FLSSS = 1.00; 32 bits, ma_xr 1.00 vs best other (FLSSS) 0.31;
  40 bits, ma_xr 0.33 vs best other (lattice) 0.14.
- Dense by n: n = 100, lattice 0.89 > ma_xr 0.80 (!); n >= 200, ma_xr 0.83-0.86 vs FLSSS 0.58-0.64
  (lattice capped, see 11.8).
- Metaheuristics with the crisp guide (ILS, SA, conventional GA, FH-GA v2): <= 0.28 dense, ~0 critical.

### Phase 10-11 outcomes so far (2026-10-03); success = mu exactly 1 unless stated
- 10.1 FSSP-2 (866 inst, 20 seeds new and baselines, 146,354 runs): ma_xr top or tied on every suite.
  Mean mu: T 1.0000, A 0.9039 (best), U 0.7469 (= the exact optimum on all 384). Against FLSSS:
  T 0/122/0, A 159/198/3 (delta +0.038, negligible), U 11/373/0. Against CKK: A delta +0.374 (medium,
  316/33/11). Against conventional crisp GA: A delta +0.259 (small), T +0.664 (large). The strong
  methods bunch at the top because FSSP-2 magnitudes are small; FSSP-2 shows the method has no weak
  suite, not where it is uniquely needed.
- 10.2 Suite H (97 inst, 20 seeds): ma_xr, ma_rand, lns_mitm, CP-SAT, FLSSS, CKK all 1.000; DP 0.990;
  fhga_v2crisp 0.934; ILS 0.922; KK 0.835; lattice 0.639; SA 0.341; random 0.211; crisp GA 0.139.
  Suite H discriminates blind / plain-GA search from strong methods, NOT among strong methods.
- 11.3 Reconciliation (48 inst, synthetic, amounts in cents): ma_xr 1.00 at n = 200/1000/5000;
  CKK ALSO 1.00 everywhere and faster (median 1-16 ms vs ma_xr 20-170 ms); KK 0.81-1.00; FLSSS 1.00 at
  n <= 1000 but 0.12 at 5000; CP-SAT <= 0.31; every metaheuristic ~0. => On this realistic dense
  workload a classical differencing method is as good and faster; the paper must say so.
- 11.1 Beyond MITM (n 56-80, 46-47 bits, 30 s): only lattice solves anything (0.542); every other
  solver, ma_xr included, 0.000. => moderate density at n < ~100 is lattice territory.
- 11.2 Scale (n 2000-10000, 32/39 bits, 30 s): ma_xr 1.00 at 32 bits and 0.84-0.91 at 39 bits;
  lns_mitm 0.69-1.00; FLSSS 0.78 only at n2000-b32; CP-SAT, ILS, SA, random all 0. KK/CKK pending (queue 4).
- 11.4a CP-SAT with 8 workers on the hard corpus (30 s): 0.000 success (mean mu 0.589).
- 11.4b MITM with max_n = 50: solves every instance it can run on (40 critical; 48 dense infeasible).
OPEN AND PIVOTAL: KK/CKK on the regime map and the scale corpus (queue 4, re-ordered first). If CKK
matches ma_xr on dense >= 32-bit instances as it did on reconciliation, the "when to use" claim must
narrow further, or shift to the instances where CKK fails.

### Pivotal results (2026-10-03, queue 4)
- 11.7 DECISIVE NOVELTY TEST, the same algorithm with a CP-SAT sub-solver (the Rothberg-2007 MIP/CP
  template): hard corpus, 30 s, 5 seeds: ma_xr_cpsat 0.00 (critical) / 0.00 (dense) vs ma_xr 0.57 / 0.73.
  The meet-in-the-middle sub-solver IS the contribution; a general solver in its place solves nothing.
- 11.6 KK/CKK: scale (n 2000-10000), CKK 0.11-0.56 and KK identical, vs ma_xr 0.84-1.00. Regime dense:
  CKK 0.78 / 0.61 / 0.39 / 0.25 at 16 / 24 / 32 / 40 bits, vs ma_xr 1.00 / 1.00 / 1.00 / 0.33. Hard corpus
  dense: CKK 0.23 vs ma_xr 0.64. Beyond-MITM: 0.00. => CKK's reconciliation win does not carry over; it is
  specific to that workload (very dense, few invoices per payment).
- 11.8 Lattice without the cap (regime map, 10 s): n = 100, 1.00 / 1.00 / 1.00 / 0.89 at 16-40 bits (beats
  ma_xr 0.20 at 40 bits); n = 200, 1.00 up to 32 bits (ties ma_xr), 0.44 at 40 bits (ma_xr 0.40); n = 500,
  0.78 at 16 bits and 0 above; n = 1000, 0. => Lattice owns n <= ~200; it collapses from n ~500.
- 11.4c 60 s budget, hard corpus dense: ma_xr 0.64 (30 s) -> 0.97 (60 s); lattice 0.25, FLSSS 0.02,
  CP-SAT 0.00. Critical: lattice 1.00, ma_xr 0.64. More time helps only ma_xr.

FINAL REGIME MAP (the "when" answer for AEb.1 / R4b.1):
- n <= ~44 or small magnitudes (DP table fits): exact methods (MITM, DP).
- Few-to-moderate items with large magnitudes (critical density, n <= ~200): lattice reduction.
- Very dense reconciliation-type workloads (few items per payment): CKK as good and faster.
- MANY items (n >= ~500, up to 10,000 tested) with large magnitudes (>= 24-32 bits): ma_xr, the only
  method that works (0.84-1.00 vs <= 0.56 for every competitor), and it keeps improving with time.

### 12.1 k-tree results, all corpora (2026-10-03) -- risk A1 resolved as a NARROWING, not a collapse
k-tree WINS: dense uniform at 39-40 bits (regime dense-40: 1.00 vs ma_xr 0.20-0.42; scale b39: 1.00 vs
0.84-0.91; hard dense: 1.00 vs 0.64 at 30 s, 0.97 at 60 s); beyond-MITM 0.67 (lattice 0.54, ma_xr 0.00).
ma_xr WINS, often heavily, on STRUCTURED distributions (FSSP-2, 20 vs 5 seeds): T geometric 1.000 vs 0.047,
heavy-tail 1.000 vs 0.955, bimodal 1.000 vs 0.985; A near-equal 0.983 vs 0.827, Avis-Lai 0.777 vs 0.647.
W/T/L ma_xr vs k-tree: T 93/29/0, A 250/82/28, U 6/378/0. Ties: reconciliation (CKK, k-tree, ma_xr all
1.00), Suite H (both 1.00), dense <= 32 bits (both 1.00; ma_xr 6-60x faster). Critical: lattice > both.
=> k-tree is a specialist for uniform random dense instances; it breaks on structured item distributions,
where ma_xr does not. The defensible claim becomes robustness across structure (never worse, often far
better) plus speed, with k-tree / lattice / CKK named as winners in their home regimes. Alternatively
(Rohit to decide): a hybrid move set (MITM exact move + k-tree move inside the memetic algorithm),
which would need a fresh pre-registered confirmation.

## Phase 12 — Robustness tests required before writing (PRE-REGISTERED 2026-10-03, before any run)

Rohit (2026-10-03): "if any other tests ... required please perform". The robustness claim ("robust across
item structure") was formed from FSSP-2 AFTER the results, so it needs fresh-data confirmation.

- [x] 12.3 Structure confirmation, fresh corpus `benchmarks/fsspb_structured` (168 + 12 parity = 180 inst,
      seed base 440000; 7 distributions x n {200,1000,5000} x planted cardinality {10, n/4-3n/4} x gap
      {exact,16} x asymmetry rotation {1:1, 1:20, 20:1}; max item ~2^40). 30 s. ma_xr and ktree 5 seeds;
      ckk, kk, flsss, flsss_exact, cpsat, lattice (n <= 250) once. Success = mu >= mu* - 1e-9.
      R1 (never meaningfully worse): for EACH of the six non-uniform distributions, ma_xr mean mu >=
          (best competitor mean mu) - 0.01.
      R2 (beats the dense specialist on structure): ma_xr success exceeds ktree's by >= 20 points,
          averaged over the six non-uniform distributions.
      Uniform is the control (no criterion; ktree expected >= ma_xr). Any distribution failing R1 is
      named in the paper together with its winner.
- [x] 12.4 Unreachable apex at scale: the 12 parity instances inside 12.3 (mu* = 1 - 1/dL proven). Reported
      as success and mean gap to mu*; no separate criterion.
- [x] 12.2 Speed equalisation: the Python-implemented baselines (ckk, ktree, ils_crisp, vanilla_ga_crisp,
      sim_anneal_crisp, fhga_v2crisp) at 120 s (4-12x their earlier budgets), 1 seed, on
      `benchmarks/subset_speed` (structured geometric n >= 1000, dense n1000-b32, scale n5000-b32). If one
      reaches ma_xr's success there (ma_xr data from the earlier runs at 10-30 s), the "slowness" defence
      fails for it and the paper says so.
- [x] 12.5 Strong configurations: lattice uncapped at 60 s on dense n >= 500 (regime map); flsss_exact on
      the regime map, scale and hard corpora; cpsat_closest with 8 workers (run alone) on scale and hard.
- [x] 12.6 Timing robustness: hard-corpus dense cells, ma_xr and ktree, seed 0, 30 s, ONE worker. Criterion:
      each solver's success within +-10 points of its 6-worker value.
- [ ] 12.7 Analysis only: bootstrap 95% CIs per regime cell; anytime curves from the traces.

### Phase 12 outcomes (2026-10-03)
- 12.3 PRE-REGISTERED CONFIRMATION, fresh 168 instances, 30 s: R1 PASS on all six non-uniform
  distributions. ma_xr mean mu 1.000 (avislai, geometric, heavytail, loguniform, nearequal) and 0.979
  (bimodal); best competitor 0.933 (ktree), 1.000 (ckk), 1.000 (flsss_exact), 1.000 (ckk), 0.958 (ktree)
  and 0.583 (ckk). R2 PASS: ma_xr success minus ktree = +32.8 points over the structured distributions.
  Control (uniform): ktree 1.000 vs ma_xr 0.375, as expected. ma_xr success 0.98-1.00 at every n
  (200 / 1000 / 5000) and both planted cardinalities. Lattice is strong only at n = 200.
- 12.4 Parity (unreachable apex, mu* = 1 - 1/dL proven): ma_xr 1.000 success, gap to mu* 0.000; ktree 0.683
  (gap 0.307); ckk 0.583; kk 0.333; lattice 0.250; FLSSS, FLSSS-exact, CP-SAT 0.000.
- 12.2 Speed equalisation (Python baselines at 120 s = 4-12x their earlier budgets): ILS, SA, conventional
  crisp GA and FH-GA v2 stay at 0.0 on dense n1000-b32 and scale n5000-b32; CKK 0.44 / 0.11; ktree 0.38 on
  structured geometric; ma_xr 1.00 on all three (10-30 s). Slowness does not explain the gaps.
- 12.5 Strong configurations: lattice 60 s, n = 500, 1.00 / 1.00 / 0.67 / 0.11 at 16 / 24 / 32 / 40 bits;
  n = 1000, 0.00. flsss_exact (whole budget on the alpha = 1 probe) is stronger than FLSSS: dense-32 0.75,
  scale-32 0.59, scale-39 0.26, hard dense 0.06; still below ma_xr. cpsat_closest with 8 workers: 0.00 on
  scale and on the hard corpus.

### 12.6 outcome: FAILED (pre-registered criterion), with the cause found
One worker vs the 6-worker runs, hard corpus seed 0: ma_xr dense 0.958 vs 0.688 (+27 pts); ma_xr critical
0.600 vs 0.450 (+15); ktree critical 0.675 vs 0.450 (+22.5); ktree dense 1.000 vs 1.000.
Cause: the machine is an Apple M1 with 4 performance + 4 efficiency cores (8 GB RAM). With 6-7 workers some
runs land on efficiency cores and get roughly half the effective compute in the same wall-clock budget.
This is noise, not a bias towards any solver, but it depresses absolute success and makes CLOSE comparisons
unreliable. Large gaps (e.g. structured 1.00 vs 0.4) are not at risk; close ones are.

## Phase 13 — Re-run the wall-clock headline experiments on performance cores only (PRE-REGISTERED)
- [x] 13.0 Gate (PASS 2026-10-03: 4 workers vs 1 worker, ma_xr dense -8.3, critical -2.5; ktree dense 0.0, critical -7.5 points): hard corpus, ma_xr and ktree, seed 0, 30 s, 4 workers vs the one-worker run (12.6). Each
      solver within +-10 points in each block; otherwise drop to 3 workers and repeat.
- [x] 13.1 Gate G re-run (ma_xr / ma_rand / lns_mitm, 88 inst, 5 seeds, 30 s, 4 workers); same thresholds
      as 9.7 (>= 10 points vs both, or time-to-optimum delta <= -0.147 vs both).
- [ ] 13.2 Structured confirmation re-run (12.3 corpus and solvers, 4 workers); same R1 / R2 criteria.
- [ ] 13.3 Regime map re-run (ma_xr, ktree, ckk, flsss, flsss_exact, cpsat, lattice uncapped; 5 seeds,
      10 s, 4 workers; lattice 3 workers because BKZ is memory-heavy).
- [ ] 13.4 Scale re-run (ma_xr, ktree, ckk, flsss_exact; 30 s, 5 seeds, 4 workers).
- [x] 13.5 CP-SAT sub-solver ablation re-run (ma_xr_cpsat vs ma_xr, 30 s, 5 seeds, 4 workers).
Not re-run, stated in the paper: FSSP-2 main, Suite H and reconciliation (small magnitudes; almost all runs
stop at the known optimum within milliseconds, so contention cannot change their outcome). Every paper
number from Phase 13 replaces the corresponding earlier number. Hardware is reported: Apple M1, 4 P-cores used.

### Phase 13 outcomes (performance cores only)
- 13.1 Gate G MET on the clean re-run. ma_xr vs ma_rand: success 0.732 vs 0.639 (+9.3 points, below the
  10-point threshold) but time-to-optimum Cliff delta -0.187 (passes the second criterion), Wilcoxon
  p = 3.7e-3. ma_xr vs lns_mitm: 0.732 vs 0.559 (+17.3), delta -0.354, p = 3.2e-6. As before, the gain is
  in the dense cells (0.833 vs 0.667 / 0.633); critical ma_xr = ma_rand (0.610 vs 0.605). Report the
  success margin vs ma_rand honestly as +9.3 points, with the gate passing on time-to-optimum.
- 13.5 CP-SAT sub-solver ablation: ma_xr_cpsat 0.000 / 0.000 (critical / dense) vs ma_xr 0.610 / 0.833.

