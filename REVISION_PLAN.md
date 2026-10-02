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
- [ ] 9.7 Confirmatory Gate G test (PRE-REGISTERED here, before the run).
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
solvers, 10 for stochastic baselines (8.5 h of compute at 20 would exceed one overnight run;
the baselines' round-2 numbers already exist at 20 seeds). FLSSS uses the fixed wrapper.
New comparator for R4b.1: `vanilla_ga_crisp`, a conventional GA steered by the shared crisp guide.

- [ ] 10.3 Regime map, fresh corpus `benchmarks/fsspb_regime_final` (198 inst, seed base 550000,
      3 per cell), all 14 solvers, 5 seeds, 10 s. -> results/r3_regime. (AEb.1, R4b.1 when/why)
- [ ] 10.2 Suite H, all 97 instances (round-2 table reported 73: resolved by running all).
      -> results/r3_hard. (R4b.1 hard instances)
- [ ] 10.1 Benchmark-FSSP-2 main comparison (T/A/U, 866), all solvers. Statistics with the
      recommended configuration as the reference arm (R1b.1): Friedman -> Wilcoxon + Holm ->
      Cliff's delta, per suite; success rate vs certified optimum; time-to-optimum. -> results/r3_main.
- [ ] 10.4 Sensitivity of the only remaining parameters: k in {16,24,32,36} x pop in {8,16,32} (ma_xr),
      k sweep for lns_mitm, on the 9.7 hard corpus, 3 seeds, 10 s. -> results/r3_sensitivity. (R2b.1)
- [ ] 10.5 Analysis + tables/figures for the new manuscript (regime heat-map first).

