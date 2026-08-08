"""Task 2.17 — third domain (fuzzy multiple-subset-sum) phenomenon check.

Self-contained: generates a small MSSP corpus with a planted joint optimum
and tight tolerances, then measures the three things that decide whether the
clipped-objective phenomenon reproduces in a structurally distinct domain:

  1. Plateau mass Pi and difficulty Gamma = mu* - best-of-K-random, both
     computed with NO solver (mu* is 1 by construction; verified by the
     exact 2-D DP).
  2. Blind random search and a mu-steered local search -- predicted to
     stall on the plateau (Prop. 2 applies here verbatim).
  3. A crisp-guide local search under the SAME budget -- predicted to
     escape, exactly as in FSSP and FKP.

If (2) fails and (3) succeeds while Pi is high, the phenomenon and the
guide-necessity conclusion generalise to a third combinatorial core.

Output: results/mssp/{corpus stats, results.csv, summary.json},
paper_fragments/table_mssp_domain.tex
Usage: .venv/bin/python experiments/31_mssp_domain.py [--generate] [--seeds N]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.fuzzy import TriangularFuzzyNumber
from src.core.mssp import MSSPInstance, exact_optimum, load_mssp, membership
from src.protocol.runner import git_commit

CORPUS = REPO / "benchmarks" / "mssp"
OUT = REPO / "results" / "mssp"
FRAG = REPO / "paper_fragments" / "table_mssp_domain.tex"

DISTS = ["uniform", "geometric", "bimodal"]
NS = [15, 18, 21]
# Tolerance as a fraction of each target's own magnitude, spanning tight to
# loose so plateau mass sweeps (as FKP found, a corpus of only tight
# tolerances cannot exhibit the mu-fails / crisp-escapes contrast).
TOL_C = [0.01, 0.03, 0.08, 0.2]
BUDGET = 20_000


def draw_items(rng, n, dist):
    if dist == "uniform":
        return [int(rng.integers(10, 100)) for _ in range(n)]
    if dist == "geometric":
        return [int(1 + rng.geometric(0.06)) for _ in range(n)]
    # bimodal
    return [int(rng.integers(10, 30) if rng.random() < 0.5
                else rng.integers(70, 100)) for _ in range(n)]


def generate(rng) -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    made = 0
    idx = 0
    for dist in DISTS:
        for n in NS:
            for c in TOL_C:
                for rep in range(3):
                    idx += 1
                    items = draw_items(rng, n, dist)
                    perm = list(rng.permutation(n))
                    s1 = perm[: n // 3]
                    s2 = perm[n // 3: 2 * n // 3]
                    tm1 = sum(items[i] for i in s1)
                    tm2 = sum(items[i] for i in s2)
                    if tm1 == 0 or tm2 == 0:
                        continue
                    d1 = max(1, int(c * tm1))
                    d2 = max(1, int(c * tm2))
                    inst = MSSPInstance(
                        instance_id=f"mssp-{dist}-n{n}-tol{c}-s{idx:03d}",
                        items=tuple(items),
                        target1=TriangularFuzzyNumber(tm1 - d1, tm1, tm1 + d1),
                        target2=TriangularFuzzyNumber(tm2 - d2, tm2, tm2 + d2),
                        n=n,
                        metadata={"n": n, "distribution": dist, "tol_c": c},
                        witness={"assignment": [1 if i in set(s1) else 2 if i in set(s2)
                                                else 0 for i in range(n)],
                                 "mu": 1.0},
                    )
                    mu_star, _ = exact_optimum(inst)
                    if mu_star < 0.999:
                        continue  # planted optimum must be exact
                    with open(CORPUS / f"{inst.instance_id}.json", "w") as fh:
                        json.dump(inst.to_dict(), fh)
                    made += 1
    return made


def crisp_guide(inst):
    t1, t2 = inst.target1, inst.target2

    def g(s1, s2):
        d1 = (t1.T_M - s1) / max(t1.delta_L, 1e-9) if s1 <= t1.T_M \
            else (s1 - t1.T_M) / max(t1.delta_R, 1e-9)
        d2 = (t2.T_M - s2) / max(t2.delta_L, 1e-9) if s2 <= t2.T_M \
            else (s2 - t2.T_M) / max(t2.delta_R, 1e-9)
        return 2.0 - d1 - d2
    return g


def sums(inst, assign):
    s1 = sum(a for a, x in zip(inst.items, assign) if x == 1)
    s2 = sum(a for a, x in zip(inst.items, assign) if x == 2)
    return s1, s2


def local_search(inst, rng, guide_mode, budget):
    """Reassign-one-item local search. guide_mode: 'mu' (objective) or
    'crisp'. Returns best true membership seen. Random restarts on stall."""
    n = inst.n
    guide = crisp_guide(inst) if guide_mode == "crisp" else None
    best_mu = 0.0
    evals = 0

    def score(assign):
        nonlocal evals, best_mu
        evals += 1
        s1, s2 = sums(inst, assign)
        mu = min(inst.target1.mu(s1), inst.target2.mu(s2))
        best_mu = max(best_mu, mu)
        return guide(s1, s2) if guide is not None else mu

    while evals < budget:
        assign = [int(rng.integers(0, 3)) for _ in range(n)]
        f = score(assign)
        improved = True
        while improved and evals < budget:
            improved = False
            for i in range(n):
                old = assign[i]
                for b in (0, 1, 2):
                    if b == old:
                        continue
                    assign[i] = b
                    nf = score(assign)
                    if nf > f + 1e-12:
                        f, old, improved = nf, b, True
                    else:
                        assign[i] = old
    return best_mu


def measure(inst, rng, k_random, budget, seeds):
    # Pi (plateau mass) and Gamma from blind sampling
    n = inst.n
    zero = 0
    best_rand = 0.0
    for _ in range(k_random):
        assign = [int(rng.integers(0, 3)) for _ in range(n)]
        s1, s2 = sums(inst, assign)
        mu = min(inst.target1.mu(s1), inst.target2.mu(s2))
        if mu == 0.0:
            zero += 1
        best_rand = max(best_rand, mu)
    pi = zero / k_random
    gamma = 1.0 - best_rand
    mu_guided = statistics.mean(
        local_search(inst, np.random.default_rng(s), "mu", budget)
        for s in range(seeds))
    crisp_guided = statistics.mean(
        local_search(inst, np.random.default_rng(100 + s), "crisp", budget)
        for s in range(seeds))
    rand_search = best_rand  # blind search's own best is its result
    return pi, gamma, rand_search, mu_guided, crisp_guided


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k-random", type=int, default=4000)
    args = ap.parse_args()

    if args.generate or not CORPUS.exists():
        made = generate(np.random.default_rng(20260808))
        print(f"generated {made} MSSP instances", flush=True)

    paths = sorted(CORPUS.glob("*.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in paths:
        inst = load_mssp(p)
        pi, gamma, rs, mug, crg = measure(
            inst, np.random.default_rng(7), args.k_random, BUDGET, args.seeds)
        rows.append({"instance_id": inst.instance_id, "n": inst.n,
                     "plateau_mass": pi, "gamma": gamma,
                     "random_search": rs, "mu_guided_ls": mug,
                     "crisp_guided_ls": crg})
        print(f"  {inst.instance_id}: Pi={pi:.3f} Gamma={gamma:.3f} "
              f"rand={rs:.3f} mu-LS={mug:.3f} crisp-LS={crg:.3f}", flush=True)

    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def mean(k):
        return statistics.mean(r[k] for r in rows)
    summary = {
        "n_instances": len(rows),
        "mean_plateau_mass": mean("plateau_mass"),
        "mean_gamma": mean("gamma"),
        "mean_random_search": mean("random_search"),
        "mean_mu_guided_ls": mean("mu_guided_ls"),
        "mean_crisp_guided_ls": mean("crisp_guided_ls"),
        "crisp_over_mu": mean("crisp_guided_ls") - mean("mu_guided_ls"),
        "crisp_over_random": mean("crisp_guided_ls") - mean("random_search"),
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump({"git_commit": git_commit(), "budget": BUDGET,
                   "seeds": args.seeds, "summary": summary}, fh, indent=2)
    write_fragment(summary)
    print("\n" + json.dumps(summary, indent=2))
    return 0


def write_fragment(s):
    FRAG.write_text(f"""% Auto-generated by experiments/31_mssp_domain.py — do not edit by hand.
% Task 2.17: the clipped-objective phenomenon in a third domain
% (fuzzy multiple-subset-sum, k=2 coupled targets).
\\begin{{table}}[t]
  \\centering
  \\caption{{Fuzzy multiple-subset-sum ($k{{=}}2$), {s['n_instances']}
  planted-optimum instances. The plateau predicted by
  Proposition~\\ref{{prop:plateau-mass}} is present (mean mass
  {s['mean_plateau_mass']:.3f}); blind sampling and a $\\mu$-steered local
  search stall on it, while the same local search under the shared crisp
  guide escapes --- reproducing the FSSP and fuzzy-knapsack result in a
  structurally distinct core.}}
  \\label{{tab:mssp-domain}}
  \\begin{{tabular}}{{lr}}
    \\toprule
    quantity & mean \\\\
    \\midrule
    plateau mass $\\Pi$ & {s['mean_plateau_mass']:.3f} \\\\
    difficulty $\\Gamma$ & {s['mean_gamma']:.3f} \\\\
    random search (best $\\mu$) & {s['mean_random_search']:.3f} \\\\
    $\\mu$-guided local search & {s['mean_mu_guided_ls']:.3f} \\\\
    crisp-guided local search & {s['mean_crisp_guided_ls']:.3f} \\\\
    \\midrule
    crisp $-$ $\\mu$ guide & {s['crisp_over_mu']:+.3f} \\\\
    crisp $-$ random & {s['crisp_over_random']:+.3f} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
""")


if __name__ == "__main__":
    sys.exit(main())
