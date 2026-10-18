"""Figure: exact recombination on a worked example (Section 4.2 of the manuscript).

A 10-item instance with an asymmetric triangular target. Two parents disagree on six items;
the child inherits the four agreed items and ExactMove (the project's closest_subset, i.e. the
meet-in-the-middle of Algorithm 1) reassigns the disagreement set. Every number drawn is computed
here and checked: the MITM result against brute force over all 2^6 assignments, and every mu by
the instance target's membership function.

The example was selected by searching seeds 0..19999 for parents with 0.25 < mu < 0.75 on
opposite sides of T^M whose recombination reaches mu = 1 and differs from both parents in at
least two items; seed 13069 is the first such seed. For illustration |D| = k = 6 (MXR-GA uses k = 36).

Outputs: paper_fragments/r3/figures/fig_recombination.{pdf,png}
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.core.fuzzy import TriangularFuzzyNumber  # noqa: E402
from src.solvers.exact_lns import _half_sums, closest_subset  # noqa: E402

OUT = REPO / "paper_fragments" / "r3" / "figures"
C_PROP, C2, C_GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, BAND = "#0b0b0b", "#52514e", "#fde8dc"

SEED, DL, DR = 13069, 200, 50


def build():
    rng = np.random.default_rng(SEED)
    n = 10
    a = rng.integers(100, 1000, n)
    u = rng.integers(0, 2, n)
    tm = int(a @ u) + int(rng.integers(60, 140))
    D = np.sort(rng.choice(n, 6, replace=False))
    v = u.copy()
    v[D] = 1 - v[D]
    T = TriangularFuzzyNumber(tm - DL, tm, tm + DR)
    fixed = np.setdiff1d(np.arange(n), D)
    r = tm - int(a[fixed] @ u[fixed])
    dstar, mask = closest_subset(a[D], r, DL, DR)
    brute = min(((r - s) / DL if s <= r else (s - r) / DR)
                for k in range(len(D) + 1) for s in map(sum, itertools.combinations(a[D], k)))
    assert abs(brute - dstar) < 1e-12, (brute, dstar)
    c = u.copy()
    c[D] = mask.astype(int)
    return a, u, v, c, D, T, r


def main() -> int:
    a, u, v, c, D, T, r = build()
    n = len(a)
    mu = {k: T.mu(int(a @ x)) for k, x in (("u", u), ("v", v), ("c", c))}
    sig = {k: int(a @ x) for k, x in (("u", u), ("v", v), ("c", c))}
    # MITM halves exactly as closest_subset forms them: first floor(k/2) items of F, then the rest.
    h = len(D) // 2
    F1, F2 = D[:h], D[h:]
    L = _half_sums(a[F1])
    R = np.sort(_half_sums(a[F2]))
    ell = int(a[F1] @ c[F1])
    rho = int(a[F2] @ c[F2])
    assert ell + rho == r and ell in L and rho in R
    print(f"a={a.tolist()} T=({T.T_L:g},{T.T_M:g},{T.T_R:g}) D(1-based)={(D + 1).tolist()} r={r}")
    print(f"u={u.tolist()} sum={sig['u']} mu={mu['u']:.2f} | v={v.tolist()} sum={sig['v']} mu={mu['v']:.2f}"
          f" | child={c.tolist()} sum={sig['c']} mu={mu['c']:.2f}")
    print(f"L={L.tolist()} R={R.tolist()} ell={ell} rho={rho}")

    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(5.2, 3.55))
    ax = fig.add_axes([0.0, 0.0, 0.68, 1.0])
    ax.set_xlim(-2.6, n + 2.4)
    ax.set_ylim(-0.2, 9.6)
    ax.axis("off")

    def cell(x, y, on, edge=INK2, txt=None):
        ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, facecolor=C_PROP if on else "white",
                               edgecolor=edge, linewidth=0.7))
        if txt is not None:
            ax.text(x + 0.5, y + 0.5, txt, ha="center", va="center", color="white" if on else INK, fontsize=6.5)

    # disagreement-set band
    for i in D:
        ax.add_patch(Rectangle((i, 5.0), 1, 3.95, facecolor=BAND, edgecolor="none", zorder=0))
    ax.text(n + 0.15, 8.45, "$i$", ha="left", va="center", color=INK2)
    for i in range(n):
        ax.text(i + 0.5, 8.45, str(i + 1), ha="center", va="center", color=INK2)
        ax.text(i + 0.5, 7.75, str(int(a[i])), ha="center", va="center", fontsize=6.5)
    ax.text(n + 0.15, 7.75, "$a_i$", ha="left", va="center", color=INK2)
    rows = (("parent $u$", u, 6.55, "u"), ("parent $v$", v, 5.3, "v"))
    for lab, x, y, k in rows:
        ax.text(-0.2, y + 0.5, lab, ha="right", va="center")
        for i in range(n):
            cell(i, y, x[i])
        ax.text(n + 0.15, y + 0.5, f"$\\sigma$={sig[k]}\n$\\mu$={mu[k]:.2f}", ha="left", va="center", fontsize=6.5)
    ax.text(D.mean() + 0.5, 9.25, "disagreement set $D$ = free set $F$", ha="center", va="center", color=C2)

    # exact move box
    agreed = int(a[np.setdiff1d(np.arange(n), D)] @ u[np.setdiff1d(np.arange(n), D)])
    ax.text(-2.5, 4.45, f"ExactMove on $F$: residual $r = T^M - \\sum_{{i\\notin F}} a_i u_i = {int(T.T_M)} - {agreed} = {r}$", ha="left", va="center")

    def lst(vals, y, name, hit):
        ax.text(-0.2, y, name, ha="right", va="center", color=INK2)
        for j, s in enumerate(vals):
            on = int(s) == hit
            ax.text(j * 1.25 + 0.5, y, str(int(s)), ha="center", va="center", fontsize=6.5,
                    color=C2 if on else INK, fontweight="bold" if on else "normal")
    f1 = ",".join(str(int(i + 1)) for i in F1)
    f2 = ",".join(str(int(i + 1)) for i in F2)
    lst(L, 3.6, f"sums of $F_1$={{{f1}}}", ell)
    lst(R, 2.85, f"sorted sums of $F_2$={{{f2}}}", rho)
    ax.text(-2.5, 2.05, f"for $\\ell$={ell}: $r-\\ell$={r - ell} found in $R$ by binary search "
                        f"(all $2^6$ assignments covered)", ha="left", va="center", color=INK2, fontsize=6.5)

    ax.text(-0.2, 0.75, "child", ha="right", va="center")
    for i in range(n):
        cell(i, 0.25, c[i], edge=C2 if i in D else INK2)
    ax.text(n + 0.15, 0.75, f"$\\sigma$={sig['c']}\n$\\mu$={mu['c']:.2f}", ha="left", va="center", fontsize=6.5)
    ax.annotate("", xy=(n / 2, 1.35), xytext=(n / 2, 1.75),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))

    # membership panel
    bx = fig.add_axes([0.75, 0.30, 0.24, 0.55])
    xs = np.linspace(T.T_L - 60, T.T_R + 40, 400)
    bx.plot(xs, [T.mu(x) for x in xs], color=INK2, lw=1)
    for k, col, mk in (("u", C_GRAY, "o"), ("v", C_GRAY, "s"), ("c", C2, "D")):
        bx.plot(sig[k], mu[k], mk, color=col, ms=5, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        bx.annotate({"u": "$u$", "v": "$v$", "c": "child"}[k], (sig[k], mu[k]), textcoords="offset points",
                    xytext=(-12 if k == "u" else 4, 4 if k != "c" else 3), fontsize=6.5)
    bx.set_xticks([T.T_L, T.T_M, T.T_R], ["$T^L$", "$T^M$", "$T^R$"], fontsize=6.5)
    bx.set_yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=6)
    bx.set_ylim(-0.03, 1.12)
    bx.set_xlabel("subset sum $\\sigma$", fontsize=6.5)
    bx.set_ylabel("$\\mu$", fontsize=6.5)
    bx.spines[["top", "right"]].set_visible(False)
    bx.set_title(f"$\\tilde T$ = ({T.T_L:g}, {T.T_M:g}, {T.T_R:g})", fontsize=6.5)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_recombination.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_recombination.png", dpi=220, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
