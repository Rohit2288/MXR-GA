"""Generate the FH-GA figures as vector PDFs, reviewer-consistent.

Replaces the draw.io PNGs. Content corrected per the revision:
  - "direction-guided" local search, not "gradient-guided" (R2.3)
  - windowed repair labelled with the recommended exact DP-window (R2.9/R4.1)
  - termination owned by the shared protocol (budget/plateau), no private
    mu >= 1-eps early exit or generation cap (R4.3)
  - an honest footnote on which mechanisms the ablation credits (R2.8/R4.1)

Output: paper_fragments/figures/fhga_framework.pdf (+ .png preview)
        paper_fragments/figures/fhga_controlflow.pdf (+ .png preview)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "paper_fragments" / "figures"

BLUE, ORANGE, GREEN = "#dbe5f5", "#fdecd2", "#dcefdc"
BLUE_E, ORANGE_E, GREEN_E = "#3b6cb7", "#c9822a", "#3f9a4f"
VIOLET, VIOLET_E = "#e7e3f7", "#6a5aa8"
INK = "#1a1a1a"


def _box(ax, x, y, w, h, fc, ec, title, lines, title_color=None, fs=8.2):
    """lines: list of str, or (str, highlight_bool). Highlighted lines are
    bold and drawn in the box edge colour (used for the reviewer-corrected
    items)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
            fontsize=fs + 1.6, fontweight="bold",
            color=title_color or ec, zorder=3)
    ax.plot([x + 0.05, x + w - 0.05], [y + h - 0.115] * 2, color=ec,
            lw=0.8, zorder=3)
    yy = y + h - 0.155
    dy = 0.0425
    for ln in lines:
        txt, hi = (ln, False) if isinstance(ln, str) else ln
        ax.text(x + w / 2, yy, txt, ha="center", va="top",
                fontsize=fs, zorder=3,
                color=ec if hi else INK,
                fontweight="bold" if hi else "normal")
        yy -= dy


def _pill(ax, x, y, text, fs=7.6):
    ax.add_patch(FancyBboxPatch(
        (x - 0.055, y - 0.028), 0.11, 0.056,
        boxstyle="round,pad=0.006,rounding_size=0.03",
        linewidth=1.0, edgecolor="#888", facecolor="white", zorder=4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=INK, style="italic", zorder=5)


def _arrow(ax, p, q, color=INK, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=13, linewidth=lw,
        color=color, linestyle=ls, zorder=1,
        shrinkA=2, shrinkB=2))


def framework():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Input
    ax.add_patch(FancyBboxPatch((0.30, 0.905), 0.40, 0.075,
        boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=1.4,
        edgecolor=VIOLET_E, facecolor=VIOLET, zorder=2))
    ax.text(0.5, 0.9425,
            r"Input: items $S=\{a_1,\dots,a_n\}$,  fuzzy target "
            r"$\tilde{T}=(T^{L},T^{M},T^{R})$",
            ha="center", va="center", fontsize=9, fontweight="bold", color=INK)

    y, h = 0.30, 0.53
    _box(ax, 0.025, y, 0.30, h, BLUE, BLUE_E, "Population & variation", [
        "Binary chromosome encoding",
        r"Hybrid init: greedy 30%",
        r"$+\ \alpha$-cut 20% $+$ random 50%",
        r"Tournament selection ($k{=}3$)",
        "Fitness-biased uniform",
        "crossover (FB-UX)",
        "Impact-aware two-phase mutation",
        r"Elitist replacement ($\mu{+}\lambda$)",
        "Restart on stagnation",
    ])
    _box(ax, 0.35, y, 0.30, h, ORANGE, ORANGE_E, "Local refinement", [
        ("Direction-guided local search", True),
        (r"(sign of $\Delta\mu$, not a gradient)", True),
        r"single-flip, 1-1 swap (top-$K$),",
        "random kick",
        "Windowed repair for",
        "outside-support offspring",
        ("recommended: exact DP-window", True),
        r"(WBF variant enumerates $2^{w}$)",
        "Lamarckian update",
    ])
    _box(ax, 0.675, y, 0.30, h, GREEN, GREEN_E, "Online fuzzy control", [
        r"Mamdani $3\times3$ inference",
        r"Inputs: diversity $D$, best $\mu^\star$",
        r"Outputs: $p_c,\ p_m$",
        "Triangular MFs, min-norm,",
        "centroid defuzzification",
        ("Termination: shared", True),
        ("budget or plateau", True),
        ("(protocol-owned)", True),
    ])

    # inter-box pills/arrows
    _arrow(ax, (0.325, y + h / 2), (0.35, y + h / 2)); _pill(ax, 0.3375, y + h / 2 + 0.055, "offspring")
    _arrow(ax, (0.65, y + h / 2), (0.675, y + h / 2)); _pill(ax, 0.6625, y + h / 2 + 0.055, "refined pop.")
    # input -> box1
    _arrow(ax, (0.42, 0.905), (0.175, y + h))
    # box3 -> output
    ax.add_patch(FancyBboxPatch((0.28, 0.045), 0.44, 0.075,
        boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=1.4,
        edgecolor=VIOLET_E, facecolor=VIOLET, zorder=2))
    ax.text(0.5, 0.0825,
            r"Output: best subset $x^\star\in\{0,1\}^n$ maximising "
            r"$\mu_{\tilde{T}}(\sum_i a_i x_i)$",
            ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    _arrow(ax, (0.5, y), (0.5, 0.12))
    # feedback loop (dashed) box3 bottom -> box1 bottom
    _arrow(ax, (0.825, y), (0.825, 0.16), color="#777", lw=1.2, ls=(0, (4, 3)))
    _arrow(ax, (0.825, 0.16), (0.175, 0.16), color="#777", lw=1.2, ls=(0, (4, 3)))
    _arrow(ax, (0.175, 0.16), (0.175, y), color="#777", lw=1.2, ls=(0, (4, 3)))
    ax.text(0.5, 0.175, "adaptive $(p_c, p_m)$ feedback per generation",
            ha="center", va="bottom", fontsize=7.6, style="italic", color="#555")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fhga_framework.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fhga_framework.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("framework written")


def _node(ax, x, y, w, h, text, fc, ec, shape="rect", fs=8.2, bold=False):
    from matplotlib.patches import Polygon
    if shape == "diamond":
        ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y + h),
                              (x + w, y + h / 2), (x + w / 2, y)],
                             closed=True, linewidth=1.5, edgecolor=ec,
                             facecolor=fc, zorder=2))
    elif shape == "round":
        ax.add_patch(FancyBboxPatch((x + 0.02, y), w - 0.04, h,
            boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.6,
            edgecolor=ec, facecolor=fc, zorder=2))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.015", linewidth=1.5,
            edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def controlflow():
    fig, ax = plt.subplots(figsize=(7.4, 11.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    cx, w = 0.30, 0.40           # main column centre-x and node width
    xl = cx - w / 2
    PINK, PINK_E = "#f7d9e6", "#c0568a"
    rows = []  # (cy, kind)

    def stack(items):
        yv = 0.965
        for it in items:
            yield yv, it
            yv -= it.get("gap", 0.075)

    seq = [
        dict(t="START: input $S$, $\\tilde{T}$", fc=VIOLET, ec=VIOLET_E, s="round", b=True, h=0.045),
        dict(t="Hybrid initialisation of $P_0$", fc=BLUE, ec=BLUE_E, s="rect", h=0.05),
        dict(t="Evaluate $\\mu$; set $x^\\star,\\ \\mu^\\star$", fc=BLUE, ec=BLUE_E, s="rect", h=0.05),
        dict(t="Generation $t$: offspring pool $Q$", fc=BLUE, ec=BLUE_E, s="rect", b=True, h=0.05),
        dict(t="Selection $\\to$ FB-UX $\\to$\nimpact-aware mutation", fc=BLUE, ec=BLUE_E, s="rect", h=0.06),
        dict(t="$|Q| \\geq N$ ?", fc=PINK, ec=PINK_E, s="diamond", h=0.075),
        dict(t="Refinement per $c \\in Q$:\ndirection-guided local search;\nwindowed DP-repair if $\\mu(c){=}0$", fc=ORANGE, ec=ORANGE_E, s="rect", h=0.085, hi=True),
        dict(t="Elitist replacement\n($e$-elite of $P_{t-1}\\ \\cup$ best of $Q$)", fc=BLUE, ec=BLUE_E, s="rect", h=0.06),
        dict(t="Update $x^\\star,\\mu^\\star$, diversity $D$;\nMamdani $\\to (p_c,p_m)$", fc=GREEN, ec=GREEN_E, s="rect", h=0.06),
        dict(t="Protocol stop?\nbudget exhausted or plateau", fc=PINK, ec=PINK_E, s="diamond", h=0.075, hi=True),
        dict(t="END: return $x^\\star$", fc=VIOLET, ec=VIOLET_E, s="round", b=True, h=0.045),
    ]
    cys = []
    yv = 0.965
    for i, nd in enumerate(seq):
        h = nd["h"]
        cy = yv - h
        _node(ax, xl, cy, w, h, nd["t"], nd["fc"], nd["ec"], nd["s"],
              bold=nd.get("b", False))
        if nd.get("hi"):
            # accent ring for the two reviewer-critical nodes
            ax.add_patch(FancyBboxPatch((xl - 0.006, cy - 0.006), w + 0.012, h + 0.012,
                boxstyle="round,pad=0.004,rounding_size=0.015", linewidth=1.4,
                edgecolor=nd["ec"], facecolor="none", zorder=1, linestyle=(0, (3, 2))))
        cys.append((cy, cy + h))
        yv = cy - 0.0245
    # vertical arrows between consecutive nodes
    for i in range(len(seq) - 1):
        _arrow(ax, (cx, cys[i][0]), (cx, cys[i + 1][1]))
    # loop-back: |Q|>=N? "No" -> back to Selection (node idx 4)
    didx = 5
    ax.text(cx + w / 2 + 0.01, (cys[didx][0] + cys[didx][1]) / 2, "No",
            fontsize=7.6, va="center", color=INK)
    xr = xl + w
    _arrow(ax, (xr, (cys[didx][0] + cys[didx][1]) / 2), (0.86, (cys[didx][0] + cys[didx][1]) / 2))
    _arrow(ax, (0.86, (cys[didx][0] + cys[didx][1]) / 2), (0.86, (cys[4][0] + cys[4][1]) / 2))
    _arrow(ax, (0.86, (cys[4][0] + cys[4][1]) / 2), (xr, (cys[4][0] + cys[4][1]) / 2))
    ax.text(cx, cys[didx][0] - 0.004, "Yes", fontsize=7.6, ha="center", va="top", color=INK)
    # protocol-stop diamond: No -> next generation (loop to node 3); Yes -> END
    sidx = 9
    smid = (cys[sidx][0] + cys[sidx][1]) / 2
    ax.text(xl - 0.008, smid, "No", fontsize=7.6, ha="right", va="center", color=INK)
    _arrow(ax, (xl, smid), (0.055, smid))
    g3 = (cys[3][0] + cys[3][1]) / 2
    _arrow(ax, (0.055, smid), (0.055, g3))
    _arrow(ax, (0.055, g3), (xl, g3))
    ax.text(0.088, (smid + g3) / 2, "next gen.\n$t\\!\\leftarrow\\! t{+}1$",
            fontsize=7.2, ha="center", va="center", color="#555", style="italic")
    ax.text(cx, cys[sidx][0] - 0.004, "Yes", fontsize=7.6, ha="center", va="top", color=INK)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fhga_controlflow.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fhga_controlflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("controlflow written")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm",
                         "font.family": "serif"})
    framework()
    controlflow()
