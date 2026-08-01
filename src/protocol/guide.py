"""Protocol-owned internal search-guidance signal (task 1.9 generalised).

Motivation (R4.3 fairness; R4.1 hard-cut-off analysis)
------------------------------------------------------
The membership mu is flat at exactly zero for every subset whose sum falls
outside the triangular support. On instances where a solver's trajectory
lies outside the support, mu therefore supplies NO search signal at all:
every neighbour scores 0, best-improvement finds no improving move, and
Metropolis acceptance sees delta = 0 everywhere. The search stalls on a
zero plateau rather than being outperformed.

FH-GA escapes this plateau by construction (hybrid initialisation seeds
inside or near the support; the repair operators re-enter it). Steering a
baseline by mu alone while FH-GA enjoys plateau escape is precisely the
asymmetric instrumentation Reviewer 4 objected to in the submitted version.

This module therefore holds ONE definition of the unclamped crisp guide,
imported by every solver that takes a guide. Making it protocol-owned --
rather than re-deriving it per solver -- is what licenses the claim that
all guided solvers are steered by an identical signal.

    g(S) = 1 - d(S),   d(S) = (T_M - S)/delta_L   if S <= T_M
                              (S - T_M)/delta_R   otherwise

Inside the support g equals mu exactly; outside, where mu is flat at zero,
g keeps decreasing and so still points back toward the target.

Reported quality is NEVER the guide. The protocol tracks and the harness
rescores mu for every evaluated subset, identically for every solver
(CLAUDE.md architecture rules 1 and 2).
"""

from __future__ import annotations

from typing import Callable, Optional

from src.core.instance import Instance

Guide = Callable[[float, float], float]


def crisp_guide(instance: Instance) -> Guide:
    """Unclamped asymmetric distance guide g(S, mu) -> float."""
    t = instance.target
    d_l = t.delta_L if t.delta_L > 0 else 1.0
    d_r = t.delta_R if t.delta_R > 0 else 1.0
    t_m = t.T_M

    def guide(S: float, mu: float) -> float:
        d = (t_m - S) / d_l if S <= t_m else (S - t_m) / d_r
        return 1.0 - d

    return guide


def make_guide(mode: str, instance: Instance) -> Optional[Guide]:
    """Factory: 'mu' -> None (steer by membership); 'crisp' -> crisp_guide."""
    if mode == "mu":
        return None
    if mode == "crisp":
        return crisp_guide(instance)
    raise ValueError(f"unknown guide mode {mode!r}")
