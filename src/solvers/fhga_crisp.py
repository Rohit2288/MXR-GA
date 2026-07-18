"""FH-GA with a crisp asymmetric weighted-distance objective (task 1.9; R4.1, R2.1).

The fuzzy-vs-crisp control: operators, parameters, budgets, protocol, and
evaluation accounting are ALL identical to FHGA — only the internal search
guidance differs. Instead of the membership mu, selection and local search
are steered by the unclamped crisp score

    g(S) = 1 - d(S),   d(S) = (T_M - S)/dL  if S <= T_M
                              (S - T_M)/dR  otherwise,

i.e. one minus the asymmetric weighted distance to T_M with tolerance
weights 1/dL, 1/dR. Inside the support g equals mu exactly; outside, where
mu is flat at zero, g keeps decreasing and therefore still provides
gradient signal. This isolates R4.1's question — whether the bounded fuzzy
objective contributes anything as a SEARCH signal — from everything else.

Reported quality is unchanged: the protocol tracks and the harness rescores
mu for every evaluated subset, identically to all other solvers.
"""

from __future__ import annotations

from src.core.instance import Instance
from src.solvers.fhga import FHGA


class FHGACrisp(FHGA):
    def __init__(self, variant: str = "thorough", config=None, name=None):
        super().__init__(variant=variant, config=config,
                         name=name or f"fhga_crisp_{variant}")

    def _make_guide(self, instance: Instance):
        t = instance.target
        d_l = t.delta_L if t.delta_L > 0 else 1.0
        d_r = t.delta_R if t.delta_R > 0 else 1.0
        t_m = t.T_M

        def guide(S: float, mu: float) -> float:
            d = (t_m - S) / d_l if S <= t_m else (S - t_m) / d_r
            return 1.0 - d

        return guide
