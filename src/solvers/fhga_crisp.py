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

The guide itself lives in `src/protocol/guide.py` and is shared verbatim
with the guided baseline variants, so that "identical search signal" is a
property of the code rather than a claim in the text.
"""

from __future__ import annotations

from src.core.instance import Instance
from src.protocol.guide import crisp_guide
from src.solvers.fhga import FHGA


class FHGACrisp(FHGA):
    def __init__(self, variant: str = "thorough", config=None, name=None):
        super().__init__(variant=variant, config=config,
                         name=name or f"fhga_crisp_{variant}")

    def _make_guide(self, instance: Instance):
        return crisp_guide(instance)
