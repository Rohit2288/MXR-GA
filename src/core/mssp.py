"""Fuzzy Multiple-Subset-Sum (MSSP), k=2 — third domain (task 2.17).

Third clipped-membership domain, chosen to vary the combinatorial core once
more (task 2.16 / FKP added a capacity constraint; this couples the
objective across TWO targets). Each item is assigned to bin 1, bin 2, or
neither; bin b must hit its own triangular target. The satisfaction degree
is the *joint* membership

    mu(x) = min( mu_{T1}(sum1), mu_{T2}(sum2) ),

which is zero whenever EITHER bin sum leaves its support. A single
reassignment of one item changes one or both bin sums by at most one item
value, so Proposition 2 (no-escape) of the plateau theory applies verbatim
with the "largest single-move change" being max_i a_i in either coordinate
--- this is the domain that most directly exercises the theory's
domain-agnostic statement.

Exact optimum: 2-D reachability DP over (sum1, sum2) with parent pointers;
verified against brute force on small n by tests/test_mssp.py. Membership is
recomputed from the reconstructed assignment by the shared scorer logic, so
the certification property holds here too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from src.core.fuzzy import TriangularFuzzyNumber


@dataclass(frozen=True)
class MSSPInstance:
    instance_id: str
    items: Tuple[int, ...]
    target1: TriangularFuzzyNumber
    target2: TriangularFuzzyNumber
    n: int
    metadata: dict = field(default_factory=dict)
    witness: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MSSPInstance":
        return cls(
            instance_id=d["instance_id"],
            items=tuple(d["items"]),
            target1=TriangularFuzzyNumber.from_dict(d["target1"]),
            target2=TriangularFuzzyNumber.from_dict(d["target2"]),
            n=d["metadata"]["n"],
            metadata=d.get("metadata", {}),
            witness=d.get("witness", {}),
        )

    def to_dict(self) -> dict:
        return {
            "schema": "mssp-1.0",
            "instance_id": self.instance_id,
            "items": list(self.items),
            "target1": self.target1.to_dict(),
            "target2": self.target2.to_dict(),
            "metadata": self.metadata,
            "witness": self.witness,
        }


def membership(inst: MSSPInstance, assignment) -> float:
    """Joint membership of an assignment (x_i in {0,1,2}); 0 = unused."""
    s1 = sum(a for a, x in zip(inst.items, assignment) if x == 1)
    s2 = sum(a for a, x in zip(inst.items, assignment) if x == 2)
    return min(inst.target1.mu(s1), inst.target2.mu(s2))


def load_mssp(path: Path | str) -> MSSPInstance:
    with open(path) as fh:
        return MSSPInstance.from_dict(json.load(fh))


def exact_optimum(inst: MSSPInstance) -> Tuple[float, List[int]]:
    """Return (best mu, assignment) by exact 2-D reachability DP.

    Tracks, for each reachable (s1, s2) with s1 <= floor(T1_R) and
    s2 <= floor(T2_R), one assignment reaching it (parent pointers). Both
    memberships are increasing then decreasing, so the joint optimum is
    found by scanning the reachable frontier once. Feasible while
    (T1_R+1)*(T2_R+1) stays modest -- the generator keeps targets small.
    """
    import math

    t1, t2 = inst.target1, inst.target2
    h1, h2 = math.floor(t1.T_R), math.floor(t2.T_R)
    # reach[(s1,s2)] = (prev_state, item_index, bin) ; seed with (0,0).
    reach = {(0, 0): None}
    for i, a in enumerate(inst.items):
        nxt = {}
        for (s1, s2), _ in reach.items():
            # bin 1
            if s1 + a <= h1 and (s1 + a, s2) not in reach and (s1 + a, s2) not in nxt:
                nxt[(s1 + a, s2)] = ((s1, s2), i, 1)
            # bin 2
            if s2 + a <= h2 and (s1, s2 + a) not in reach and (s1, s2 + a) not in nxt:
                nxt[(s1, s2 + a)] = ((s1, s2), i, 2)
        reach.update(nxt)

    best_mu, best_state = -1.0, (0, 0)
    for (s1, s2) in reach:
        m = min(t1.mu(s1), t2.mu(s2))
        if m > best_mu:
            best_mu, best_state = m, (s1, s2)

    assignment = [0] * inst.n
    state = best_state
    while reach.get(state) is not None:
        prev, i, b = reach[state]
        assignment[i] = b
        state = prev
    assert min(inst.target1.mu(sum(a for a, x in zip(inst.items, assignment) if x == 1)),
               inst.target2.mu(sum(a for a, x in zip(inst.items, assignment) if x == 2))) \
        >= best_mu - 1e-12
    return best_mu, assignment
