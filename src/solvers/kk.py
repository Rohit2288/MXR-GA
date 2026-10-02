"""Karmarkar-Karp differencing and Complete Karmarkar-Karp for FSSP.

Required by the novelty check (paper_fragments/novelty_check.md §3): KK is
the standard fast heuristic for partitioning MANY LARGE numbers, the regime
in which the proposed method is claimed to be the right tool.

Reduction. Subset sum with target T over items A (total S) becomes number
partitioning of A + {d}, with d = |S - 2T|:
  * if 2T <= S: each half of a perfect partition sums to S - T, and the
    items on d's side sum to T;
  * if 2T >  S: each half sums to T, and the side WITHOUT d sums to T.
An imperfect partition gives a near-T subset, which is scored as is.

KK (Karmarkar & Karp 1982): repeatedly replace the two largest numbers by
their difference (committing them to opposite sides). CKK (Korf 1998):
depth-first search in which the second branch replaces them by their SUM
(same side); the first leaf is the KK solution, and the search continues
as an anytime complete method with the standard bound (if the largest
number is at least the sum of the rest, the best completion is fixed).
Here CKK runs until the protocol stops it.

Sides are reconstructed exactly with a parity union-find over the
merge decisions, so every leaf yields an explicit subset, scored through
the protocol on its true sum (certified, like every solver here).
Target: T_M; one counted evaluation per leaf.
"""

from __future__ import annotations

import bisect
from typing import List, Tuple

import numpy as np

from src.core.instance import Instance
from src.protocol.budget import BudgetedObjective
from src.solvers.base import Solver


def _sides(n_total: int, merges: List[Tuple[int, int, int]]) -> List[int]:
    """Side (0/1) of every original element under the merge decisions
    (a, b, parity): parity 1 = opposite sides, 0 = same side."""
    parent = list(range(n_total))
    par = [0] * n_total            # parity to parent

    def find(x):
        p = 0
        root = x
        while parent[root] != root:
            p ^= par[root]
            root = parent[root]
        # path compression with parity
        cur, cp = x, p
        while parent[cur] != cur:
            nxt, np_ = parent[cur], cp ^ par[cur]
            parent[cur], par[cur] = root, cp
            cur, cp = nxt, np_
        return root, p

    for a, b, parity in merges:
        ra, pa = find(a)
        rb, pb = find(b)
        if ra != rb:
            parent[rb] = ra
            par[rb] = pa ^ pb ^ parity
    return [find(x)[1] for x in range(n_total)]


class CKK(Solver):
    """complete=False gives plain KK (a single leaf)."""

    def __init__(self, complete: bool = True):
        self.complete = complete
        self.name = "ckk" if complete else "kk"
        self.last_stats: dict = {}

    def search(self, instance: Instance, objective: BudgetedObjective,
               rng: np.random.Generator) -> None:
        n = instance.n
        a = [int(x) for x in instance.items]
        t = int(round(instance.target.T_M))
        s = sum(a)
        d = abs(s - 2 * t)
        d_side_is_subset = 2 * t <= s
        vals = a + [d]
        n_tot = n + 1
        stats = self.last_stats = {"leaves": 0}

        def leaf(merges):
            side = _sides(n_tot, merges)
            sd = side[n]
            if d_side_is_subset:
                subset = [i for i in range(n) if side[i] == sd]
            else:
                subset = [i for i in range(n) if side[i] != sd]
            stats["leaves"] += 1
            objective.evaluate(subset)

        L = sorted((v, i) for i, v in enumerate(vals))   # ascending
        total = sum(vals)
        merges: List[Tuple[int, int, int]] = []
        best = [float("inf")]

        # Iterative DFS with an undo log. A frame records the two popped
        # entries, the branch taken (0 = difference, 1 = sum) and the
        # entry inserted for it.
        frames = []

        def apply(fa, fb, branch):
            nonlocal total
            if branch == 0:
                new = (fa[0] - fb[0], fa[1])
                merges.append((fa[1], fb[1], 1))
                total -= 2 * fb[0]
            else:
                new = (fa[0] + fb[0], fa[1])
                merges.append((fa[1], fb[1], 0))
            bisect.insort(L, new)
            return new

        def undo(fa, fb, branch, new):
            nonlocal total
            del L[bisect.bisect_left(L, new)]
            merges.pop()
            if branch == 0:
                total += 2 * fb[0]
            L.append(fb)
            L.append(fa)

        def descend():
            while True:
                big = L[-1]
                rest = total - big[0]
                if len(L) == 1 or big[0] >= rest:
                    disc = big[0] - rest
                    if disc < best[0]:
                        best[0] = disc
                    # Completion: every other entry opposite to the largest.
                    extra = [(big[1], e[1], 1) for e in L[:-1]]
                    merges.extend(extra)
                    leaf(merges)
                    del merges[len(merges) - len(extra):]
                    return
                fa = L.pop()
                fb = L.pop()
                new = apply(fa, fb, 0)
                frames.append([fa, fb, 0, new])

        descend()
        if not self.complete:
            return
        while frames:
            fa, fb, branch, new = frames.pop()
            undo(fa, fb, branch, new)
            if branch == 1:
                continue
            # Bound: with the largest entry fixed, the discrepancy cannot
            # drop below largest - (sum of the rest); prune if no better.
            L.pop()
            L.pop()
            new = apply(fa, fb, 1)
            if L[-1][0] - (total - L[-1][0]) >= best[0]:
                undo(fa, fb, 1, new)
                continue
            frames.append([fa, fb, 1, new])
            descend()
