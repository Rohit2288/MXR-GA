"""THE single scorer (task 0.3).

Every reported quality number in this project comes from `evaluate()`:
it validates the subset indices, sums the actual (unscaled) items, and
recomputes membership from the instance target. No solver self-reports
quality, and nothing else in the codebase computes it. (R2.5, R3.1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from src.core.instance import Instance


class InvalidSubsetError(ValueError):
    """Raised when a solver returns an index set that is not a valid subset."""


@dataclass(frozen=True)
class Evaluation:
    subset_indices: Tuple[int, ...]   # canonical: sorted, deduplication rejected
    subset_sum: int
    mu: float


def evaluate(instance: Instance, subset_indices: Iterable[int]) -> Evaluation:
    """Score a subset of item indices against an instance.

    Raises InvalidSubsetError on out-of-range, duplicate, or non-integer
    indices. The empty subset is valid (sum 0).

    The validation is expressed with set/min/max rather than a per-element
    Python loop purely for speed -- this function is on the hot path of every
    solver run. The accepted/rejected sets and the returned values are
    identical to the original element-wise formulation; tests/test_evaluate.py
    pins the rejection cases.
    """
    idx = tuple(subset_indices)
    if not idx:
        return Evaluation(subset_indices=(), subset_sum=0,
                          mu=instance.target.mu(0))
    n = instance.n
    # bool is a subclass of int and must be excluded, so the check is on the
    # exact type. Collecting the distinct types with set(map(type, ...)) does
    # this entirely in C; the per-element Python loop runs only to build the
    # error message on the (never-taken-in-practice) failure path.
    if set(map(type, idx)) != {int}:
        for i in idx:
            if type(i) is not int:
                raise InvalidSubsetError(f"non-integer index {i!r}")
    if len(set(idx)) != len(idx):
        seen = set()
        for i in idx:
            if i in seen:
                raise InvalidSubsetError(f"duplicate index {i}")
            seen.add(i)
    lo, hi = min(idx), max(idx)
    if lo < 0:
        raise InvalidSubsetError(f"index {lo} out of range for n={n}")
    if hi >= n:
        raise InvalidSubsetError(f"index {hi} out of range for n={n}")
    items = instance.items
    s = sum(map(items.__getitem__, idx))
    return Evaluation(
        subset_indices=tuple(sorted(idx)),
        subset_sum=s,
        mu=instance.target.mu(s),
    )
