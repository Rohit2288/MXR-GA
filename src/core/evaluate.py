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
    """
    idx = tuple(subset_indices)
    n = instance.n
    seen = set()
    for i in idx:
        if isinstance(i, bool) or not isinstance(i, int):
            raise InvalidSubsetError(f"non-integer index {i!r}")
        if not 0 <= i < n:
            raise InvalidSubsetError(f"index {i} out of range for n={n}")
        if i in seen:
            raise InvalidSubsetError(f"duplicate index {i}")
        seen.add(i)
    s = sum(instance.items[i] for i in idx)
    return Evaluation(
        subset_indices=tuple(sorted(idx)),
        subset_sum=s,
        mu=instance.target.mu(s),
    )
