from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np

from .abundance import pair_collision_fraction


def top_parent_candidates(
    parent_scores: np.ndarray,
    target: int,
    top_m: int = 50,
    provided: Mapping[int, Sequence[int]] | Sequence[int] | None = None,
) -> np.ndarray:
    """Return candidate parents by single-parent count likelihood."""

    if provided is not None:
        vals = provided.get(target, []) if isinstance(provided, Mapping) else provided
        arr = np.asarray(list(vals), dtype=np.int64)
        return arr[arr != target]
    scores = np.asarray(parent_scores, dtype=np.float64).copy()
    scores[int(target)] = -np.inf
    m = min(int(top_m), max(0, scores.shape[0] - 1))
    if m <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(-scores, np.arange(m))[:m]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64, copy=False)


def generate_parent_pairs(candidates: Sequence[int], include_homotypic: bool = False) -> list[tuple[int, int]]:
    cand = sorted({int(x) for x in candidates})
    pairs = list(itertools.combinations(cand, 2))
    if include_homotypic:
        pairs.extend((x, x) for x in cand)
    return pairs


def pair_prior_logprob(
    a: int,
    b: int,
    n_cells: np.ndarray,
    total_cells: int,
    pairs: Sequence[tuple[int, int]],
    mode: str = "collision_normalized",
) -> float:
    """Prior over tested pairs; use mode='none' to disable."""

    if mode in ("none", None):
        return 0.0
    if mode != "collision_normalized":
        raise ValueError("pair_prior_mode must be 'collision_normalized' or 'none'")
    total = max(1.0, float(total_cells))
    weights = []
    selected = None
    for pair in pairs:
        fa = float(n_cells[pair[0]]) / total
        fb = float(n_cells[pair[1]]) / total
        w = max(pair_collision_fraction(fa, fb, pair[0] == pair[1]), np.finfo(np.float64).tiny)
        weights.append(w)
        if pair == (a, b):
            selected = w
    if selected is None:
        return -math.inf
    return float(math.log(selected) - math.log(float(np.sum(weights))))
