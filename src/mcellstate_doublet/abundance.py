from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


def nb_logpmf(x: int | float, mean: float, phi: float) -> float:
    """Negative binomial log PMF with var = mean + mean^2 / phi."""

    x = float(x)
    mean = max(float(mean), np.finfo(np.float64).tiny)
    phi = max(float(phi), np.finfo(np.float64).tiny)
    p = phi / (phi + mean)
    return float(gammaln(x + phi) - gammaln(phi) - gammaln(x + 1.0) + phi * math.log(p) + x * math.log1p(-p))


@dataclass(frozen=True)
class SimpletSizePrior:
    mean: float
    phi: float
    min_mean: float = 1e-6

    def logpmf(self, x: int | float) -> float:
        return nb_logpmf(x, max(self.mean, self.min_mean), self.phi)


def fit_simplet_size_prior(
    n_cells: np.ndarray,
    exclude_index: int | None = None,
    exclude_singletons: bool = True,
    min_phi: float = 0.5,
    max_phi: float = 100.0,
) -> SimpletSizePrior:
    """Fit a broad empirical NB prior to cluster sizes."""

    sizes = np.asarray(n_cells, dtype=np.float64)
    mask = np.ones(sizes.shape[0], dtype=bool)
    if exclude_index is not None:
        mask[int(exclude_index)] = False
    if exclude_singletons:
        mask &= sizes > 1
    train = sizes[mask]
    if train.size == 0:
        train = sizes[sizes > 0]
    mean = float(np.mean(train)) if train.size else 1.0
    var = float(np.var(train, ddof=1)) if train.size > 1 else mean + mean * mean / min_phi
    if var <= mean:
        phi = max_phi
    else:
        phi = mean * mean / (var - mean)
    phi = float(np.clip(phi, min_phi, max_phi))
    return SimpletSizePrior(mean=max(mean, 1e-6), phi=phi)


def pair_collision_fraction(f_a: float, f_b: float, homotypic: bool = False) -> float:
    return float(f_a * f_a if homotypic else 2.0 * f_a * f_b)


def expected_doublet_count(
    total_cells: int,
    n_cells: np.ndarray,
    a: int,
    b: int,
    doublet_rate: float,
) -> float:
    total = max(1.0, float(total_cells))
    f_a = float(n_cells[a]) / total
    f_b = float(n_cells[b]) / total
    return total * float(doublet_rate) * pair_collision_fraction(f_a, f_b, homotypic=(a == b))


def abundance_logbf(
    observed: int,
    total_cells: int,
    n_cells: np.ndarray,
    a: int,
    b: int,
    doublet_rate: float,
    phi_doublet: float,
    simplet_prior: SimpletSizePrior,
) -> float:
    mu = expected_doublet_count(total_cells, n_cells, a, b, doublet_rate)
    return nb_logpmf(observed, mu, phi_doublet) - simplet_prior.logpmf(observed)
