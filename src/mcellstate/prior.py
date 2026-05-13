from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar

from .likelihood import full_partition_log_likelihood
from .state import PartitionState


def global_gene_frequency(X: sparse.csr_matrix | np.ndarray, *, min_mass: float = 1e-12) -> np.ndarray:
    if sparse.issparse(X):
        gene_sums = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    else:
        gene_sums = np.asarray(X, dtype=np.float64).sum(axis=0)
    total = float(gene_sums.sum())
    if total <= 0.0:
        gene_sums = np.ones_like(gene_sums, dtype=np.float64)
        total = float(gene_sums.sum())
    p = gene_sums / total
    floor = float(min_mass) / max(len(p), 1)
    p = np.maximum(p, floor)
    p /= p.sum()
    return p


def make_prior(
    X: sparse.csr_matrix | np.ndarray,
    tau: float = 1.0,
    mode: str = "global_frequency",
    *,
    min_mass: float = 1e-12,
) -> np.ndarray:
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if mode != "global_frequency":
        raise ValueError(f"unknown prior mode: {mode}")
    return float(tau) * global_gene_frequency(X, min_mass=min_mass)


def optimize_tau(
    state: PartitionState,
    psi: np.ndarray,
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, np.ndarray, float]:
    psi = np.asarray(psi, dtype=np.float64)
    tau0 = float(psi.sum())
    if tau0 <= 0.0:
        raise ValueError("psi must have positive total concentration")

    base = psi / tau0
    lower, upper = bounds if bounds is not None else (tau0 / 1_000.0, tau0 * 1_000.0)
    lower = max(float(lower), 1e-12)
    upper = max(float(upper), lower * 10.0)

    def objective(log_tau: float) -> float:
        tau = float(np.exp(log_tau))
        return -full_partition_log_likelihood(state, tau * base)

    result = minimize_scalar(
        objective,
        bounds=(np.log(lower), np.log(upper)),
        method="bounded",
    )
    tau = float(np.exp(result.x))
    psi_opt = tau * base
    ll = full_partition_log_likelihood(state, psi_opt)
    return tau, psi_opt, ll
