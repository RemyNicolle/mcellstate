from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import betaln, gammaln, logsumexp

from .state import ClusterState


def validate_psi(psi: np.ndarray, n_genes: int) -> np.ndarray:
    psi_arr = np.asarray(psi, dtype=np.float64)
    if psi_arr.ndim != 1 or psi_arr.shape[0] != n_genes:
        raise ValueError("psi must be a one-dimensional vector with one value per gene")
    if np.any(psi_arr <= 0) or not np.all(np.isfinite(psi_arr)):
        raise ValueError("psi must be finite and strictly positive")
    return psi_arr


def dm_posterior_predictive_logpmf_sparse(
    x_indices: np.ndarray,
    x_data: np.ndarray,
    nx: float,
    ctrain_values: np.ndarray,
    ntrain: float,
    psi_values: np.ndarray,
    psi0: float,
) -> float:
    """Collapsed DM posterior predictive, omitting multinomial coefficients."""

    out = gammaln(ntrain + psi0) - gammaln(ntrain + nx + psi0)
    out += np.sum(gammaln(ctrain_values + x_data + psi_values) - gammaln(ctrain_values + psi_values))
    return float(out)


def self_loo_score(state: ClusterState, k: int, psi: np.ndarray) -> float:
    """Cell-level leave-one-out self-predictive score for target cluster k."""

    psi = validate_psi(psi, state.n_genes)
    psi0 = float(psi.sum())
    Ck_dense = state.C.getrow(k).toarray().ravel().astype(np.float64)
    Nk = float(state.N[k])
    total = 0.0
    for cell in state.cell_indices[k]:
        row = state.X.getrow(int(cell))
        nx = float(row.data.sum())
        if nx <= 0:
            continue
        idx = row.indices
        x = row.data.astype(np.float64, copy=False)
        ctrain_values = Ck_dense[idx] - x
        if state.n_cells[k] <= 1:
            ctrain_values = np.zeros_like(x, dtype=np.float64)
            ntrain = 0.0
        else:
            ctrain_values = np.maximum(ctrain_values, 0.0)
            ntrain = max(0.0, Nk - nx)
        total += dm_posterior_predictive_logpmf_sparse(
            idx, x, nx, ctrain_values, ntrain, psi[idx], psi0
        )
    return float(total)


@dataclass(frozen=True)
class ParentModel:
    """Parent posterior means evaluated only on requested genes."""

    state: ClusterState
    psi: np.ndarray
    psi0: float
    denom: np.ndarray

    def probs(self, parent: int, genes: np.ndarray) -> np.ndarray:
        counts = self.state.C.getrow(int(parent))[:, genes].toarray().ravel().astype(np.float64)
        return (counts + self.psi[genes]) / self.denom[int(parent)]

    def log_probs(self, parent: int, genes: np.ndarray) -> np.ndarray:
        return np.log(self.probs(parent, genes))


def parent_log_probs(state: ClusterState, psi: np.ndarray) -> ParentModel:
    """Build cached parent posterior mean model without dense K x G storage."""

    psi = validate_psi(psi, state.n_genes)
    psi0 = float(psi.sum())
    return ParentModel(state=state, psi=psi, psi0=psi0, denom=state.N + psi0)


def parent_scores_for_target(state: ClusterState, k: int, parent_model: ParentModel) -> np.ndarray:
    """L_parent(k; a) for every parent a."""

    row = state.C.getrow(k)
    if row.nnz == 0:
        return np.zeros(state.n_clusters, dtype=np.float64)
    scores = np.empty(state.n_clusters, dtype=np.float64)
    for a in range(state.n_clusters):
        scores[a] = float(row.data @ parent_model.log_probs(a, row.indices))
    return scores


@dataclass(frozen=True)
class MixtureScore:
    log_likelihood: float
    lambda_map: float
    lambda_mean: float
    lambda_grid: np.ndarray
    lambda_log_posterior: np.ndarray


def make_lambda_grid(size: int) -> np.ndarray:
    if size < 3:
        raise ValueError("lambda_grid_size must be at least 3")
    return np.linspace(0.0, 1.0, int(size), dtype=np.float64)


def mixture_score_for_pair(
    target_indices: np.ndarray,
    target_counts: np.ndarray,
    parent_model: ParentModel,
    a: int,
    b: int,
    lambda_grid: np.ndarray,
    beta_lambda: tuple[float, float] = (2.0, 2.0),
    return_posterior: bool = False,
) -> MixtureScore:
    """Log-integrated convex-mixture likelihood over a fixed lambda grid."""

    alpha, beta = map(float, beta_lambda)
    if alpha <= 0 or beta <= 0:
        raise ValueError("beta_lambda parameters must be positive")
    qa = parent_model.probs(a, target_indices)
    qb = parent_model.probs(b, target_indices)
    grid = np.asarray(lambda_grid, dtype=np.float64)
    log_terms = np.empty(grid.shape[0], dtype=np.float64)
    for i, lam in enumerate(grid):
        q = lam * qa + (1.0 - lam) * qb
        q = np.maximum(q, np.finfo(np.float64).tiny)
        ll = float(target_counts @ np.log(q))
        if lam <= 0.0 or lam >= 1.0:
            if alpha == 1.0 and beta == 1.0:
                log_prior = 0.0
            else:
                log_prior = -math.inf
        else:
            log_prior = (alpha - 1.0) * math.log(lam) + (beta - 1.0) * math.log1p(-lam) - betaln(alpha, beta)
        log_terms[i] = ll + log_prior

    if grid.shape[0] == 1:
        log_integral = float(log_terms[0])
    else:
        log_integral = float(logsumexp(log_terms) + math.log(1.0 / (grid.shape[0] - 1)))
    norm = logsumexp(log_terms)
    posterior = np.exp(log_terms - norm) if np.isfinite(norm) else np.full_like(grid, 1.0 / grid.shape[0])
    lambda_map = float(grid[int(np.argmax(log_terms))])
    lambda_mean = float(np.sum(grid * posterior))
    if not return_posterior:
        posterior = np.array([], dtype=np.float64)
    return MixtureScore(log_integral, lambda_map, lambda_mean, grid, posterior)


def cell_lambda_maps(
    state: ClusterState,
    k: int,
    a: int,
    b: int,
    parent_model: ParentModel,
    lambda_grid: np.ndarray,
) -> np.ndarray:
    """Diagnostic per-cell MAP lambda under the best parent pair."""

    out = np.empty(state.n_cells[k], dtype=np.float64)
    for j, cell in enumerate(state.cell_indices[k]):
        row = state.X.getrow(int(cell))
        if row.nnz == 0:
            out[j] = np.nan
            continue
        qa = parent_model.probs(a, row.indices)
        qb = parent_model.probs(b, row.indices)
        scores = []
        for lam in lambda_grid:
            q = np.maximum(lam * qa + (1.0 - lam) * qb, np.finfo(np.float64).tiny)
            scores.append(float(row.data @ np.log(q)))
        out[j] = float(lambda_grid[int(np.argmax(scores))])
    return out
