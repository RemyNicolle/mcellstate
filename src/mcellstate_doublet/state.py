from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ClusterState:
    """Sparse cluster-level sufficient statistics for an existing partition."""

    X: sparse.csr_matrix
    z_codes: np.ndarray
    cluster_ids: np.ndarray
    C: sparse.csr_matrix
    N: np.ndarray
    n_cells: np.ndarray
    cell_indices: tuple[np.ndarray, ...]

    @property
    def n_clusters(self) -> int:
        return int(self.C.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.C.shape[1])

    @property
    def total_cells(self) -> int:
        return int(self.X.shape[0])


def as_csr_counts(X: Any) -> sparse.csr_matrix:
    """Return X as CSR float64 without densifying sparse inputs."""

    if sparse.issparse(X):
        out = X.tocsr().astype(np.float64, copy=False)
    else:
        out = sparse.csr_matrix(np.asarray(X, dtype=np.float64))
    out.sum_duplicates()
    out.eliminate_zeros()
    if np.any(out.data < 0):
        raise ValueError("X must contain non-negative UMI counts")
    return out


def build_cluster_state(X: Any, z: Any) -> ClusterState:
    """Build sparse cluster x gene counts from cells x genes counts and labels."""

    X_csr = as_csr_counts(X)
    z_arr = np.asarray(z)
    if z_arr.ndim != 1 or z_arr.shape[0] != X_csr.shape[0]:
        raise ValueError("z must be a one-dimensional label vector with one entry per cell")

    cluster_ids, z_codes = np.unique(z_arr, return_inverse=True)
    n_clusters = int(cluster_ids.shape[0])
    rows = z_codes.astype(np.int64, copy=False)
    cols = np.arange(X_csr.shape[0], dtype=np.int64)
    data = np.ones(X_csr.shape[0], dtype=np.float64)
    membership = sparse.csr_matrix((data, (rows, cols)), shape=(n_clusters, X_csr.shape[0]))
    C = (membership @ X_csr).tocsr()
    C.sum_duplicates()
    C.eliminate_zeros()
    N = np.asarray(C.sum(axis=1)).ravel().astype(np.float64)
    n_cells = np.bincount(z_codes, minlength=n_clusters).astype(np.int64)
    cell_indices = tuple(np.flatnonzero(z_codes == k) for k in range(n_clusters))
    return ClusterState(X_csr, z_codes.astype(np.int64), cluster_ids, C, N, n_cells, cell_indices)


def make_prior(X: Any, tau: float = 1.0, mode: str = "global_frequency") -> np.ndarray:
    """Create positive Dirichlet pseudocounts for genes.

    Parameters
    ----------
    X:
        Raw cells x genes UMI count matrix.
    tau:
        Total prior mass. ``tau=1`` is intentionally weak.
    mode:
        ``"global_frequency"`` uses empirical gene frequencies, with a tiny
        floor. ``"uniform"`` spreads mass evenly over genes.
    """

    X_csr = as_csr_counts(X)
    if tau <= 0:
        raise ValueError("tau must be positive")
    n_genes = X_csr.shape[1]
    if mode == "uniform":
        return np.full(n_genes, tau / max(1, n_genes), dtype=np.float64)
    if mode != "global_frequency":
        raise ValueError("mode must be 'global_frequency' or 'uniform'")
    gene_counts = np.asarray(X_csr.sum(axis=0)).ravel().astype(np.float64)
    smoothed = gene_counts + 1e-12
    return tau * smoothed / smoothed.sum()
