from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import gammaln

from .state import PartitionState, SparseCountVector


def cluster_log_likelihood_from_sparse(
    indices: np.ndarray,
    values: np.ndarray,
    psi: np.ndarray,
    *,
    psi0: float | None = None,
) -> float:
    psi = np.asarray(psi, dtype=np.float64)
    values = np.asarray(values, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    psi0 = float(psi.sum()) if psi0 is None else float(psi0)
    total = int(values.sum())
    ll = gammaln(psi0) - gammaln(total + psi0)
    if indices.size:
        ll += np.sum(gammaln(values.astype(np.float64) + psi[indices]) - gammaln(psi[indices]))
    return float(ll)


def cluster_log_likelihood(
    cluster: SparseCountVector,
    psi: np.ndarray,
    *,
    psi0: float | None = None,
) -> float:
    indices, values = cluster.sorted_items()
    return cluster_log_likelihood_from_sparse(indices, values, psi, psi0=psi0)


def dense_cluster_log_likelihood(counts: np.ndarray, psi: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    psi = np.asarray(psi, dtype=np.float64)
    total = int(counts.sum())
    return float(
        gammaln(psi.sum())
        - gammaln(total + psi.sum())
        + np.sum(gammaln(counts.astype(np.float64) + psi) - gammaln(psi)),
    )


def full_partition_log_likelihood(state: PartitionState, psi: np.ndarray) -> float:
    return float(state.total_log_likelihood_cached(np.asarray(psi, dtype=np.float64)))


def full_partition_log_likelihood_from_assignment(
    X: sparse.csr_matrix,
    z: np.ndarray,
    psi: np.ndarray,
) -> float:
    state = PartitionState.from_assignment(X, z)
    return full_partition_log_likelihood(state, psi)


def singleton_log_likelihood(
    state: PartitionState,
    cell: int,
    psi: np.ndarray,
    *,
    psi0: float | None = None,
) -> float:
    indices, values = state.cell_counts(int(cell))
    return cluster_log_likelihood_from_sparse(indices, values, psi, psi0=psi0)


def delta_add_sparse(
    cluster: SparseCountVector,
    add_indices: np.ndarray,
    add_values: np.ndarray,
    psi: np.ndarray,
    *,
    psi0: float | None = None,
) -> float:
    psi = np.asarray(psi, dtype=np.float64)
    psi0 = float(psi.sum()) if psi0 is None else float(psi0)
    add_indices = np.asarray(add_indices, dtype=np.int64)
    add_values = np.asarray(add_values, dtype=np.int64)
    n_add = int(add_values.sum())
    delta = gammaln(cluster.total + psi0) - gammaln(cluster.total + n_add + psi0)
    if add_indices.size:
        current = cluster.get_many(add_indices).astype(np.float64)
        delta += np.sum(
            gammaln(current + add_values.astype(np.float64) + psi[add_indices])
            - gammaln(current + psi[add_indices]),
        )
    return float(delta)


def delta_remove_sparse(
    cluster: SparseCountVector,
    remove_indices: np.ndarray,
    remove_values: np.ndarray,
    psi: np.ndarray,
    *,
    psi0: float | None = None,
) -> float:
    psi = np.asarray(psi, dtype=np.float64)
    psi0 = float(psi.sum()) if psi0 is None else float(psi0)
    remove_indices = np.asarray(remove_indices, dtype=np.int64)
    remove_values = np.asarray(remove_values, dtype=np.int64)
    n_remove = int(remove_values.sum())
    delta = gammaln(cluster.total + psi0) - gammaln(cluster.total - n_remove + psi0)
    if remove_indices.size:
        current = cluster.get_many(remove_indices)
        if np.any(current < remove_values):
            raise ValueError("cannot remove more counts than a cluster contains")
        current = current.astype(np.float64)
        delta += np.sum(
            gammaln(current - remove_values.astype(np.float64) + psi[remove_indices])
            - gammaln(current + psi[remove_indices]),
        )
    return float(delta)


def delta_merge(
    state: PartitionState,
    psi: np.ndarray,
    cluster_a: int,
    cluster_b: int,
) -> float:
    cluster_a = int(cluster_a)
    cluster_b = int(cluster_b)
    if cluster_a == cluster_b:
        raise ValueError("cannot merge a cluster with itself")
    vector_a = state.clusters[cluster_a]
    vector_b = state.clusters[cluster_b]
    psi0 = float(np.asarray(psi, dtype=np.float64).sum())
    if vector_a.nnz >= vector_b.nnz:
        large, small = vector_a, vector_b
    else:
        large, small = vector_b, vector_a
    indices, values = small.sorted_items()
    return float(
        delta_add_sparse(large, indices, values, psi, psi0=psi0)
        - cluster_log_likelihood(small, psi, psi0=psi0),
    )


def delta_peel_cell(
    state: PartitionState,
    psi: np.ndarray,
    cell: int,
    source_cluster: int | None = None,
) -> float:
    cell = int(cell)
    source_cluster = int(state.z[cell] if source_cluster is None else source_cluster)
    indices, values = state.cell_counts(cell)
    psi0 = float(np.asarray(psi, dtype=np.float64).sum())
    return float(
        delta_remove_sparse(state.clusters[source_cluster], indices, values, psi, psi0=psi0)
        + cluster_log_likelihood_from_sparse(indices, values, psi, psi0=psi0),
    )


def delta_move_cell(
    state: PartitionState,
    psi: np.ndarray,
    cell: int,
    target_cluster: int,
    *,
    source_cluster: int | None = None,
) -> float:
    cell = int(cell)
    target_cluster = int(target_cluster)
    source_cluster = int(state.z[cell] if source_cluster is None else source_cluster)
    if source_cluster == target_cluster:
        raise ValueError("source and target clusters must differ")
    indices, values = state.cell_counts(cell)
    psi0 = float(np.asarray(psi, dtype=np.float64).sum())
    return float(
        delta_remove_sparse(state.clusters[source_cluster], indices, values, psi, psi0=psi0)
        + delta_add_sparse(state.clusters[target_cluster], indices, values, psi, psi0=psi0),
    )


def delta_peel_block(
    source_cluster: SparseCountVector,
    block_indices: np.ndarray,
    block_values: np.ndarray,
    psi: np.ndarray,
) -> float:
    psi0 = float(np.asarray(psi, dtype=np.float64).sum())
    return float(
        delta_remove_sparse(source_cluster, block_indices, block_values, psi, psi0=psi0)
        + cluster_log_likelihood_from_sparse(block_indices, block_values, psi, psi0=psi0),
    )


def delta_move_block(
    source_cluster: SparseCountVector,
    target_cluster: SparseCountVector,
    block_indices: np.ndarray,
    block_values: np.ndarray,
    psi: np.ndarray,
) -> float:
    psi0 = float(np.asarray(psi, dtype=np.float64).sum())
    return float(
        delta_remove_sparse(source_cluster, block_indices, block_values, psi, psi0=psi0)
        + delta_add_sparse(target_cluster, block_indices, block_values, psi, psi0=psi0),
    )
