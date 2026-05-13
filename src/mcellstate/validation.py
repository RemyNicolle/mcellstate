from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .likelihood import full_partition_log_likelihood
from .state import PartitionState


@dataclass
class SyntheticDataset:
    X: sparse.csr_matrix
    z_true: np.ndarray
    cluster_probs: np.ndarray


def generate_synthetic_dataset(
    *,
    n_clusters: int = 3,
    cells_per_cluster: int = 12,
    n_genes: int = 30,
    umi_low: int = 40,
    umi_high: int = 80,
    marker_strength: float = 25.0,
    background_strength: float = 1.0,
    seed: int | None = None,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    cells_per_cluster = int(cells_per_cluster)
    n_cells = int(n_clusters) * cells_per_cluster
    z_true = np.repeat(np.arange(n_clusters, dtype=np.int64), cells_per_cluster)

    cluster_probs = np.empty((n_clusters, n_genes), dtype=np.float64)
    marker_span = max(1, n_genes // n_clusters)
    for cluster_id in range(n_clusters):
        alpha = np.full(n_genes, background_strength, dtype=np.float64)
        start = (cluster_id * marker_span) % n_genes
        stop = min(start + marker_span, n_genes)
        alpha[start:stop] += marker_strength
        cluster_probs[cluster_id] = rng.dirichlet(alpha)

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for cell in range(n_cells):
        cluster_id = int(z_true[cell])
        total = int(rng.integers(umi_low, umi_high + 1))
        counts = rng.multinomial(total, cluster_probs[cluster_id])
        genes = np.flatnonzero(counts)
        rows.extend([cell] * len(genes))
        cols.extend(genes.tolist())
        data.extend(counts[genes].astype(int).tolist())

    X = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes), dtype=np.int64)
    X.sum_duplicates()
    X.sort_indices()
    return SyntheticDataset(X=X, z_true=z_true, cluster_probs=cluster_probs)


def rand_index(z_a: np.ndarray, z_b: np.ndarray) -> float:
    z_a = np.asarray(z_a, dtype=np.int64)
    z_b = np.asarray(z_b, dtype=np.int64)
    if z_a.shape != z_b.shape:
        raise ValueError("z_a and z_b must have the same shape")
    n = len(z_a)
    if n < 2:
        return 1.0
    agreements = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_a = z_a[i] == z_a[j]
            same_b = z_b[i] == z_b[j]
            agreements += int(same_a == same_b)
            total += 1
    return agreements / total


def assert_state_consistent(state: PartitionState) -> None:
    state.validate()


def assert_likelihood_monotone(initial_ll: float, final_ll: float) -> None:
    if final_ll + 1e-10 < initial_ll:
        raise AssertionError(f"likelihood decreased from {initial_ll} to {final_ll}")


def full_likelihood(state: PartitionState, psi: np.ndarray) -> float:
    return full_partition_log_likelihood(state, psi)
