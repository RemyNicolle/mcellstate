import numpy as np
from scipy import sparse
from scipy.special import gammaln

from mcellstate.likelihood import (
    cluster_log_likelihood_from_sparse,
    dense_cluster_log_likelihood,
    full_partition_log_likelihood,
    full_partition_log_likelihood_from_assignment,
)
from mcellstate.state import PartitionState


def test_manual_cluster_log_likelihood_matches_formula():
    psi = np.asarray([0.5, 1.5, 2.0], dtype=np.float64)
    counts = np.asarray([2, 0, 3], dtype=np.int64)
    indices = np.flatnonzero(counts)
    values = counts[indices]

    expected = (
        gammaln(psi.sum())
        - gammaln(counts.sum() + psi.sum())
        + (gammaln(counts[0] + psi[0]) - gammaln(psi[0]))
        + (gammaln(counts[2] + psi[2]) - gammaln(psi[2]))
    )

    assert np.isclose(cluster_log_likelihood_from_sparse(indices, values, psi), expected)
    assert np.isclose(dense_cluster_log_likelihood(counts, psi), expected)


def test_full_partition_likelihood_matches_assignment_recompute():
    X = sparse.csr_matrix(
        np.asarray(
            [
                [3, 0, 0, 1],
                [2, 1, 0, 0],
                [0, 0, 4, 0],
                [0, 1, 3, 0],
            ],
            dtype=np.int64,
        ),
    )
    z = np.asarray([0, 0, 1, 1], dtype=np.int64)
    psi = np.full(X.shape[1], 0.7, dtype=np.float64)

    state = PartitionState.from_assignment(X, z)
    ll_state = full_partition_log_likelihood(state, psi)
    ll_assignment = full_partition_log_likelihood_from_assignment(X, z, psi)

    assert np.isclose(ll_state, ll_assignment)
