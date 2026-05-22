import numpy as np
from scipy import sparse

from mcellstate.likelihood import (
    delta_merge,
    delta_move_block,
    delta_move_cell,
    delta_peel_block,
    delta_peel_cell,
    full_partition_log_likelihood,
)
from mcellstate.state import PartitionState, SparseCountVector
from mcellstate.validation import generate_synthetic_dataset


def make_test_state() -> tuple[PartitionState, np.ndarray]:
    X = sparse.csr_matrix(
        np.asarray(
            [
                [4, 0, 0, 0],
                [3, 1, 0, 0],
                [0, 0, 4, 0],
                [0, 0, 3, 1],
            ],
            dtype=np.int64,
        ),
    )
    z = np.asarray([0, 0, 1, 1], dtype=np.int64)
    psi = np.full(X.shape[1], 0.5, dtype=np.float64)
    return PartitionState.from_assignment(X, z), psi


def test_merge_delta_matches_full_recompute():
    state, psi = make_test_state()
    before = full_partition_log_likelihood(state, psi)
    delta = delta_merge(state, psi, 0, 1)

    mutated = state.copy()
    mutated.merge_clusters(0, 1)
    after = full_partition_log_likelihood(mutated, psi)

    assert np.isclose(after - before, delta)


def test_peel_delta_matches_full_recompute():
    state, psi = make_test_state()
    before = full_partition_log_likelihood(state, psi)
    delta = delta_peel_cell(state, psi, cell=1, source_cluster=0)

    mutated = state.copy()
    mutated.peel_cell_to_new_cluster(1, 0)
    after = full_partition_log_likelihood(mutated, psi)

    assert np.isclose(after - before, delta)


def test_move_delta_matches_full_recompute():
    state, psi = make_test_state()
    before = full_partition_log_likelihood(state, psi)
    delta = delta_move_cell(state, psi, cell=1, target_cluster=1, source_cluster=0)

    mutated = state.copy()
    mutated.move_cell(1, 0, 1)
    after = full_partition_log_likelihood(mutated, psi)

    assert np.isclose(after - before, delta)


def test_block_deltas_match_cell_special_cases():
    state, psi = make_test_state()
    cell = 1
    genes, counts = state.cell_counts(cell)

    peel_cell_delta = delta_peel_cell(state, psi, cell=cell, source_cluster=0)
    peel_block_delta = delta_peel_block(state.clusters[0], genes, counts, psi)
    assert np.isclose(peel_cell_delta, peel_block_delta)

    move_cell_delta = delta_move_cell(
        state, psi, cell=cell, target_cluster=1, source_cluster=0
    )
    move_block_delta = delta_move_block(
        state.clusters[0], state.clusters[1], genes, counts, psi
    )
    assert np.isclose(move_cell_delta, move_block_delta)


def test_state_updates_preserve_consistency_and_drop_empty_clusters():
    state, _ = make_test_state()
    state.validate()

    new_cluster = state.peel_cell_to_new_cluster(1, 0)
    state.validate()
    assert new_cluster in state.active_cluster_ids

    state.move_cell(1, new_cluster, 1)
    state.validate()
    assert new_cluster not in state.active_cluster_ids

    state.merge_clusters(0, 1)
    state.validate()
    assert len(state.active_cluster_ids) == 1


def test_block_state_updates_preserve_consistency():
    state, _ = make_test_state()
    state.validate()

    new_cluster = state.peel_block_to_new_cluster([0], 0)
    state.validate()
    assert new_cluster in state.active_cluster_ids

    state.move_block([0], new_cluster, 1)
    state.validate()
    assert new_cluster not in state.active_cluster_ids


def test_sparse_count_vector_round_trip():
    vector = SparseCountVector.from_indices(
        np.asarray([3, 1, 5], dtype=np.int64),
        np.asarray([2, 4, 1], dtype=np.int64),
    )
    genes, counts = vector.sorted_items()
    assert np.array_equal(genes, np.asarray([1, 3, 5], dtype=np.int64))
    assert np.array_equal(counts, np.asarray([4, 2, 1], dtype=np.int64))


def test_state_likelihood_cache_tracks_mutations():
    state, psi = make_test_state()
    cached_before = state.total_log_likelihood_cached(psi)
    direct_before = full_partition_log_likelihood(state, psi)
    assert np.isclose(cached_before, direct_before)

    state.merge_clusters(0, 1)
    cached_after = state.total_log_likelihood_cached(psi)
    direct_after = full_partition_log_likelihood(state, psi)
    assert np.isclose(cached_after, direct_after)


def test_additional_initialization_modes_produce_valid_partitions():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=5,
        n_genes=20,
        marker_strength=24.0,
        seed=75,
    )

    random_online = PartitionState.from_csr(
        synthetic.X,
        init="random_online",
        seed=75,
        n_clusters=6,
    )
    leiden_less = PartitionState.from_csr(
        synthetic.X,
        init="leiden_less_overclustered",
        seed=75,
        n_clusters=6,
    )

    random_online.validate()
    leiden_less.validate()
    assert 1 < len(random_online.active_cluster_ids) <= synthetic.X.shape[0]
    assert 1 < len(leiden_less.active_cluster_ids) <= synthetic.X.shape[0]
