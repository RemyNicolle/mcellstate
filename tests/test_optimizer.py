import numpy as np
from scipy import sparse

from mcellstate.optimizer import Optimizer
from mcellstate.prior import make_prior
from mcellstate.state import PartitionState
from mcellstate.validation import generate_synthetic_dataset, rand_index
from mcellstate.warm_start import overclustered_leiden_labels


def test_optimizer_improves_likelihood_on_synthetic_data():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=8,
        n_genes=24,
        marker_strength=40.0,
        seed=7,
    )
    state = PartitionState.from_csr(synthetic.X, init="singletons")
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        n_proposals=4000,
        seed=7,
    )
    initial_ll = optimizer.fit(max_rounds=0).log_likelihood
    result = optimizer.fit(max_rounds=40, restarts=1)

    assert result.log_likelihood > initial_ll
    assert rand_index(result.z, synthetic.z_true) > 0.75


def test_optimizer_can_split_bad_one_cluster_initialization():
    X = sparse.csr_matrix(
        np.asarray(
            [
                [18, 2, 0, 0],
                [19, 1, 0, 0],
                [17, 3, 0, 0],
                [0, 0, 18, 2],
                [0, 0, 19, 1],
                [0, 0, 17, 3],
            ],
            dtype=np.int64,
        ),
    )
    z_true = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    state = PartitionState.from_csr(X, init="one_cluster")
    psi = make_prior(X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        n_proposals=2000,
        seed=5,
    )
    result = optimizer.fit(max_rounds=30, restarts=1)

    assert len(result.state.active_cluster_ids) >= 2
    assert rand_index(result.z, z_true) > 0.8


def test_leiden_overclustered_warm_start_is_nontrivial():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=10,
        n_genes=30,
        marker_strength=35.0,
        seed=23,
    )
    state = PartitionState.from_csr(synthetic.X, init="leiden_overclustered", seed=23)
    n_clusters = len(np.unique(state.z))
    assert 1 < n_clusters < synthetic.X.shape[0]


def test_local_leiden_warm_start_responds_to_target_clusters():
    synthetic = generate_synthetic_dataset(
        n_clusters=4,
        cells_per_cluster=8,
        n_genes=28,
        marker_strength=30.0,
        seed=29,
    )
    coarse = overclustered_leiden_labels(
        synthetic.X,
        seed=29,
        target_clusters=4,
        prefer_external=False,
    )
    fine = overclustered_leiden_labels(
        synthetic.X,
        seed=29,
        target_clusters=12,
        prefer_external=False,
    )
    assert len(np.unique(coarse)) <= len(np.unique(fine))


def test_advanced_search_phases_keep_exact_likelihood_accounting():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=5,
        n_genes=18,
        marker_strength=25.0,
        seed=31,
    )
    state = PartitionState.from_csr(synthetic.X, init="leiden_overclustered", seed=31, n_clusters=8)
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        n_proposals=300,
        seed=31,
        greedy_merge_sweeps=1,
        greedy_merge_candidates=128,
        cluster_reassign_sweeps=1,
        cluster_reassign_max_sources=16,
        perturb_every=1,
        perturb_steps=2,
        perturb_temperature=0.5,
        tau_update_interval=1,
        validate_batches=True,
    )
    result = optimizer.fit(max_rounds=2, update_psi=True, restarts=1, stall_rounds=3)

    assert result.history
    assert all("timing_s" in record for record in result.history)
    assert any(record["tau_update"]["updated"] for record in result.history)
    result.state.validate()


def test_optimizer_can_run_until_no_improvement():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=6,
        n_genes=20,
        marker_strength=28.0,
        seed=41,
    )
    state = PartitionState.from_csr(synthetic.X, init="leiden_overclustered", seed=41, n_clusters=10)
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        n_proposals=200,
        seed=41,
        cluster_reassign_sweeps=0,
        perturb_every=0,
        serial_refine_passes=0,
    )
    result = optimizer.fit(max_rounds=None, restarts=1, stall_rounds=1, eta=0.0)

    assert result.history
    final_round = result.history[-1]
    assert abs(final_round["log_likelihood_after"] - final_round["log_likelihood_before"]) < 1e-10


def test_gpu_heavy_mode_disables_cpu_heavy_refinement_phases():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=6,
        n_genes=20,
        marker_strength=28.0,
        seed=43,
    )
    state = PartitionState.from_csr(synthetic.X, init="leiden_overclustered", seed=43, n_clusters=10)
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        optimizer_mode="gpu-heavy",
        backend="cpu",
        n_proposals=200,
        seed=43,
    )

    assert optimizer.full_merge_stage is False
    assert optimizer.greedy_merge_sweeps == 0
    assert optimizer.exact_cell_reassign_passes == 0
    assert optimizer.cluster_reassign_sweeps == 0
    assert optimizer.serial_refine_passes == 0
    assert optimizer.perturb_every == 0
    assert optimizer.perturb_steps == 0
