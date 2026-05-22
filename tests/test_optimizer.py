import numpy as np
from scipy import sparse

from mcellstate.backends import CPUBackend
from mcellstate.likelihood import full_partition_log_likelihood
from mcellstate.optimizer import Optimizer
from mcellstate.prior import make_prior
from mcellstate.proposals import (
    BlockMoveProposal,
    MergeProposal,
    MoveProposal,
    PeelProposal,
)
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
    state = PartitionState.from_csr(
        synthetic.X, init="leiden_overclustered", seed=31, n_clusters=8
    )
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
    state = PartitionState.from_csr(
        synthetic.X, init="leiden_overclustered", seed=41, n_clusters=10
    )
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
    assert (
        abs(final_round["log_likelihood_after"] - final_round["log_likelihood_before"])
        < 1e-10
    )


def test_gpu_fast_weak_policy_keeps_legacy_disabled_phases():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=6,
        n_genes=20,
        marker_strength=28.0,
        seed=43,
    )
    state = PartitionState.from_csr(
        synthetic.X, init="leiden_overclustered", seed=43, n_clusters=10
    )
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        optimizer_mode="gpu",
        backend="cpu",
        n_proposals=200,
        seed=43,
    )

    assert optimizer.search_policy == Optimizer.GPU_FAST_WEAK_POLICY
    assert optimizer.full_merge_stage is False
    assert optimizer.greedy_merge_sweeps == 0
    assert optimizer.exact_cell_reassign_passes == 0
    assert optimizer.cluster_reassign_sweeps == 0
    assert optimizer.serial_refine_passes == 0
    assert optimizer.perturb_every == 0
    assert optimizer.perturb_steps == 0
    assert optimizer.backend_threads >= 2
    assert optimizer.recompute_ll_each_round is False
    assert optimizer.cuda_chunk_size == 8192
    assert optimizer.sampler.random_proposals is True
    assert optimizer.sampler.max_unique_proposals == 200
    optimizer._configure_stage(state, "coarsen")
    weights = dict(
        zip(
            optimizer.sampler.family_names,
            optimizer.sampler.family_weights.tolist(),
            strict=True,
        )
    )
    assert weights["block_peel"] == 0.0
    assert weights["block_move"] == 0.0


def test_cuda_backend_defaults_to_structured_search_policy():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=5,
        n_genes=18,
        marker_strength=25.0,
        seed=70,
    )
    state = PartitionState.from_csr(
        synthetic.X, init="leiden_overclustered", seed=70, n_clusters=9
    )
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cuda",
        optimizer_mode=Optimizer.EFFECTIVE_MODE,
        n_proposals=256,
        seed=70,
    )

    assert optimizer.search_policy == Optimizer.GPU_STRUCTURED_POLICY
    assert optimizer.merge_closure_enabled is True
    assert optimizer.random_proposals is False
    assert optimizer.greedy_merge_sweeps >= 1
    assert optimizer.exact_cell_reassign_passes >= 1
    assert optimizer.cluster_reassign_sweeps >= 1
    assert optimizer.serial_refine_passes >= 1


def test_merge_closure_recovers_artificially_split_clusters():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=16,
        marker_strength=32.0,
        seed=71,
    )
    z_split = np.asarray([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    state = PartitionState.from_assignment(synthetic.X, z_split)
    psi = make_prior(synthetic.X, tau=1.0)
    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        search_policy=Optimizer.GPU_STRUCTURED_POLICY,
        n_proposals=256,
        seed=71,
        full_merge_stage=False,
        greedy_merge_sweeps=0,
        exact_cell_reassign_passes=0,
        cluster_reassign_sweeps=0,
        serial_refine_passes=0,
        merge_closure_max_batches=4,
        merge_closure_pairs_per_batch=128,
    )
    backend = CPUBackend(psi, state)
    before_ll = full_partition_log_likelihood(state, psi)

    result = optimizer._merge_closure_phase(
        state,
        backend,
        stage="coarsen",
        reason="test",
    )
    after_ll = full_partition_log_likelihood(state, psi)

    assert result["n_steps"] >= 1
    assert after_ll > before_ll
    assert len(state.active_cluster_ids) < len(np.unique(z_split))
    state.validate()


def test_guided_move_sampler_prefers_overlap_target_with_uniform_support():
    X = sparse.csr_matrix(
        np.asarray(
            [
                [8, 7, 0],
                [7, 6, 0],
                [9, 8, 0],
                [8, 7, 0],
                [0, 0, 9],
                [0, 0, 8],
            ],
            dtype=np.int64,
        ),
    )
    state = PartitionState.from_assignment(
        X,
        np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
    )
    psi = np.full(X.shape[1], 0.5, dtype=np.float64)
    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        n_proposals=64,
        seed=72,
    )
    sampler = optimizer.sampler
    sampler.move_uniform_prob = 0.15
    sampler.prepare_round(state)
    sampler._sample_non_singleton_cluster = lambda state_arg: 0  # type: ignore[method-assign]
    sampler._sample_low_fit_cell = (  # type: ignore[method-assign]
        lambda state_arg, source_cluster: 0
    )
    sampler._sample_non_singleton_cell = lambda state_arg: (0, 0)  # type: ignore[method-assign]

    counts = {1: 0, 2: 0}
    for _ in range(400):
        proposal = sampler._sample_move_biased(state)
        assert proposal is not None
        counts[int(proposal.target_cluster)] += 1

    assert counts[1] > counts[2]
    assert counts[2] > 0


def test_block_move_grouping_adds_exact_positive_block_and_preserves_counts():
    X = sparse.csr_matrix(
        np.asarray(
            [
                [8, 0, 0],
                [7, 0, 0],
                [0, 8, 0],
                [9, 0, 0],
                [8, 0, 0],
                [0, 8, 0],
                [0, 7, 0],
            ],
            dtype=np.int64,
        ),
    )
    state = PartitionState.from_assignment(
        X,
        np.asarray([0, 0, 0, 1, 1, 2, 2], dtype=np.int64),
    )
    psi = np.full(X.shape[1], 0.5, dtype=np.float64)
    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        search_policy=Optimizer.GPU_STRUCTURED_POLICY,
        n_proposals=64,
        seed=73,
    )
    backend = CPUBackend(psi, state)
    proposals = [
        MoveProposal(cell=0, source_cluster=0, target_cluster=1),
        MoveProposal(cell=1, source_cluster=0, target_cluster=1),
    ]
    scores = backend.score_batch(state, proposals)

    augmented, augmented_scores, stats = optimizer._augment_with_grouped_block_moves(
        state,
        backend,
        proposals,
        scores,
    )
    block_indices = [
        idx
        for idx, proposal in enumerate(augmented)
        if isinstance(proposal, BlockMoveProposal)
    ]

    assert stats["positive_single_moves"] == 2
    assert stats["source_target_groups"] == 1
    assert stats["positive_blocks"] >= 1
    assert block_indices

    block_idx = block_indices[0]
    block_proposal = augmented[block_idx]
    block_delta = float(augmented_scores[block_idx])
    assert isinstance(block_proposal, BlockMoveProposal)

    before_ll = full_partition_log_likelihood(state, psi)
    touched = optimizer._commit_batch(
        state,
        [
            {
                "proposal": block_proposal,
                "delta": block_delta,
                "touch_set": block_proposal.touch_set(),
                "touch_ids": block_proposal.touch_ids(),
            }
        ],
    )
    after_ll = full_partition_log_likelihood(state, psi)

    assert touched == {0, 1}
    assert np.isclose(after_ll - before_ll, block_delta)
    state.validate()


def test_gpu_structured_reaches_at_least_gpu_fast_weak_likelihood_on_synthetic():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=4,
        n_genes=16,
        marker_strength=28.0,
        seed=74,
    )
    init_state = PartitionState.from_csr(
        synthetic.X, init="leiden_overclustered", seed=74, n_clusters=10
    )
    psi = make_prior(synthetic.X, tau=1.0)

    fast = Optimizer(
        state=init_state.copy(),
        psi=psi,
        backend="cpu",
        search_policy=Optimizer.GPU_FAST_WEAK_POLICY,
        n_proposals=512,
        seed=74,
    ).fit(
        max_rounds=6,
        restarts=1,
        stall_rounds=3,
        improvement_window=3,
        eta=0.0,
    )
    structured = Optimizer(
        state=init_state.copy(),
        psi=psi,
        backend="cpu",
        search_policy=Optimizer.GPU_STRUCTURED_POLICY,
        n_proposals=512,
        seed=74,
    ).fit(
        max_rounds=6,
        restarts=1,
        stall_rounds=3,
        improvement_window=3,
        eta=0.0,
    )

    assert structured.log_likelihood >= fast.log_likelihood


def test_random_walk_accepts_finite_bad_non_merge_moves():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=3,
        n_genes=12,
        marker_strength=28.0,
        seed=46,
    )
    state = PartitionState.from_csr(
        synthetic.X, init=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    )
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        n_proposals=16,
        seed=46,
        random_accept_prob=1.0,
        random_accept_max_fraction=1.0,
        cluster_reassign_sweeps=0,
        perturb_every=0,
        serial_refine_passes=0,
    )
    proposals = [
        MergeProposal(0, 1),
        MoveProposal(cell=0, source_cluster=0, target_cluster=1),
        PeelProposal(cell=1, source_cluster=0),
    ]
    scores = np.asarray([-1.0, -2.0, -3.0], dtype=np.float64)

    accepted = optimizer._select_positive_nonconflicting(proposals, scores)

    assert accepted
    assert all(item.get("random_walk", False) for item in accepted)
    assert all(not isinstance(item["proposal"], MergeProposal) for item in accepted)


def test_optimizer_uses_backend_selection_hook_when_available():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=2,
        n_genes=12,
        marker_strength=25.0,
        seed=47,
    )
    state = PartitionState.from_csr(
        synthetic.X, init=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    )
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        n_proposals=16,
        seed=47,
        multi_greedy_trials=1,
        cluster_reassign_sweeps=0,
        perturb_every=0,
        serial_refine_passes=0,
        random_accept_prob=0.0,
    )

    candidates = [
        {
            "proposal": MergeProposal(0, 1),
            "delta": 4.0,
            "touch_set": frozenset((0, 1)),
            "touch_ids": (0, 1),
        },
        {
            "proposal": PeelProposal(cell=4, source_cluster=2),
            "delta": 3.0,
            "touch_set": frozenset((2,)),
            "touch_ids": (2,),
        },
    ]

    calls: list[int] = []

    class DummyBackend:
        def select_nonconflicting_candidates(self, items):
            calls.append(len(items))
            return [items[1]]

    accepted = optimizer._select_positive_nonconflicting_candidates(
        candidates, backend=DummyBackend()
    )

    assert calls == [2]
    assert accepted == [candidates[1]]


def test_optimizer_samples_proposals_in_chunks():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=2,
        n_genes=12,
        marker_strength=25.0,
        seed=48,
    )
    state = PartitionState.from_csr(
        synthetic.X, init=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    )
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="cpu",
        n_proposals=5,
        seed=48,
        proposal_batch_size=2,
        cluster_reassign_sweeps=0,
        perturb_every=0,
        serial_refine_passes=0,
        random_accept_prob=0.0,
    )

    calls: list[int] = []

    def fake_sample_chunk(state_arg, backend_arg, count, seed):  # noqa: ARG001
        calls.append(int(count))
        return [MergeProposal(0, 1)] * int(count)

    class DummyBackend:
        def score_batch(self, state_arg, proposals):  # noqa: ARG002
            return np.ones(len(proposals), dtype=np.float64)

    optimizer._sample_proposal_chunk = fake_sample_chunk  # type: ignore[method-assign]
    (
        proposals,
        scores,
        proposal_s,
        scoring_s,
        chunk_count,
    ) = optimizer._sample_and_score_proposals(  # noqa: E501
        state,
        DummyBackend(),
    )

    assert calls == [2, 2, 1]
    assert chunk_count == 3
    assert len(proposals) == 5
    assert scores.shape == (5,)
    assert proposal_s >= 0.0
    assert scoring_s >= 0.0


def test_cpu_only_mode_enables_parallel_proposal_sampling_defaults():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=6,
        n_genes=20,
        marker_strength=28.0,
        seed=45,
    )
    state = PartitionState.from_csr(
        synthetic.X, init="leiden_overclustered", seed=45, n_clusters=10
    )
    psi = make_prior(synthetic.X, tau=1.0)

    optimizer = Optimizer(
        state=state,
        psi=psi,
        optimizer_mode="cpu-only",
        backend="cpu",
        n_proposals=200,
        seed=45,
    )
    optimizer._configure_stage(state, "coarsen")
    weights = dict(
        zip(
            optimizer.sampler.family_names,
            optimizer.sampler.family_weights.tolist(),
            strict=True,
        )
    )

    assert optimizer.proposal_workers >= 2
    assert optimizer.backend_threads >= 2
    assert optimizer.recompute_ll_each_round is True
    assert weights["move"] > weights["peel"]
