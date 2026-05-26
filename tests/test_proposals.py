from __future__ import annotations

import numpy as np

from mcellstate.prior import make_prior
from mcellstate.proposals import (
    BlockMoveProposal,
    MergeProposal,
    MoveProposal,
    ProposalSampler,
)
from mcellstate.state import PartitionState
from mcellstate.validation import generate_synthetic_dataset


def test_sampler_notify_state_changed_keeps_incremental_merge_rebuild():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=30.0,
        seed=71,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    )
    psi = make_prior(synthetic.X, tau=1.0)

    sampler = ProposalSampler(seed=71, proposal_workers=1)
    sampler.set_scoring_context(psi)

    rebuild_calls: list[bool] = []
    original = sampler._rebuild_merge_neighbors

    def wrapped(state, *, full_rebuild):
        rebuild_calls.append(bool(full_rebuild))
        return original(state, full_rebuild=full_rebuild)

    sampler._rebuild_merge_neighbors = wrapped  # type: ignore[method-assign]

    sampler.prepare_round(state)
    state._note_state_change([0], [0])
    sampler.notify_state_changed(state)
    sampler.prepare_round(state)

    assert rebuild_calls[0] is True
    assert rebuild_calls[-1] is False


def test_incremental_merge_rebuild_prunes_stale_targets():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=3,
        n_genes=12,
        marker_strength=30.0,
        seed=72,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64),
    )
    psi = make_prior(synthetic.X, tau=1.0)

    sampler = ProposalSampler(
        pi_merge=1.0,
        pi_peel=0.0,
        pi_move=0.0,
        pi_block_peel=0.0,
        pi_block_move=0.0,
        deterministic_merge_ratio=1.0,
        seed=72,
    )
    sampler.set_scoring_context(psi)
    sampler.prepare_round(state)

    sampler._merge_neighbors = {
        0: np.asarray([1, 2], dtype=np.int64),
        1: np.asarray([0, 2], dtype=np.int64),
    }
    sampler._merge_neighbor_weights = {
        0: np.asarray([0.5, 0.5], dtype=np.float64),
        1: np.asarray([0.5, 0.5], dtype=np.float64),
    }
    sampler._merge_pair_scores = {
        (0, 1): 1.0,
        (0, 2): 2.0,
        (1, 2): 3.0,
    }
    sampler._deterministic_merge_pairs = [(1, 2), (0, 2), (0, 1)]

    state.merge_clusters(0, 2)
    sampler.notify_state_changed(state)
    sampler.prepare_round(state)

    active = set(state.active_cluster_ids)
    assert 2 not in active
    assert all(
        cluster_a in active and cluster_b in active
        for cluster_a, cluster_b in sampler._deterministic_merge_pairs
    )
    assert all(
        np.all(np.isin(neighbors, list(active)))
        for neighbors in sampler._merge_neighbors.values()
    )
    assert all(
        proposal.cluster_a in active and proposal.cluster_b in active
        for proposal in sampler.sample_batch(state, 16)
        if isinstance(proposal, MergeProposal)
    )


def test_move_family_sampling_uses_uniform_fast_path():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=30.0,
        seed=73,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    )
    psi = make_prior(synthetic.X, tau=1.0)

    sampler = ProposalSampler(
        pi_merge=0.0,
        pi_peel=0.0,
        pi_move=1.0,
        pi_block_peel=0.0,
        pi_block_move=0.0,
        move_uniform_prob=1.0,
        proposal_workers=1,
        seed=73,
    )
    sampler.set_scoring_context(psi)

    def forbidden(*args, **kwargs):
        raise AssertionError("expression-ranked move targets should not be used")

    sampler.rank_target_clusters = forbidden  # type: ignore[method-assign]
    sampler._candidate_targets_for_payload = forbidden  # type: ignore[method-assign]

    proposals = sampler.sample_batch(state, 64)

    assert 0 < len(proposals) <= 64
    assert all(isinstance(proposal, MoveProposal) for proposal in proposals)


def test_random_proposals_skip_guided_precomputation():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=30.0,
        seed=75,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    )

    sampler = ProposalSampler(
        pi_merge=1.0,
        pi_peel=0.0,
        pi_move=0.0,
        pi_block_peel=0.0,
        pi_block_move=0.0,
        random_proposals=True,
        seed=75,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("guided proposal precomputation should be skipped")

    sampler._refresh_bad_cell_cache = forbidden  # type: ignore[method-assign]
    sampler._refresh_signature_pools = forbidden  # type: ignore[method-assign]
    sampler._refresh_current_signatures = forbidden  # type: ignore[method-assign]
    sampler._rebuild_merge_neighbors = forbidden  # type: ignore[method-assign]

    proposals = sampler.sample_batch(state, 16)

    assert proposals
    assert all(isinstance(proposal, MergeProposal) for proposal in proposals)
    assert sampler._deterministic_merge_pairs == []


def test_random_sampler_caps_unique_proposals():
    synthetic = generate_synthetic_dataset(
        n_clusters=4,
        cells_per_cluster=5,
        n_genes=12,
        marker_strength=30.0,
        seed=76,
    )
    state = PartitionState.from_csr(synthetic.X, init="singletons")

    sampler = ProposalSampler(
        pi_merge=1.0,
        pi_peel=0.0,
        pi_move=0.0,
        pi_block_peel=0.0,
        pi_block_move=0.0,
        random_proposals=True,
        max_unique_proposals=7,
        seed=76,
    )

    proposals = sampler.sample_batch(state, 128)

    assert sampler._proposal_draw_count(128) == 9
    assert 0 < len(proposals) <= 7
    assert all(isinstance(proposal, MergeProposal) for proposal in proposals)


def test_sample_batch_offloads_guided_merge_family_to_backend():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=25.0,
        seed=76,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
    )
    sampler = ProposalSampler(
        pi_merge=1.0,
        pi_peel=0.0,
        pi_move=0.0,
        pi_block_peel=0.0,
        pi_block_move=0.0,
        deterministic_merge_ratio=0.0,
        proposal_workers=1,
        seed=76,
    )

    calls: list[tuple[int, float]] = []

    class DummyBackend:
        def sample_guided_merge_pairs(
            self,
            state_arg,
            n_pairs,
            *,
            epsilon_uniform,
            include_cached_pairs,
            max_unique_pairs,
            seed,
        ):
            del state_arg, include_cached_pairs, max_unique_pairs, seed
            calls.append((int(n_pairs), float(epsilon_uniform)))
            return [MergeProposal(0, 1)]

    proposals = sampler.sample_batch(state, 8, backend=DummyBackend())

    assert calls == [(8, sampler.merge_uniform_prob)]
    assert proposals
    assert any(isinstance(proposal, MergeProposal) for proposal in proposals)


def test_sample_batch_offloads_guided_move_family_to_backend():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=25.0,
        seed=77,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
    )
    sampler = ProposalSampler(
        pi_merge=0.0,
        pi_peel=0.0,
        pi_move=1.0,
        pi_block_peel=0.0,
        pi_block_move=0.0,
        proposal_workers=1,
        seed=77,
    )

    calls: list[tuple[int, float, int]] = []

    class DummyBackend:
        def sample_guided_move_proposals(
            self,
            state_arg,
            n_proposals,
            *,
            uniform_prob,
            limit,
            max_unique_proposals,
            seed,
        ):
            del state_arg, max_unique_proposals, seed
            calls.append((int(n_proposals), float(uniform_prob), int(limit)))
            return [
                MoveProposal(cell=0, source_cluster=int(state.z[0]), target_cluster=1)
            ]

    proposals = sampler.sample_batch(state, 8, backend=DummyBackend())

    assert calls == [(8, sampler.move_uniform_prob, sampler.move_neighbor_limit)]
    assert proposals
    assert any(isinstance(proposal, MoveProposal) for proposal in proposals)


def test_block_move_family_sampling_uses_uniform_fast_path():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=30.0,
        seed=74,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    )
    psi = make_prior(synthetic.X, tau=1.0)

    sampler = ProposalSampler(
        pi_merge=0.0,
        pi_peel=0.0,
        pi_move=0.0,
        pi_block_peel=0.0,
        pi_block_move=1.0,
        block_move_uniform_prob=1.0,
        proposal_workers=1,
        seed=74,
    )
    sampler.set_scoring_context(psi)

    def forbidden(*args, **kwargs):
        raise AssertionError("expression-ranked move targets should not be used")

    sampler.rank_target_clusters = forbidden  # type: ignore[method-assign]
    sampler._candidate_targets_for_payload = forbidden  # type: ignore[method-assign]

    proposals = sampler.sample_batch(state, 32)

    assert 0 < len(proposals) <= 32
    assert all(isinstance(proposal, BlockMoveProposal) for proposal in proposals)
