from __future__ import annotations

import numpy as np

from mcellstate.prior import make_prior
from mcellstate.proposals import BlockMoveProposal, MoveProposal, ProposalSampler
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
        proposal_workers=1,
        seed=73,
    )
    sampler.set_scoring_context(psi)

    def forbidden(*args, **kwargs):
        raise AssertionError("expression-ranked move targets should not be used")

    sampler.rank_target_clusters = forbidden  # type: ignore[method-assign]
    sampler._candidate_targets_for_payload = forbidden  # type: ignore[method-assign]

    proposals = sampler.sample_batch(state, 64)

    assert len(proposals) == 64
    assert all(isinstance(proposal, MoveProposal) for proposal in proposals)


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
        proposal_workers=1,
        seed=74,
    )
    sampler.set_scoring_context(psi)

    def forbidden(*args, **kwargs):
        raise AssertionError("expression-ranked move targets should not be used")

    sampler.rank_target_clusters = forbidden  # type: ignore[method-assign]
    sampler._candidate_targets_for_payload = forbidden  # type: ignore[method-assign]

    proposals = sampler.sample_batch(state, 32)

    assert len(proposals) == 32
    assert all(isinstance(proposal, BlockMoveProposal) for proposal in proposals)
