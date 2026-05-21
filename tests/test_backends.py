import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mcellstate.backends import (
    CPUBackend,
    TorchCPUBackend,
    TorchCudaBackend,
    _torch_device_supports_float64,
    make_backend,
)
from mcellstate.proposals import (
    BlockPayload,
    BlockMoveProposal,
    BlockPeelProposal,
    MergeProposal,
    MoveProposal,
    PeelProposal,
)
from mcellstate.state import PartitionState
from mcellstate.validation import generate_synthetic_dataset


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_backend_matches_cpu_scores():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=25.0,
        seed=11,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
    )
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)
    block_01 = BlockPayload.from_cells(state, (0, 1))
    block_45 = BlockPayload.from_cells(state, (4, 5))

    proposals = [
        MergeProposal(0, 1),
        MergeProposal(2, 3),
        MoveProposal(cell=0, source_cluster=0, target_cluster=1),
        MoveProposal(cell=5, source_cluster=2, target_cluster=1),
        PeelProposal(cell=0, source_cluster=0),
        BlockPeelProposal(block=block_01, source_cluster=0),
        BlockMoveProposal(block=block_45, source_cluster=2, target_cluster=1),
    ]

    cpu = CPUBackend(psi, state)
    gpu = TorchCudaBackend(psi, state, chunk_size=None)

    cpu_scores = cpu.score_batch(state, proposals)
    gpu_scores = gpu.score_batch(state, proposals)
    assert np.allclose(cpu_scores, gpu_scores, atol=1e-8, rtol=1e-8)


def test_auto_backend_falls_back_to_cpu_without_accelerator_float64_support():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=2,
        n_genes=8,
        seed=13,
    )
    state = PartitionState.from_csr(synthetic.X, init="singletons")
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)

    backend = make_backend("auto", psi, state)
    if torch.cuda.is_available() and _torch_device_supports_float64("cuda"):
        assert backend.__class__.__name__ == "TorchCudaBackend"
    elif torch is not None:
        assert backend.__class__.__name__ == "TorchCPUBackend"
    else:
        assert backend.__class__.__name__ == "CPUBackend"


def test_torch_cpu_backend_matches_reference_scores():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=3,
        n_genes=10,
        seed=19,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 1, 1, 2, 2]),
    )
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)
    block_01 = BlockPayload.from_cells(state, (0, 1))
    block_23 = BlockPayload.from_cells(state, (2, 3))
    proposals = [
        MergeProposal(0, 1),
        MoveProposal(cell=0, source_cluster=0, target_cluster=1),
        PeelProposal(cell=0, source_cluster=0),
        BlockPeelProposal(block=block_01, source_cluster=0),
        BlockMoveProposal(block=block_23, source_cluster=1, target_cluster=0),
    ]

    cpu = CPUBackend(psi, state)
    torch_cpu = TorchCPUBackend(psi, state, num_threads=2)
    assert np.allclose(
        cpu.score_batch(state, proposals), torch_cpu.score_batch(state, proposals)
    )


def test_parallel_cpu_backend_matches_reference_scores():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=4,
        n_genes=12,
        seed=21,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
    )
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)
    block_01 = BlockPayload.from_cells(state, (0, 1))
    block_45 = BlockPayload.from_cells(state, (4, 5))
    proposals = [
        MergeProposal(0, 1),
        MergeProposal(2, 3),
        MoveProposal(cell=0, source_cluster=0, target_cluster=1),
        MoveProposal(cell=5, source_cluster=2, target_cluster=1),
        PeelProposal(cell=0, source_cluster=0),
        BlockPeelProposal(block=block_01, source_cluster=0),
        BlockMoveProposal(block=block_45, source_cluster=2, target_cluster=1),
    ]

    cpu = CPUBackend(psi, state, num_threads=1)
    cpu_parallel = CPUBackend(psi, state, num_threads=2)
    assert np.allclose(
        cpu.score_batch(state, proposals), cpu_parallel.score_batch(state, proposals)
    )


def test_backends_mark_stale_proposals_as_negative_infinity():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=3,
        n_genes=10,
        seed=22,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 0, 1, 1, 1]),
    )
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)
    state.merge_clusters(0, 1)
    stale = [
        MergeProposal(0, 1),
        MoveProposal(cell=0, source_cluster=0, target_cluster=1),
        PeelProposal(cell=3, source_cluster=1),
    ]

    cpu = CPUBackend(psi, state)
    torch_cpu = TorchCPUBackend(psi, state, num_threads=2)

    assert np.all(np.isneginf(cpu.score_batch(state, stale)))
    assert np.all(np.isneginf(torch_cpu.score_batch(state, stale)))


def test_torch_backend_can_generate_random_proposals():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=25.0,
        seed=24,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
    )
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)
    backend = TorchCPUBackend(psi, state, num_threads=2)

    proposals = backend.sample_random_proposals(
        state,
        64,
        family_weights=np.asarray([0.8, 0.1, 0.1, 0.0, 0.0], dtype=np.float64),
        max_unique_proposals=32,
        seed=24,
    )

    assert proposals
    assert len(proposals) <= 32
    assert all(
        isinstance(proposal, (MergeProposal, PeelProposal, MoveProposal))
        for proposal in proposals
    )


def test_torch_backend_selects_nonconflicting_candidates_on_tensor_path():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=2,
        n_genes=12,
        marker_strength=25.0,
        seed=25,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 1, 1, 2, 2]),
    )
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)
    backend = TorchCPUBackend(psi, state, num_threads=2)

    candidates = [
        {
            "proposal": MergeProposal(0, 1),
            "delta": 10.0,
            "touch_set": frozenset((0, 1)),
            "touch_ids": (0, 1),
        },
        {
            "proposal": PeelProposal(cell=4, source_cluster=2),
            "delta": 9.0,
            "touch_set": frozenset((2,)),
            "touch_ids": (2,),
        },
        {
            "proposal": MoveProposal(cell=5, source_cluster=2, target_cluster=0),
            "delta": 8.0,
            "touch_set": frozenset((0, 2)),
            "touch_ids": (0, 2),
        },
        {
            "proposal": MoveProposal(cell=2, source_cluster=1, target_cluster=0),
            "delta": 7.0,
            "touch_set": frozenset((0, 1)),
            "touch_ids": (1, 0),
        },
    ]

    accepted = backend.select_nonconflicting_candidates(candidates)

    assert [item["proposal"] for item in accepted] == [
        candidates[0]["proposal"],
        candidates[1]["proposal"],
    ]


def test_torch_backend_state_cache_syncs_incrementally():
    synthetic = generate_synthetic_dataset(
        n_clusters=3,
        cells_per_cluster=2,
        n_genes=10,
        marker_strength=25.0,
        seed=26,
    )
    state = PartitionState.from_csr(
        synthetic.X,
        init=np.asarray([0, 0, 1, 1, 2, 2]),
    )
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)
    backend = TorchCPUBackend(psi, state, num_threads=2)

    state.merge_clusters(0, 1)
    backend.score_batch(state, [MergeProposal(0, 1)])

    assert np.array_equal(backend.z_tensor.cpu().numpy(), state.z)
    assert int(backend.cluster_totals[1].item()) == 0
    assert int(backend.cluster_sizes_tensor[0].item()) == state.cluster_size(0)
    active_ids = backend.active_cluster_ids_tensor[
        : backend._state_cache.active_cluster_count
    ].cpu()
    assert set(active_ids.tolist()) == set(state.active_cluster_ids)


def test_mps_backend_is_not_supported():
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=2,
        n_genes=8,
        seed=17,
    )
    state = PartitionState.from_csr(synthetic.X, init="singletons")
    psi = np.full(state.n_genes, 0.5, dtype=np.float64)

    with pytest.raises(ValueError, match="unknown backend: mps"):
        make_backend("mps", psi, state)
