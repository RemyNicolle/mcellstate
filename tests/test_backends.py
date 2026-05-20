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
    gpu = TorchCudaBackend(psi, state)

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
