from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .likelihood import (
    cluster_log_likelihood,
    cluster_log_likelihood_from_sparse,
    delta_add_sparse,
    delta_move_block,
    delta_move_cell,
    delta_peel_block,
    delta_peel_cell,
)
from .proposals import (
    BlockMoveProposal,
    BlockPeelProposal,
    MergeProposal,
    MoveProposal,
    PeelProposal,
    Proposal,
)
from .state import PartitionState

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


_TORCH_CPU_THREADS_CONFIGURED: int | None = None


class CPUBackend:
    def __init__(self, psi: np.ndarray, state: PartitionState, *, num_threads: int | None = None) -> None:
        self.psi = np.asarray(psi, dtype=np.float64)
        self.psi0 = float(self.psi.sum())
        self.num_threads = max(1, int(num_threads) if num_threads is not None else 1)
        self.cell_ll = self._precompute_cell_ll(state)

    def _parallel_ranges(self, n_items: int, *, min_items_per_worker: int = 32) -> list[tuple[int, int]]:
        n_items = int(n_items)
        if self.num_threads <= 1 or n_items <= min_items_per_worker:
            return [(0, n_items)] if n_items > 0 else []
        max_workers = min(self.num_threads, max(1, n_items // max(1, min_items_per_worker)))
        max_workers = max(1, min(max_workers, n_items))
        if max_workers <= 1:
            return [(0, n_items)]
        chunk = (n_items + max_workers - 1) // max_workers
        return [(start, min(start + chunk, n_items)) for start in range(0, n_items, chunk)]

    def _run_parallel_ranges(
        self,
        ranges: list[tuple[int, int]],
        worker,
    ) -> None:
        if len(ranges) <= 1:
            if ranges:
                start, end = ranges[0]
                worker(start, end)
            return
        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [executor.submit(worker, start, end) for start, end in ranges]
            for future in futures:
                future.result()

    def _precompute_cell_ll(self, state: PartitionState) -> np.ndarray:
        values = np.empty(state.n_cells, dtype=np.float64)
        ranges = self._parallel_ranges(state.n_cells, min_items_per_worker=64)

        def fill_range(start: int, end: int) -> None:
            for cell in range(start, end):
                indices, counts = state.cell_counts(cell)
                values[cell] = cluster_log_likelihood_from_sparse(indices, counts, self.psi, psi0=self.psi0)

        self._run_parallel_ranges(ranges, fill_range)
        return values

    def score_batch(self, state: PartitionState, proposals: list[Proposal]) -> np.ndarray:
        if not proposals:
            return np.empty(0, dtype=np.float64)
        merge_small_ll: dict[int, float] = {}
        merge_cluster_ids = sorted(
            {
                int(cluster_id)
                for proposal in proposals
                if isinstance(proposal, MergeProposal)
                for cluster_id in (proposal.cluster_a, proposal.cluster_b)
            },
        )
        for cluster_id in merge_cluster_ids:
            merge_small_ll[int(cluster_id)] = state.cluster_log_likelihood_cached(int(cluster_id), self.psi)

        scores = np.empty(len(proposals), dtype=np.float64)
        ranges = self._parallel_ranges(len(proposals), min_items_per_worker=64)
        if len(ranges) <= 1:
            for idx, proposal in enumerate(proposals):
                scores[idx] = self._score_single_proposal(state, proposal, merge_small_ll)
            return scores

        def score_range(start: int, end: int) -> None:
            for idx in range(start, end):
                scores[idx] = self._score_single_proposal(state, proposals[idx], merge_small_ll)

        self._run_parallel_ranges(ranges, score_range)
        return scores

    def _score_single_proposal(
        self,
        state: PartitionState,
        proposal: Proposal,
        merge_small_ll: dict[int, float],
    ) -> float:
            if isinstance(proposal, MergeProposal):
                vector_a = state.clusters[proposal.cluster_a]
                vector_b = state.clusters[proposal.cluster_b]
                if vector_a.nnz <= vector_b.nnz:
                    small, large = vector_a, vector_b
                else:
                    small, large = vector_b, vector_a
                indices, values = small.sorted_items()
                delta = delta_add_sparse(large, indices, values, self.psi, psi0=self.psi0)
                delta -= merge_small_ll[proposal.cluster_a if small is vector_a else proposal.cluster_b]
                return float(delta)
            elif isinstance(proposal, PeelProposal):
                return float(delta_peel_cell(state, self.psi, proposal.cell, proposal.source_cluster))
            elif isinstance(proposal, MoveProposal):
                return float(
                    delta_move_cell(
                    state,
                    self.psi,
                    proposal.cell,
                    proposal.target_cluster,
                    source_cluster=proposal.source_cluster,
                    ),
                )
            elif isinstance(proposal, BlockPeelProposal):
                return float(
                    delta_peel_block(
                    state.clusters[proposal.source_cluster],
                    proposal.block.indices,
                    proposal.block.values,
                    self.psi,
                    ),
                )
            elif isinstance(proposal, BlockMoveProposal):
                return float(
                    delta_move_block(
                    state.clusters[proposal.source_cluster],
                    state.clusters[proposal.target_cluster],
                    proposal.block.indices,
                    proposal.block.values,
                    self.psi,
                    ),
                )
            else:  # pragma: no cover - defensive branch
                raise TypeError(f"unsupported proposal type: {type(proposal)!r}")


class TorchDeviceBackend(CPUBackend):
    def __init__(
        self,
        psi: np.ndarray,
        state: PartitionState,
        *,
        device: str,
        chunk_size: int = 8192,
        num_threads: int | None = None,
    ) -> None:
        global _TORCH_CPU_THREADS_CONFIGURED
        _require_torch_device(device)
        if device == "cpu" and num_threads is not None:
            desired_threads = int(num_threads)
            if _TORCH_CPU_THREADS_CONFIGURED is None:
                torch.set_num_threads(desired_threads)
                _TORCH_CPU_THREADS_CONFIGURED = desired_threads
        super().__init__(psi, state, num_threads=num_threads)
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.psi0_tensor = torch.tensor(self.psi0, dtype=torch.float64, device=self.device)

    def score_batch(self, state: PartitionState, proposals: list[Proposal]) -> np.ndarray:
        if not proposals:
            return np.empty(0, dtype=np.float64)

        scores = np.empty(len(proposals), dtype=np.float64)
        by_type: dict[str, list[tuple[int, Proposal]]] = defaultdict(list)
        for idx, proposal in enumerate(proposals):
            by_type[proposal.kind].append((idx, proposal))

        for chunk in self._chunked(by_type.get("peel", [])):
            self._score_peels(state, chunk, scores)
        for chunk in self._chunked(by_type.get("move", [])):
            self._score_moves(state, chunk, scores)
        for chunk in self._chunked(by_type.get("block_peel", [])):
            self._score_block_peels(state, chunk, scores)
        for chunk in self._chunked(by_type.get("block_move", [])):
            self._score_block_moves(state, chunk, scores)
        for chunk in self._chunked(by_type.get("merge", [])):
            self._score_merges(state, chunk, scores)
        return scores

    def _fill_rows_parallel(
        self,
        n_rows: int,
        worker,
    ) -> None:
        ranges = self._parallel_ranges(n_rows, min_items_per_worker=32)
        self._run_parallel_ranges(ranges, worker)

    def _chunked(self, indexed: list[tuple[int, Proposal]]) -> list[list[tuple[int, Proposal]]]:
        if not indexed:
            return []
        return [indexed[start : start + self.chunk_size] for start in range(0, len(indexed), self.chunk_size)]

    def _score_peels(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        lengths = np.asarray([state.cell_nnz[proposal.cell] for proposal in batch], dtype=np.int64)
        max_len = int(lengths.max(initial=0))
        if max_len == 0:
            deltas = self._remove_totals_only(
                np.asarray([state.cluster_total(proposal.source_cluster) for proposal in batch], dtype=np.float64),
                np.asarray([state.cell_totals[proposal.cell] for proposal in batch], dtype=np.float64),
            )
            deltas += self.cell_ll[[proposal.cell for proposal in batch]]
            for (idx, _), delta in zip(indexed, deltas, strict=True):
                scores[idx] = delta
            return

        psi_pad = np.ones((len(batch), max_len), dtype=np.float64)
        source_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        cell_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        source_totals = np.empty(len(batch), dtype=np.float64)
        cell_totals = np.empty(len(batch), dtype=np.float64)

        def fill_rows(start: int, end: int) -> None:
            for row in range(start, end):
                proposal = batch[row]
                genes, counts = state.cell_counts(proposal.cell)
                length = len(genes)
                if length:
                    psi_pad[row, :length] = self.psi[genes]
                    cell_pad[row, :length] = counts
                    source_pad[row, :length] = state.clusters[proposal.source_cluster].get_many(genes)
                source_totals[row] = state.cluster_total(proposal.source_cluster)
                cell_totals[row] = state.cell_totals[proposal.cell]

        self._fill_rows_parallel(len(batch), fill_rows)

        deltas = self._remove_delta_tensor(source_pad, cell_pad, psi_pad, source_totals, cell_totals)
        deltas += self.cell_ll[[proposal.cell for proposal in batch]]
        for (idx, _), delta in zip(indexed, deltas, strict=True):
            scores[idx] = delta

    def _score_moves(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        lengths = np.asarray([state.cell_nnz[proposal.cell] for proposal in batch], dtype=np.int64)
        max_len = int(lengths.max(initial=0))
        if max_len == 0:
            source_delta = self._remove_totals_only(
                np.asarray([state.cluster_total(proposal.source_cluster) for proposal in batch], dtype=np.float64),
                np.asarray([state.cell_totals[proposal.cell] for proposal in batch], dtype=np.float64),
            )
            target_delta = self._add_totals_only(
                np.asarray([state.cluster_total(proposal.target_cluster) for proposal in batch], dtype=np.float64),
                np.asarray([state.cell_totals[proposal.cell] for proposal in batch], dtype=np.float64),
            )
            deltas = source_delta + target_delta
            for (idx, _), delta in zip(indexed, deltas, strict=True):
                scores[idx] = delta
            return

        psi_pad = np.ones((len(batch), max_len), dtype=np.float64)
        source_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        target_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        cell_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        source_totals = np.empty(len(batch), dtype=np.float64)
        target_totals = np.empty(len(batch), dtype=np.float64)
        cell_totals = np.empty(len(batch), dtype=np.float64)

        def fill_rows(start: int, end: int) -> None:
            for row in range(start, end):
                proposal = batch[row]
                genes, counts = state.cell_counts(proposal.cell)
                length = len(genes)
                if length:
                    psi_pad[row, :length] = self.psi[genes]
                    cell_pad[row, :length] = counts
                    source_pad[row, :length] = state.clusters[proposal.source_cluster].get_many(genes)
                    target_pad[row, :length] = state.clusters[proposal.target_cluster].get_many(genes)
                source_totals[row] = state.cluster_total(proposal.source_cluster)
                target_totals[row] = state.cluster_total(proposal.target_cluster)
                cell_totals[row] = state.cell_totals[proposal.cell]

        self._fill_rows_parallel(len(batch), fill_rows)

        deltas = self._remove_delta_tensor(source_pad, cell_pad, psi_pad, source_totals, cell_totals)
        deltas += self._add_delta_tensor(target_pad, cell_pad, psi_pad, target_totals, cell_totals)
        for (idx, _), delta in zip(indexed, deltas, strict=True):
            scores[idx] = delta

    def _score_merges(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        small_vectors = []
        large_vectors = []
        small_ll = np.empty(len(batch), dtype=np.float64)
        small_totals = np.empty(len(batch), dtype=np.float64)
        large_totals = np.empty(len(batch), dtype=np.float64)
        lengths = np.empty(len(batch), dtype=np.int64)

        for row, proposal in enumerate(batch):
            vector_a = state.clusters[proposal.cluster_a]
            vector_b = state.clusters[proposal.cluster_b]
            if vector_a.nnz <= vector_b.nnz:
                small, large = vector_a, vector_b
            else:
                small, large = vector_b, vector_a
            small_vectors.append(small)
            large_vectors.append(large)
            small_ll[row] = state.cluster_log_likelihood_cached(
                proposal.cluster_a if small is vector_a else proposal.cluster_b,
                self.psi,
            )
            small_totals[row] = small.total
            large_totals[row] = large.total
            lengths[row] = small.nnz

        max_len = int(lengths.max(initial=0))
        if max_len == 0:
            deltas = self._add_totals_only(large_totals, small_totals) - small_ll
            for (idx, _), delta in zip(indexed, deltas, strict=True):
                scores[idx] = delta
            return

        psi_pad = np.ones((len(batch), max_len), dtype=np.float64)
        large_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        small_pad = np.zeros((len(batch), max_len), dtype=np.float64)

        def fill_rows(start: int, end: int) -> None:
            for row in range(start, end):
                small = small_vectors[row]
                large = large_vectors[row]
                genes, counts = small.sorted_items()
                length = len(genes)
                if length:
                    psi_pad[row, :length] = self.psi[genes]
                    small_pad[row, :length] = counts
                    large_pad[row, :length] = large.get_many(genes)

        self._fill_rows_parallel(len(batch), fill_rows)

        deltas = self._add_delta_tensor(large_pad, small_pad, psi_pad, large_totals, small_totals) - small_ll
        for (idx, _), delta in zip(indexed, deltas, strict=True):
            scores[idx] = delta

    def _score_block_peels(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        block_ll = np.empty(len(batch), dtype=np.float64)
        source_totals = np.empty(len(batch), dtype=np.float64)
        block_totals = np.empty(len(batch), dtype=np.float64)
        lengths = np.asarray([proposal.block.indices.size for proposal in batch], dtype=np.int64)

        for row, proposal in enumerate(batch):
            block_ll[row] = cluster_log_likelihood_from_sparse(
                proposal.block.indices,
                proposal.block.values,
                self.psi,
                psi0=self.psi0,
            )
            source_totals[row] = state.cluster_total(proposal.source_cluster)
            block_totals[row] = proposal.block.total

        max_len = int(lengths.max(initial=0))
        if max_len == 0:
            deltas = self._remove_totals_only(source_totals, block_totals) + block_ll
            for (idx, _), delta in zip(indexed, deltas, strict=True):
                scores[idx] = delta
            return

        psi_pad = np.ones((len(batch), max_len), dtype=np.float64)
        source_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        block_pad = np.zeros((len(batch), max_len), dtype=np.float64)

        def fill_rows(start: int, end: int) -> None:
            for row in range(start, end):
                proposal = batch[row]
                genes = proposal.block.indices
                counts = proposal.block.values
                length = len(genes)
                if length:
                    psi_pad[row, :length] = self.psi[genes]
                    block_pad[row, :length] = counts
                    source_pad[row, :length] = state.clusters[proposal.source_cluster].get_many(genes)

        self._fill_rows_parallel(len(batch), fill_rows)

        deltas = self._remove_delta_tensor(source_pad, block_pad, psi_pad, source_totals, block_totals) + block_ll
        for (idx, _), delta in zip(indexed, deltas, strict=True):
            scores[idx] = delta

    def _score_block_moves(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        source_totals = np.empty(len(batch), dtype=np.float64)
        target_totals = np.empty(len(batch), dtype=np.float64)
        block_totals = np.empty(len(batch), dtype=np.float64)
        lengths = np.asarray([proposal.block.indices.size for proposal in batch], dtype=np.int64)

        for row, proposal in enumerate(batch):
            source_totals[row] = state.cluster_total(proposal.source_cluster)
            target_totals[row] = state.cluster_total(proposal.target_cluster)
            block_totals[row] = proposal.block.total

        max_len = int(lengths.max(initial=0))
        if max_len == 0:
            deltas = self._remove_totals_only(source_totals, block_totals)
            deltas += self._add_totals_only(target_totals, block_totals)
            for (idx, _), delta in zip(indexed, deltas, strict=True):
                scores[idx] = delta
            return

        psi_pad = np.ones((len(batch), max_len), dtype=np.float64)
        source_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        target_pad = np.zeros((len(batch), max_len), dtype=np.float64)
        block_pad = np.zeros((len(batch), max_len), dtype=np.float64)

        def fill_rows(start: int, end: int) -> None:
            for row in range(start, end):
                proposal = batch[row]
                genes = proposal.block.indices
                counts = proposal.block.values
                length = len(genes)
                if length:
                    psi_pad[row, :length] = self.psi[genes]
                    block_pad[row, :length] = counts
                    source_pad[row, :length] = state.clusters[proposal.source_cluster].get_many(genes)
                    target_pad[row, :length] = state.clusters[proposal.target_cluster].get_many(genes)

        self._fill_rows_parallel(len(batch), fill_rows)

        deltas = self._remove_delta_tensor(source_pad, block_pad, psi_pad, source_totals, block_totals)
        deltas += self._add_delta_tensor(target_pad, block_pad, psi_pad, target_totals, block_totals)
        for (idx, _), delta in zip(indexed, deltas, strict=True):
            scores[idx] = delta

    def _remove_totals_only(self, cluster_totals: np.ndarray, remove_totals: np.ndarray) -> np.ndarray:
        return gammaln_array(cluster_totals + self.psi0) - gammaln_array(cluster_totals - remove_totals + self.psi0)

    def _add_totals_only(self, cluster_totals: np.ndarray, add_totals: np.ndarray) -> np.ndarray:
        return gammaln_array(cluster_totals + self.psi0) - gammaln_array(cluster_totals + add_totals + self.psi0)

    def _remove_delta_tensor(
        self,
        cluster_pad: np.ndarray,
        remove_pad: np.ndarray,
        psi_pad: np.ndarray,
        cluster_totals: np.ndarray,
        remove_totals: np.ndarray,
    ) -> np.ndarray:
        cluster_tensor = torch.as_tensor(cluster_pad, dtype=torch.float64, device=self.device)
        remove_tensor = torch.as_tensor(remove_pad, dtype=torch.float64, device=self.device)
        psi_tensor = torch.as_tensor(psi_pad, dtype=torch.float64, device=self.device)
        cluster_totals_tensor = torch.as_tensor(cluster_totals, dtype=torch.float64, device=self.device)
        remove_totals_tensor = torch.as_tensor(remove_totals, dtype=torch.float64, device=self.device)
        delta = torch.lgamma(cluster_totals_tensor + self.psi0_tensor)
        delta -= torch.lgamma(cluster_totals_tensor - remove_totals_tensor + self.psi0_tensor)
        delta += torch.sum(
            torch.lgamma(cluster_tensor - remove_tensor + psi_tensor) - torch.lgamma(cluster_tensor + psi_tensor),
            dim=1,
        )
        return delta.cpu().numpy()

    def _add_delta_tensor(
        self,
        cluster_pad: np.ndarray,
        add_pad: np.ndarray,
        psi_pad: np.ndarray,
        cluster_totals: np.ndarray,
        add_totals: np.ndarray,
    ) -> np.ndarray:
        cluster_tensor = torch.as_tensor(cluster_pad, dtype=torch.float64, device=self.device)
        add_tensor = torch.as_tensor(add_pad, dtype=torch.float64, device=self.device)
        psi_tensor = torch.as_tensor(psi_pad, dtype=torch.float64, device=self.device)
        cluster_totals_tensor = torch.as_tensor(cluster_totals, dtype=torch.float64, device=self.device)
        add_totals_tensor = torch.as_tensor(add_totals, dtype=torch.float64, device=self.device)
        delta = torch.lgamma(cluster_totals_tensor + self.psi0_tensor)
        delta -= torch.lgamma(cluster_totals_tensor + add_totals_tensor + self.psi0_tensor)
        delta += torch.sum(
            torch.lgamma(cluster_tensor + add_tensor + psi_tensor) - torch.lgamma(cluster_tensor + psi_tensor),
            dim=1,
        )
        return delta.cpu().numpy()


class TorchCudaBackend(TorchDeviceBackend):
    def __init__(
        self,
        psi: np.ndarray,
        state: PartitionState,
        *,
        chunk_size: int = 8192,
    ) -> None:
        super().__init__(psi, state, device="cuda", chunk_size=chunk_size)


class TorchCPUBackend(TorchDeviceBackend):
    def __init__(
        self,
        psi: np.ndarray,
        state: PartitionState,
        *,
        chunk_size: int = 8192,
        num_threads: int | None = None,
    ) -> None:
        super().__init__(psi, state, device="cpu", chunk_size=chunk_size, num_threads=num_threads)

def make_backend(
    backend: str,
    psi: np.ndarray,
    state: PartitionState,
    *,
    num_threads: int | None = None,
) -> CPUBackend:
    backend = backend.lower()
    if backend in {"cpu", "numpy"}:
        return CPUBackend(psi, state, num_threads=num_threads)
    if backend in {"torch-cpu", "cpu-torch", "torch"}:
        return TorchCPUBackend(psi, state, num_threads=num_threads)
    if backend == "cuda":
        return TorchCudaBackend(psi, state)
    if backend == "auto":
        if _torch_device_available("cuda") and _torch_device_supports_float64("cuda"):
            return TorchCudaBackend(psi, state)
        if torch is not None:
            return TorchCPUBackend(psi, state, num_threads=num_threads)
        return CPUBackend(psi, state)
    raise ValueError(f"unknown backend: {backend}")


def gammaln_array(values: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    return gammaln(np.asarray(values, dtype=np.float64))


def _torch_device_available(device: str) -> bool:
    if torch is None:
        return False
    if device == "cpu":
        return True
    if device == "cuda":
        return bool(torch.cuda.is_available())
    raise ValueError(f"unknown torch device: {device}")


def _torch_device_supports_float64(device: str) -> bool:
    if not _torch_device_available(device):
        return False
    try:
        probe = torch.tensor([1.5], dtype=torch.float64, device=device)
        torch.lgamma(probe)
    except Exception:
        return False
    return True


def _require_torch_device(device: str) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if not _torch_device_available(device):
        if device == "cpu":
            raise RuntimeError("CPU torch backend is not available")
        if device == "cuda":
            raise RuntimeError("CUDA is not available")
        raise RuntimeError(f"{device} is not available")
    if not _torch_device_supports_float64(device):
        raise RuntimeError(
            f"{device.upper()} is available but does not support the required float64 lgamma scoring path",
        )
