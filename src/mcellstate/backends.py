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


def _proposal_is_valid(state: PartitionState, proposal: Proposal) -> bool:
    if isinstance(proposal, MergeProposal):
        return (
            int(proposal.cluster_a) in state.clusters
            and int(proposal.cluster_b) in state.clusters
            and int(proposal.cluster_a) != int(proposal.cluster_b)
        )
    if isinstance(proposal, PeelProposal):
        cell = int(proposal.cell)
        return (
            0 <= cell < state.n_cells
            and int(proposal.source_cluster) in state.clusters
            and int(state.z[cell]) == int(proposal.source_cluster)
            and state.cluster_size(int(proposal.source_cluster)) > 1
        )
    if isinstance(proposal, MoveProposal):
        cell = int(proposal.cell)
        return (
            0 <= cell < state.n_cells
            and int(proposal.source_cluster) in state.clusters
            and int(proposal.target_cluster) in state.clusters
            and int(proposal.source_cluster) != int(proposal.target_cluster)
            and int(state.z[cell]) == int(proposal.source_cluster)
        )
    if isinstance(proposal, BlockPeelProposal):
        source_cluster = int(proposal.source_cluster)
        return (
            source_cluster in state.clusters
            and 0 < len(proposal.block.cells) < state.cluster_size(source_cluster)
            and all(
                0 <= int(cell) < state.n_cells
                and int(state.z[int(cell)]) == source_cluster
                for cell in proposal.block.cells
            )
        )
    if isinstance(proposal, BlockMoveProposal):
        source_cluster = int(proposal.source_cluster)
        target_cluster = int(proposal.target_cluster)
        return (
            source_cluster in state.clusters
            and target_cluster in state.clusters
            and source_cluster != target_cluster
            and 0 < len(proposal.block.cells) < state.cluster_size(source_cluster)
            and all(
                0 <= int(cell) < state.n_cells
                and int(state.z[int(cell)]) == source_cluster
                for cell in proposal.block.cells
            )
        )
    return False


class CPUBackend:
    def __init__(
        self, psi: np.ndarray, state: PartitionState, *, num_threads: int | None = None
    ) -> None:
        self.psi = np.asarray(psi, dtype=np.float64)
        self.psi0 = float(self.psi.sum())
        self.num_threads = max(1, int(num_threads) if num_threads is not None else 1)
        self.cell_ll = self._precompute_cell_ll(state)

    def _parallel_ranges(
        self, n_items: int, *, min_items_per_worker: int = 32
    ) -> list[tuple[int, int]]:
        n_items = int(n_items)
        if self.num_threads <= 1 or n_items <= min_items_per_worker:
            return [(0, n_items)] if n_items > 0 else []
        max_workers = min(
            self.num_threads, max(1, n_items // max(1, min_items_per_worker))
        )
        max_workers = max(1, min(max_workers, n_items))
        if max_workers <= 1:
            return [(0, n_items)]
        chunk = (n_items + max_workers - 1) // max_workers
        return [
            (start, min(start + chunk, n_items)) for start in range(0, n_items, chunk)
        ]

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
                values[cell] = cluster_log_likelihood_from_sparse(
                    indices, counts, self.psi, psi0=self.psi0
                )

        self._run_parallel_ranges(ranges, fill_range)
        return values

    def score_batch(
        self, state: PartitionState, proposals: list[Proposal]
    ) -> np.ndarray:
        if not proposals:
            return np.empty(0, dtype=np.float64)
        scores = np.full(len(proposals), -np.inf, dtype=np.float64)
        valid_items = [
            (idx, proposal)
            for idx, proposal in enumerate(proposals)
            if _proposal_is_valid(state, proposal)
        ]
        if not valid_items:
            return scores
        valid_indices = [idx for idx, _ in valid_items]
        valid_proposals = [proposal for _, proposal in valid_items]
        merge_small_ll: dict[int, float] = {}
        merge_cluster_ids = sorted(
            {
                int(cluster_id)
                for proposal in valid_proposals
                if isinstance(proposal, MergeProposal)
                for cluster_id in (proposal.cluster_a, proposal.cluster_b)
            },
        )
        for cluster_id in merge_cluster_ids:
            merge_small_ll[int(cluster_id)] = state.cluster_log_likelihood_cached(
                int(cluster_id), self.psi
            )

        ranges = self._parallel_ranges(len(valid_proposals), min_items_per_worker=64)
        if len(ranges) <= 1:
            for idx, proposal in zip(valid_indices, valid_proposals, strict=True):
                scores[idx] = self._score_single_proposal(
                    state, proposal, merge_small_ll
                )
            return scores

        def score_range(start: int, end: int) -> None:
            for local_idx in range(start, end):
                scores[valid_indices[local_idx]] = self._score_single_proposal(
                    state,
                    valid_proposals[local_idx],
                    merge_small_ll,
                )

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
            delta -= merge_small_ll[
                proposal.cluster_a if small is vector_a else proposal.cluster_b
            ]
            return float(delta)
        elif isinstance(proposal, PeelProposal):
            return float(
                delta_peel_cell(state, self.psi, proposal.cell, proposal.source_cluster)
            )
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
        self.psi0_tensor = torch.tensor(
            self.psi0, dtype=torch.float64, device=self.device
        )
        
        # Initialize GPU-resident state variables
        self.state_version = state.version
        self.psi_tensor = torch.tensor(self.psi, dtype=torch.float64, device=self.device)
        self.cell_totals_tensor = torch.tensor(state.cell_totals, dtype=torch.float64, device=self.device)
        self.X_indptr_tensor = torch.tensor(state.X.indptr, dtype=torch.int64, device=self.device)
        self.X_indices_tensor = torch.tensor(state.X.indices, dtype=torch.int64, device=self.device)
        self.X_data_tensor = torch.tensor(state.X.data, dtype=torch.float64, device=self.device)
        self.cell_ll_tensor = torch.tensor(self.cell_ll, dtype=torch.float64, device=self.device)
        
        capacity = max(1000, state.next_cluster_id + 2000)
        self.cluster_capacity = capacity
        self.cluster_matrix = torch.zeros((capacity, state.n_genes), dtype=torch.float64, device=self.device)
        self.cluster_totals = torch.zeros(capacity, dtype=torch.float64, device=self.device)
        for cid, vector in state.clusters.items():
            genes, counts = vector.sorted_items()
            if len(genes):
                self.cluster_matrix[cid, torch.tensor(genes, dtype=torch.int64, device=self.device)] = torch.tensor(counts, dtype=torch.float64, device=self.device)
            self.cluster_totals[cid] = float(vector.total)

    def _ensure_cluster_capacity(self, next_id: int) -> None:
        if next_id >= self.cluster_capacity:
            new_capacity = next_id + 2000
            new_matrix = torch.zeros((new_capacity, self.cluster_matrix.shape[1]), dtype=torch.float64, device=self.device)
            new_totals = torch.zeros(new_capacity, dtype=torch.float64, device=self.device)
            new_matrix[:self.cluster_capacity] = self.cluster_matrix
            new_totals[:self.cluster_capacity] = self.cluster_totals
            self.cluster_matrix = new_matrix
            self.cluster_totals = new_totals
            self.cluster_capacity = new_capacity

    def score_batch(
        self, state: PartitionState, proposals: list[Proposal]
    ) -> np.ndarray:
        if not proposals:
            return np.empty(0, dtype=np.float64)

        # Automated state synchronization check
        if self.state_version != state.version:
            self._ensure_cluster_capacity(state.next_cluster_id)
            for cid in state.last_touched_clusters:
                if cid in state.active_cluster_ids:
                    vector = state.clusters[cid]
                    genes, counts = vector.sorted_items()
                    self.cluster_matrix[cid].zero_()
                    if len(genes):
                        self.cluster_matrix[cid, torch.tensor(genes, dtype=torch.int64, device=self.device)] = torch.tensor(counts, dtype=torch.float64, device=self.device)
                    self.cluster_totals[cid] = float(vector.total)
                else:
                    self.cluster_matrix[cid].zero_()
                    self.cluster_totals[cid] = 0.0
            self.state_version = state.version

        scores = np.full(len(proposals), -np.inf, dtype=np.float64)
        by_type: dict[str, list[tuple[int, Proposal]]] = defaultdict(list)
        for idx, proposal in enumerate(proposals):
            if not _proposal_is_valid(state, proposal):
                continue
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

    def _chunked(
        self, indexed: list[tuple[int, Proposal]]
    ) -> list[list[tuple[int, Proposal]]]:
        if not indexed:
            return []
        return [
            indexed[start : start + self.chunk_size]
            for start in range(0, len(indexed), self.chunk_size)
        ]

    def _score_peels(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        cell_ids = torch.tensor([proposal.cell for proposal in batch], dtype=torch.int64, device=self.device)
        source_cids = torch.tensor([proposal.source_cluster for proposal in batch], dtype=torch.int64, device=self.device)
        
        starts = self.X_indptr_tensor[cell_ids]
        ends = self.X_indptr_tensor[cell_ids + 1]
        lengths = ends - starts
        max_len = int(lengths.max(initial=0).item())
        
        if max_len == 0:
            source_totals = self.cluster_totals[source_cids]
            cell_totals = self.cell_totals_tensor[cell_ids]
            deltas = torch.lgamma(source_totals + self.psi0_tensor) - torch.lgamma(source_totals - cell_totals + self.psi0_tensor)
            deltas += self.cell_ll_tensor[cell_ids]
            scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()
            return
            
        grid = torch.arange(max_len, device=self.device).expand(len(batch), max_len)
        mask = grid < lengths.unsqueeze(1)
        
        index_tensor = starts.unsqueeze(1) + torch.arange(max_len, device=self.device)
        clamped_indices = torch.where(mask, index_tensor, torch.zeros_like(index_tensor))
        
        genes_2d = torch.gather(self.X_indices_tensor, 0, clamped_indices.view(-1)).view(len(batch), max_len)
        counts_2d = torch.gather(self.X_data_tensor, 0, clamped_indices.view(-1)).view(len(batch), max_len)
        
        genes_2d = torch.where(mask, genes_2d, torch.zeros_like(genes_2d))
        counts_2d = torch.where(mask, counts_2d, torch.zeros_like(counts_2d))
        
        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(len(batch), max_len)
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))
        
        rows = source_cids.unsqueeze(1).expand(len(batch), max_len)
        source_pad = self.cluster_matrix[rows, genes_2d]
        source_pad = torch.where(mask, source_pad, torch.zeros_like(source_pad))
        
        source_totals = self.cluster_totals[source_cids]
        cell_totals = self.cell_totals_tensor[cell_ids]
        
        delta = torch.lgamma(source_totals + self.psi0_tensor)
        delta -= torch.lgamma(source_totals - cell_totals + self.psi0_tensor)
        delta += torch.sum(
            torch.lgamma(source_pad - counts_2d + psi_pad) - torch.lgamma(source_pad + psi_pad),
            dim=1
        )
        
        delta += self.cell_ll_tensor[cell_ids]
        scores[[idx for idx, _ in indexed]] = delta.cpu().numpy()

    def _score_moves(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        cell_ids = torch.tensor([proposal.cell for proposal in batch], dtype=torch.int64, device=self.device)
        source_cids = torch.tensor([proposal.source_cluster for proposal in batch], dtype=torch.int64, device=self.device)
        target_cids = torch.tensor([proposal.target_cluster for proposal in batch], dtype=torch.int64, device=self.device)
        
        starts = self.X_indptr_tensor[cell_ids]
        ends = self.X_indptr_tensor[cell_ids + 1]
        lengths = ends - starts
        max_len = int(lengths.max(initial=0).item())
        
        if max_len == 0:
            source_totals = self.cluster_totals[source_cids]
            target_totals = self.cluster_totals[target_cids]
            cell_totals = self.cell_totals_tensor[cell_ids]
            
            source_delta = torch.lgamma(source_totals + self.psi0_tensor) - torch.lgamma(source_totals - cell_totals + self.psi0_tensor)
            target_delta = torch.lgamma(target_totals + self.psi0_tensor) - torch.lgamma(target_totals + cell_totals + self.psi0_tensor)
            deltas = source_delta + target_delta
            scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()
            return
            
        grid = torch.arange(max_len, device=self.device).expand(len(batch), max_len)
        mask = grid < lengths.unsqueeze(1)
        
        index_tensor = starts.unsqueeze(1) + torch.arange(max_len, device=self.device)
        clamped_indices = torch.where(mask, index_tensor, torch.zeros_like(index_tensor))
        
        genes_2d = torch.gather(self.X_indices_tensor, 0, clamped_indices.view(-1)).view(len(batch), max_len)
        counts_2d = torch.gather(self.X_data_tensor, 0, clamped_indices.view(-1)).view(len(batch), max_len)
        
        genes_2d = torch.where(mask, genes_2d, torch.zeros_like(genes_2d))
        counts_2d = torch.where(mask, counts_2d, torch.zeros_like(counts_2d))
        
        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(len(batch), max_len)
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))
        
        rows_source = source_cids.unsqueeze(1).expand(len(batch), max_len)
        source_pad = self.cluster_matrix[rows_source, genes_2d]
        source_pad = torch.where(mask, source_pad, torch.zeros_like(source_pad))
        
        rows_target = target_cids.unsqueeze(1).expand(len(batch), max_len)
        target_pad = self.cluster_matrix[rows_target, genes_2d]
        target_pad = torch.where(mask, target_pad, torch.zeros_like(target_pad))
        
        source_totals = self.cluster_totals[source_cids]
        target_totals = self.cluster_totals[target_cids]
        cell_totals = self.cell_totals_tensor[cell_ids]
        
        source_delta = torch.lgamma(source_totals + self.psi0_tensor)
        source_delta -= torch.lgamma(source_totals - cell_totals + self.psi0_tensor)
        source_delta += torch.sum(
            torch.lgamma(source_pad - counts_2d + psi_pad) - torch.lgamma(source_pad + psi_pad),
            dim=1
        )
        
        target_delta = torch.lgamma(target_totals + self.psi0_tensor)
        target_delta -= torch.lgamma(target_totals + cell_totals + self.psi0_tensor)
        target_delta += torch.sum(
            torch.lgamma(target_pad + counts_2d + psi_pad) - torch.lgamma(target_pad + psi_pad),
            dim=1
        )
        
        deltas = source_delta + target_delta
        scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()

    def _score_merges(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        small_cids = []
        large_cids = []
        small_ll_arr = np.empty(len(batch), dtype=np.float64)
        small_totals_arr = np.empty(len(batch), dtype=np.float64)
        large_totals_arr = np.empty(len(batch), dtype=np.float64)
        lengths = np.empty(len(batch), dtype=np.int64)
        
        small_vectors = []
        
        for row, proposal in enumerate(batch):
            vector_a = state.clusters[proposal.cluster_a]
            vector_b = state.clusters[proposal.cluster_b]
            if vector_a.nnz <= vector_b.nnz:
                small_cid, large_cid = proposal.cluster_a, proposal.cluster_b
                small, large = vector_a, vector_b
            else:
                small_cid, large_cid = proposal.cluster_b, proposal.cluster_a
                small, large = vector_b, vector_a
            
            small_cids.append(small_cid)
            large_cids.append(large_cid)
            small_vectors.append(small)
            
            small_ll_arr[row] = state.cluster_log_likelihood_cached(small_cid, self.psi)
            small_totals_arr[row] = small.total
            large_totals_arr[row] = large.total
            lengths[row] = small.nnz
            
        max_len = int(lengths.max(initial=0))
        if max_len == 0:
            large_totals = torch.tensor(large_totals_arr, dtype=torch.float64, device=self.device)
            small_totals = torch.tensor(small_totals_arr, dtype=torch.float64, device=self.device)
            small_ll = torch.tensor(small_ll_arr, dtype=torch.float64, device=self.device)
            
            deltas = torch.lgamma(large_totals + self.psi0_tensor) - torch.lgamma(large_totals + small_totals + self.psi0_tensor) - small_ll
            scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()
            return
            
        genes_np = np.zeros((len(batch), max_len), dtype=np.int64)
        small_counts_np = np.zeros((len(batch), max_len), dtype=np.float64)
        mask_np = np.zeros((len(batch), max_len), dtype=bool)
        
        for row, small in enumerate(small_vectors):
            genes, counts = small.sorted_items()
            length = len(genes)
            if length:
                genes_np[row, :length] = genes
                small_counts_np[row, :length] = counts
                mask_np[row, :length] = True
                
        genes_2d = torch.tensor(genes_np, dtype=torch.int64, device=self.device)
        small_pad = torch.tensor(small_counts_np, dtype=torch.float64, device=self.device)
        mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
        
        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(len(batch), max_len)
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))
        
        large_cids_tensor = torch.tensor(large_cids, dtype=torch.int64, device=self.device)
        rows = large_cids_tensor.unsqueeze(1).expand(len(batch), max_len)
        large_pad = self.cluster_matrix[rows, genes_2d]
        large_pad = torch.where(mask, large_pad, torch.zeros_like(large_pad))
        
        large_totals = torch.tensor(large_totals_arr, dtype=torch.float64, device=self.device)
        small_totals = torch.tensor(small_totals_arr, dtype=torch.float64, device=self.device)
        small_ll = torch.tensor(small_ll_arr, dtype=torch.float64, device=self.device)
        
        delta = torch.lgamma(large_totals + self.psi0_tensor)
        delta -= torch.lgamma(large_totals + small_totals + self.psi0_tensor)
        delta += torch.sum(
            torch.lgamma(large_pad + small_pad + psi_pad) - torch.lgamma(large_pad + psi_pad),
            dim=1
        )
        delta -= small_ll
        scores[[idx for idx, _ in indexed]] = delta.cpu().numpy()

    def _score_block_peels(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        block_ll_arr = np.empty(len(batch), dtype=np.float64)
        source_totals_arr = np.empty(len(batch), dtype=np.float64)
        block_totals_arr = np.empty(len(batch), dtype=np.float64)
        lengths = np.asarray([proposal.block.indices.size for proposal in batch], dtype=np.int64)
        
        for row, proposal in enumerate(batch):
            block_ll_arr[row] = cluster_log_likelihood_from_sparse(
                proposal.block.indices,
                proposal.block.values,
                self.psi,
                psi0=self.psi0,
            )
            source_totals_arr[row] = state.cluster_total(proposal.source_cluster)
            block_totals_arr[row] = proposal.block.total
            
        max_len = int(lengths.max(initial=0))
        if max_len == 0:
            source_totals = torch.tensor(source_totals_arr, dtype=torch.float64, device=self.device)
            block_totals = torch.tensor(block_totals_arr, dtype=torch.float64, device=self.device)
            block_ll = torch.tensor(block_ll_arr, dtype=torch.float64, device=self.device)
            
            deltas = torch.lgamma(source_totals + self.psi0_tensor) - torch.lgamma(source_totals - block_totals + self.psi0_tensor) + block_ll
            scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()
            return
            
        genes_np = np.zeros((len(batch), max_len), dtype=np.int64)
        block_counts_np = np.zeros((len(batch), max_len), dtype=np.float64)
        mask_np = np.zeros((len(batch), max_len), dtype=bool)
        
        for row, proposal in enumerate(batch):
            genes = proposal.block.indices
            counts = proposal.block.values
            length = len(genes)
            if length:
                genes_np[row, :length] = genes
                block_counts_np[row, :length] = counts
                mask_np[row, :length] = True
                
        genes_2d = torch.tensor(genes_np, dtype=torch.int64, device=self.device)
        block_pad = torch.tensor(block_counts_np, dtype=torch.float64, device=self.device)
        mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
        
        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(len(batch), max_len)
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))
        
        source_cids = torch.tensor([proposal.source_cluster for proposal in batch], dtype=torch.int64, device=self.device)
        rows = source_cids.unsqueeze(1).expand(len(batch), max_len)
        source_pad = self.cluster_matrix[rows, genes_2d]
        source_pad = torch.where(mask, source_pad, torch.zeros_like(source_pad))
        
        source_totals = torch.tensor(source_totals_arr, dtype=torch.float64, device=self.device)
        block_totals = torch.tensor(block_totals_arr, dtype=torch.float64, device=self.device)
        block_ll = torch.tensor(block_ll_arr, dtype=torch.float64, device=self.device)
        
        delta = torch.lgamma(source_totals + self.psi0_tensor)
        delta -= torch.lgamma(source_totals - block_totals + self.psi0_tensor)
        delta += torch.sum(
            torch.lgamma(source_pad - block_pad + psi_pad) - torch.lgamma(source_pad + psi_pad),
            dim=1
        )
        delta += block_ll
        scores[[idx for idx, _ in indexed]] = delta.cpu().numpy()

    def _score_block_moves(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        source_totals_arr = np.empty(len(batch), dtype=np.float64)
        target_totals_arr = np.empty(len(batch), dtype=np.float64)
        block_totals_arr = np.empty(len(batch), dtype=np.float64)
        lengths = np.asarray([proposal.block.indices.size for proposal in batch], dtype=np.int64)
        
        for row, proposal in enumerate(batch):
            source_totals_arr[row] = state.cluster_total(proposal.source_cluster)
            target_totals_arr[row] = state.cluster_total(proposal.target_cluster)
            block_totals_arr[row] = proposal.block.total
            
        max_len = int(lengths.max(initial=0))
        if max_len == 0:
            source_totals = torch.tensor(source_totals_arr, dtype=torch.float64, device=self.device)
            target_totals = torch.tensor(target_totals_arr, dtype=torch.float64, device=self.device)
            block_totals = torch.tensor(block_totals_arr, dtype=torch.float64, device=self.device)
            
            deltas = torch.lgamma(source_totals + self.psi0_tensor) - torch.lgamma(source_totals - block_totals + self.psi0_tensor)
            deltas += torch.lgamma(target_totals + self.psi0_tensor) - torch.lgamma(target_totals + block_totals + self.psi0_tensor)
            scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()
            return
            
        genes_np = np.zeros((len(batch), max_len), dtype=np.int64)
        block_counts_np = np.zeros((len(batch), max_len), dtype=np.float64)
        mask_np = np.zeros((len(batch), max_len), dtype=bool)
        
        for row, proposal in enumerate(batch):
            genes = proposal.block.indices
            counts = proposal.block.values
            length = len(genes)
            if length:
                genes_np[row, :length] = genes
                block_counts_np[row, :length] = counts
                mask_np[row, :length] = True
                
        genes_2d = torch.tensor(genes_np, dtype=torch.int64, device=self.device)
        block_pad = torch.tensor(block_counts_np, dtype=torch.float64, device=self.device)
        mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
        
        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(len(batch), max_len)
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))
        
        source_cids = torch.tensor([proposal.source_cluster for proposal in batch], dtype=torch.int64, device=self.device)
        rows_source = source_cids.unsqueeze(1).expand(len(batch), max_len)
        source_pad = self.cluster_matrix[rows_source, genes_2d]
        source_pad = torch.where(mask, source_pad, torch.zeros_like(source_pad))
        
        target_cids = torch.tensor([proposal.target_cluster for proposal in batch], dtype=torch.int64, device=self.device)
        rows_target = target_cids.unsqueeze(1).expand(len(batch), max_len)
        target_pad = self.cluster_matrix[rows_target, genes_2d]
        target_pad = torch.where(mask, target_pad, torch.zeros_like(target_pad))
        
        source_totals = torch.tensor(source_totals_arr, dtype=torch.float64, device=self.device)
        target_totals = torch.tensor(target_totals_arr, dtype=torch.float64, device=self.device)
        block_totals = torch.tensor(block_totals_arr, dtype=torch.float64, device=self.device)
        
        source_delta = torch.lgamma(source_totals + self.psi0_tensor)
        source_delta -= torch.lgamma(source_totals - block_totals + self.psi0_tensor)
        source_delta += torch.sum(
            torch.lgamma(source_pad - block_pad + psi_pad) - torch.lgamma(source_pad + psi_pad),
            dim=1
        )
        
        target_delta = torch.lgamma(target_totals + self.psi0_tensor)
        target_delta -= torch.lgamma(target_totals + block_totals + self.psi0_tensor)
        target_delta += torch.sum(
            torch.lgamma(target_pad + block_pad + psi_pad) - torch.lgamma(target_pad + psi_pad),
            dim=1
        )
        
        deltas = source_delta + target_delta
        scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()


class TorchCudaBackend(TorchDeviceBackend):
    def __init__(
        self,
        psi: np.ndarray,
        state: PartitionState,
        *,
        chunk_size: int = 8192,
        num_threads: int | None = None,
    ) -> None:
        super().__init__(
            psi,
            state,
            device="cuda",
            chunk_size=chunk_size,
            num_threads=num_threads,
        )


class TorchCPUBackend(TorchDeviceBackend):
    def __init__(
        self,
        psi: np.ndarray,
        state: PartitionState,
        *,
        chunk_size: int = 8192,
        num_threads: int | None = None,
    ) -> None:
        super().__init__(
            psi, state, device="cpu", chunk_size=chunk_size, num_threads=num_threads
        )


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
        return TorchCudaBackend(psi, state, num_threads=num_threads)
    if backend == "auto":
        if _torch_device_available("cuda") and _torch_device_supports_float64("cuda"):
            return TorchCudaBackend(psi, state, num_threads=num_threads)
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
