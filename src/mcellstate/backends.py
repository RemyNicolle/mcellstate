from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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


@dataclass
class _DeviceStateCache:
    version: int
    cluster_capacity: int
    active_cluster_ids_tensor: torch.Tensor
    active_cluster_lookup_tensor: torch.Tensor
    active_cluster_count: int
    cluster_sizes_tensor: torch.Tensor
    z_tensor: torch.Tensor
    cluster_matrix: torch.Tensor
    cluster_totals: torch.Tensor


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
        chunk_size: int | None = 8192,
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
        self.chunk_size = (
            None if chunk_size is None or int(chunk_size) <= 0 else int(chunk_size)
        )
        self._cuda_target_fraction = 0.0125
        self._cuda_bytes_per_entry = 64.0
        self._cuda_min_chunk_size = 512
        self._cuda_max_chunk_size = 65_536
        self.psi0_tensor = torch.tensor(
            self.psi0, dtype=torch.float64, device=self.device
        )

        self.psi_tensor = torch.tensor(
            self.psi, dtype=torch.float64, device=self.device
        )
        self.cell_totals_tensor = torch.tensor(
            state.cell_totals, dtype=torch.float64, device=self.device
        )
        self.X_indptr_tensor = torch.tensor(
            state.X.indptr, dtype=torch.int64, device=self.device
        )
        self.X_indices_tensor = torch.tensor(
            state.X.indices, dtype=torch.int64, device=self.device
        )
        self.X_data_tensor = torch.tensor(
            state.X.data, dtype=torch.float64, device=self.device
        )
        self.cell_ll_tensor = torch.tensor(
            self.cell_ll, dtype=torch.float64, device=self.device
        )

        self._state_cache = self._build_state_cache(state)
        self.state_version = self._state_cache.version
        self.cluster_capacity = self._state_cache.cluster_capacity
        self.cluster_matrix = self._state_cache.cluster_matrix
        self.cluster_totals = self._state_cache.cluster_totals
        self.z_tensor = self._state_cache.z_tensor
        self.cluster_sizes_tensor = self._state_cache.cluster_sizes_tensor
        self.active_cluster_ids_tensor = self._state_cache.active_cluster_ids_tensor
        self.active_cluster_lookup_tensor = (
            self._state_cache.active_cluster_lookup_tensor
        )

    def _build_state_cache(self, state: PartitionState) -> _DeviceStateCache:
        cluster_capacity = max(1000, state.next_cluster_id + 2000)
        active_cluster_ids_np = state.active_cluster_array()
        active_cluster_ids_tensor = torch.full(
            (cluster_capacity,), -1, dtype=torch.int64, device=self.device
        )
        active_cluster_lookup_tensor = torch.full(
            (cluster_capacity,), -1, dtype=torch.int64, device=self.device
        )
        active_cluster_count = int(active_cluster_ids_np.size)
        if active_cluster_count:
            active_tensor = torch.tensor(
                active_cluster_ids_np, dtype=torch.int64, device=self.device
            )
            active_cluster_ids_tensor[:active_cluster_count] = active_tensor
            active_cluster_lookup_tensor[active_tensor] = torch.arange(
                active_cluster_count, dtype=torch.int64, device=self.device
            )

        cluster_sizes_tensor = torch.zeros(
            cluster_capacity, dtype=torch.int64, device=self.device
        )
        z_tensor = torch.tensor(state.z, dtype=torch.int64, device=self.device)
        cluster_matrix = torch.zeros(
            (cluster_capacity, state.n_genes), dtype=torch.float64, device=self.device
        )
        cluster_totals = torch.zeros(
            cluster_capacity, dtype=torch.float64, device=self.device
        )
        for cid, vector in state.clusters.items():
            genes, counts = vector.sorted_items()
            cluster_sizes_tensor[cid] = len(state.cells_by_cluster[cid])
            if len(genes):
                cluster_matrix[
                    cid, torch.tensor(genes, dtype=torch.int64, device=self.device)
                ] = torch.tensor(counts, dtype=torch.float64, device=self.device)
            cluster_totals[cid] = float(vector.total)

        return _DeviceStateCache(
            version=state.version,
            cluster_capacity=cluster_capacity,
            active_cluster_ids_tensor=active_cluster_ids_tensor,
            active_cluster_lookup_tensor=active_cluster_lookup_tensor,
            active_cluster_count=active_cluster_count,
            cluster_sizes_tensor=cluster_sizes_tensor,
            z_tensor=z_tensor,
            cluster_matrix=cluster_matrix,
            cluster_totals=cluster_totals,
        )

    def _ensure_cluster_capacity(self, next_id: int) -> None:
        cache = self._state_cache
        if next_id < cache.cluster_capacity:
            return
        new_capacity = next_id + 2000
        new_active_ids = torch.full(
            (new_capacity,), -1, dtype=torch.int64, device=self.device
        )
        new_active_ids[: cache.active_cluster_count] = cache.active_cluster_ids_tensor[
            : cache.active_cluster_count
        ]
        new_lookup = torch.full(
            (new_capacity,), -1, dtype=torch.int64, device=self.device
        )
        new_lookup[
            : cache.active_cluster_lookup_tensor.shape[0]
        ] = cache.active_cluster_lookup_tensor
        new_sizes = torch.zeros(new_capacity, dtype=torch.int64, device=self.device)
        new_sizes[: cache.cluster_sizes_tensor.shape[0]] = cache.cluster_sizes_tensor
        new_matrix = torch.zeros(
            (new_capacity, cache.cluster_matrix.shape[1]),
            dtype=torch.float64,
            device=self.device,
        )
        new_matrix[: cache.cluster_matrix.shape[0]] = cache.cluster_matrix
        new_totals = torch.zeros(new_capacity, dtype=torch.float64, device=self.device)
        new_totals[: cache.cluster_totals.shape[0]] = cache.cluster_totals

        cache.cluster_capacity = new_capacity
        cache.active_cluster_ids_tensor = new_active_ids
        cache.active_cluster_lookup_tensor = new_lookup
        cache.cluster_sizes_tensor = new_sizes
        cache.cluster_matrix = new_matrix
        cache.cluster_totals = new_totals

        self.cluster_capacity = new_capacity
        self.active_cluster_ids_tensor = new_active_ids
        self.active_cluster_lookup_tensor = new_lookup
        self.cluster_sizes_tensor = new_sizes
        self.cluster_matrix = new_matrix
        self.cluster_totals = new_totals

    def _add_active_cluster(self, cluster_id: int) -> None:
        cluster_id = int(cluster_id)
        cache = self._state_cache
        if cluster_id >= cache.cluster_capacity:
            self._ensure_cluster_capacity(cluster_id + 1)
            cache = self._state_cache
        if int(cache.active_cluster_lookup_tensor[cluster_id].item()) >= 0:
            return
        pos = int(cache.active_cluster_count)
        if pos >= cache.active_cluster_ids_tensor.shape[0]:
            self._ensure_cluster_capacity(
                max(cluster_id + 1, cache.cluster_capacity + 1)
            )
            cache = self._state_cache
        cache.active_cluster_ids_tensor[pos] = cluster_id
        cache.active_cluster_lookup_tensor[cluster_id] = pos
        cache.active_cluster_count = pos + 1

    def _remove_active_cluster(self, cluster_id: int) -> None:
        cluster_id = int(cluster_id)
        cache = self._state_cache
        if cluster_id >= cache.active_cluster_lookup_tensor.shape[0]:
            return
        pos = int(cache.active_cluster_lookup_tensor[cluster_id].item())
        if pos < 0:
            return
        last_pos = int(cache.active_cluster_count) - 1
        last_cluster = int(cache.active_cluster_ids_tensor[last_pos].item())
        if pos != last_pos:
            cache.active_cluster_ids_tensor[pos] = last_cluster
            cache.active_cluster_lookup_tensor[last_cluster] = pos
        cache.active_cluster_ids_tensor[last_pos] = -1
        cache.active_cluster_lookup_tensor[cluster_id] = -1
        cache.active_cluster_count = last_pos

    def _sync_state_cache(self, state: PartitionState) -> None:
        if self.state_version == state.version:
            return
        self._ensure_cluster_capacity(state.next_cluster_id)
        touched_clusters = {int(cid) for cid in state.last_touched_clusters}
        for cid in touched_clusters:
            if cid in state.active_cluster_ids:
                self._add_active_cluster(cid)
            else:
                self._remove_active_cluster(cid)

            if cid in state.active_cluster_ids:
                vector = state.clusters[cid]
                genes, counts = vector.sorted_items()
                self.cluster_matrix[cid].zero_()
                if len(genes):
                    self.cluster_matrix[
                        cid, torch.tensor(genes, dtype=torch.int64, device=self.device)
                    ] = torch.tensor(counts, dtype=torch.float64, device=self.device)
                self.cluster_totals[cid] = float(vector.total)
                self.cluster_sizes_tensor[cid] = int(len(state.cells_by_cluster[cid]))
                cells = torch.tensor(
                    state.cells_by_cluster[cid].cells,
                    dtype=torch.int64,
                    device=self.device,
                )
                self.z_tensor[cells] = int(cid)
            else:
                self.cluster_matrix[cid].zero_()
                self.cluster_totals[cid] = 0.0
                self.cluster_sizes_tensor[cid] = 0
        self.state_version = state.version
        self._state_cache.version = state.version

    def score_batch(
        self, state: PartitionState, proposals: list[Proposal]
    ) -> np.ndarray:
        if not proposals:
            return np.empty(0, dtype=np.float64)

        self._sync_state_cache(state)

        scores = np.full(len(proposals), -np.inf, dtype=np.float64)
        by_type: dict[str, list[tuple[int, Proposal]]] = defaultdict(list)
        for idx, proposal in enumerate(proposals):
            if not _proposal_is_valid(state, proposal):
                continue
            by_type[proposal.kind].append((idx, proposal))

        for chunk in self._chunk_proposals(state, by_type.get("peel", [])):
            self._score_peels(state, chunk, scores)
        for chunk in self._chunk_proposals(state, by_type.get("move", [])):
            self._score_moves(state, chunk, scores)
        for chunk in self._chunk_proposals(state, by_type.get("block_peel", [])):
            self._score_block_peels(state, chunk, scores)
        for chunk in self._chunk_proposals(state, by_type.get("block_move", [])):
            self._score_block_moves(state, chunk, scores)
        for chunk in self._chunk_proposals(state, by_type.get("merge", [])):
            self._score_merges(state, chunk, scores)
        return scores

    def sample_random_proposals(
        self,
        state: PartitionState,
        n_proposals: int,
        *,
        family_weights: np.ndarray,
        max_unique_proposals: int | None = None,
        seed: int | None = None,
    ) -> list[Proposal]:
        if n_proposals <= 0:
            return []
        if torch is None:
            return []

        weights = np.asarray(family_weights, dtype=np.float64)
        if weights.shape != (5,):
            raise ValueError("family_weights must have five entries")
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("family weights must be non-negative with positive mass")
        weights = weights / float(weights.sum())

        rng = np.random.default_rng(seed)
        draw_seed = int(rng.integers(np.iinfo(np.int64).max))
        generator = torch.Generator(device=str(self.device))
        generator.manual_seed(draw_seed)

        counts = np.asarray(rng.multinomial(int(n_proposals), weights), dtype=np.int64)
        merge_count, peel_count, move_count = (
            int(counts[0]),
            int(counts[1]),
            int(counts[2]),
        )

        self._sync_state_cache(state)
        cache = self._state_cache
        if cache.active_cluster_count == 0:
            return []

        active_clusters = cache.active_cluster_ids_tensor[: cache.active_cluster_count]
        n_active = int(cache.active_cluster_count)
        cluster_lookup = cache.active_cluster_lookup_tensor
        cluster_sizes = cache.cluster_sizes_tensor
        z_tensor = cache.z_tensor

        records: list[torch.Tensor] = []

        if merge_count > 0 and n_active >= 2:
            first = torch.randint(
                n_active, (merge_count,), generator=generator, device=self.device
            )
            second = torch.randint(
                n_active - 1,
                (merge_count,),
                generator=generator,
                device=self.device,
            )
            second = second + (second >= first).to(second.dtype)
            cluster_a = torch.minimum(active_clusters[first], active_clusters[second])
            cluster_b = torch.maximum(active_clusters[first], active_clusters[second])
            kind = torch.zeros_like(cluster_a)
            sentinel = torch.full_like(cluster_a, -1)
            records.append(torch.stack((kind, cluster_a, cluster_b, sentinel), dim=1))

        if peel_count > 0 and state.n_cells > 0:
            draws = max(peel_count * 2, peel_count + 32)
            cell_ids = torch.randint(
                state.n_cells, (draws,), generator=generator, device=self.device
            )
            source = z_tensor[cell_ids]
            valid = cluster_sizes[source] > 1
            if torch.any(valid):
                cell_ids = cell_ids[valid]
                source = source[valid]
                if cell_ids.numel() > peel_count:
                    perm = torch.randperm(
                        cell_ids.numel(), generator=generator, device=self.device
                    )[:peel_count]
                    cell_ids = cell_ids.index_select(0, perm)
                    source = source.index_select(0, perm)
                kind = torch.ones_like(cell_ids)
                sentinel = torch.full_like(cell_ids, -1)
                records.append(torch.stack((kind, cell_ids, source, sentinel), dim=1))

        if move_count > 0 and n_active >= 2 and state.n_cells > 0:
            cell_ids = torch.randint(
                state.n_cells, (move_count,), generator=generator, device=self.device
            )
            source = z_tensor[cell_ids]
            source_pos = cluster_lookup[source]
            valid = source_pos >= 0
            if torch.any(valid):
                cell_ids = cell_ids[valid]
                source = source[valid]
                source_pos = source_pos[valid]
                target_pos = torch.randint(
                    n_active - 1,
                    (cell_ids.numel(),),
                    generator=generator,
                    device=self.device,
                )
                target_pos = target_pos + (target_pos >= source_pos).to(
                    target_pos.dtype
                )
                target = active_clusters[target_pos]
                kind = torch.full_like(cell_ids, 2)
                records.append(torch.stack((kind, cell_ids, source, target), dim=1))

        if not records:
            return []

        merged = torch.cat(records, dim=0)
        if merged.numel() == 0:
            return []
        merged = torch.unique(merged, dim=0)
        if (
            max_unique_proposals is not None
            and int(max_unique_proposals) > 0
            and merged.size(0) > int(max_unique_proposals)
        ):
            keep = torch.randperm(
                merged.size(0), generator=generator, device=self.device
            )[: int(max_unique_proposals)]
            merged = merged.index_select(0, keep)

        proposals: list[Proposal] = []
        for row in merged.to("cpu").tolist():
            kind, a, b, c = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            if kind == 0:
                proposals.append(MergeProposal(cluster_a=a, cluster_b=b))
            elif kind == 1:
                proposals.append(PeelProposal(cell=a, source_cluster=b))
            elif kind == 2:
                proposals.append(
                    MoveProposal(cell=a, source_cluster=b, target_cluster=c)
                )
        return proposals

    def sample_guided_merge_pairs(
        self,
        state: PartitionState,
        n_pairs: int,
        *,
        epsilon_uniform: float = 0.05,
        include_cached_pairs: bool = True,
        cached_pairs: list[tuple[int, int]] | None = None,
        max_unique_pairs: int | None = None,
        seed: int | None = None,
    ) -> list[MergeProposal]:
        n_pairs = max(0, int(n_pairs))
        if n_pairs <= 0 or torch is None:
            return []

        self._sync_state_cache(state)
        cache = self._state_cache
        n_active = int(cache.active_cluster_count)
        if n_active < 2:
            return []

        epsilon_uniform = max(0.0, min(1.0, float(epsilon_uniform)))
        rng = np.random.default_rng(seed)
        draw_seed = int(rng.integers(np.iinfo(np.int64).max))
        generator = torch.Generator(device=str(self.device))
        generator.manual_seed(draw_seed)

        active_clusters = cache.active_cluster_ids_tensor[: cache.active_cluster_count]
        active_lookup = cache.active_cluster_lookup_tensor
        active_matrix = self.cluster_matrix.index_select(0, active_clusters)
        presence = active_matrix > 0
        proposals: dict[tuple[int, int], MergeProposal] = {}

        if include_cached_pairs and cached_pairs:
            cached_budget = min(len(cached_pairs), max(1, n_pairs // 4))
            for cluster_a, cluster_b in cached_pairs[:cached_budget]:
                cluster_a = int(cluster_a)
                cluster_b = int(cluster_b)
                if cluster_a == cluster_b:
                    continue
                if (
                    cluster_a < 0
                    or cluster_b < 0
                    or cluster_a >= active_lookup.shape[0]
                    or cluster_b >= active_lookup.shape[0]
                ):
                    continue
                if (
                    int(active_lookup[cluster_a].item()) < 0
                    or int(active_lookup[cluster_b].item()) < 0
                ):
                    continue
                pair = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
                proposals.setdefault(
                    pair, MergeProposal(cluster_a=pair[0], cluster_b=pair[1])
                )
                if len(proposals) >= n_pairs:
                    return list(proposals.values())

        member_counts = presence.sum(dim=0)
        valid_gene_ids = torch.nonzero(member_counts >= 2, as_tuple=False).flatten()
        gene_weights: torch.Tensor | None = None
        if valid_gene_ids.numel() > 0:
            counts = member_counts.index_select(0, valid_gene_ids).to(torch.float64)
            pair_counts = counts * (counts - 1.0) * 0.5
            idf = torch.log((float(n_active) + 1.0) / (counts + 1.0))
            gene_weights = idf * pair_counts
            positive = gene_weights > 0
            valid_gene_ids = valid_gene_ids[positive]
            gene_weights = gene_weights[positive]
            if gene_weights.numel() > 0:
                gene_weights = gene_weights / gene_weights.sum()
            else:
                gene_weights = None

        block_size = 1024
        attempts = 0
        max_attempts = max(4, min(16, n_pairs))
        while len(proposals) < n_pairs and attempts < max_attempts:
            attempts += 1
            remaining = n_pairs - len(proposals)
            draw_count = max(remaining * 2, remaining + 32)

            pair_batches: list[torch.Tensor] = []
            if gene_weights is None:
                uniform_count = draw_count
                guided_count = 0
            else:
                uniform_mask = (
                    torch.rand(draw_count, generator=generator, device=self.device)
                    < epsilon_uniform
                )
                uniform_count = int(uniform_mask.sum().item())
                guided_count = int(draw_count - uniform_count)

            if uniform_count > 0:
                first = torch.randint(
                    n_active,
                    (uniform_count,),
                    generator=generator,
                    device=self.device,
                )
                second = torch.randint(
                    n_active - 1,
                    (uniform_count,),
                    generator=generator,
                    device=self.device,
                )
                second = second + (second >= first).to(second.dtype)
                uniform_a = torch.minimum(
                    active_clusters[first], active_clusters[second]
                )
                uniform_b = torch.maximum(
                    active_clusters[first], active_clusters[second]
                )
                pair_batches.append(torch.stack((uniform_a, uniform_b), dim=1))

            if guided_count > 0 and gene_weights is not None:
                sampled = torch.multinomial(
                    gene_weights,
                    num_samples=guided_count,
                    replacement=True,
                    generator=generator,
                )
                sampled_genes = valid_gene_ids.index_select(0, sampled)
                guided_batches: list[torch.Tensor] = []
                for start in range(0, guided_count, block_size):
                    genes_block = sampled_genes[start : start + block_size]
                    gene_presence = presence.index_select(1, genes_block)
                    random_scores = torch.rand(
                        gene_presence.shape,
                        generator=generator,
                        device=self.device,
                    )
                    random_scores = random_scores.masked_fill(~gene_presence, 2.0)
                    chosen = torch.topk(
                        random_scores,
                        k=2,
                        dim=0,
                        largest=False,
                    ).indices
                    guided_a = active_clusters.index_select(0, chosen[0])
                    guided_b = active_clusters.index_select(0, chosen[1])
                    guided_batches.append(
                        torch.stack(
                            (
                                torch.minimum(guided_a, guided_b),
                                torch.maximum(guided_a, guided_b),
                            ),
                            dim=1,
                        )
                    )
                if guided_batches:
                    pair_batches.append(torch.cat(guided_batches, dim=0))

            if not pair_batches:
                break
            merged = torch.cat(pair_batches, dim=0)
            if merged.numel() == 0:
                break
            valid_pairs = merged[:, 0] != merged[:, 1]
            merged = merged[valid_pairs]
            if merged.numel() == 0:
                continue
            merged = torch.unique(merged, dim=0)

            pair_cap = max_unique_pairs if max_unique_pairs is not None else remaining
            pair_cap = max(1, int(pair_cap))
            if merged.size(0) > pair_cap:
                keep = torch.randperm(
                    merged.size(0), generator=generator, device=self.device
                )[:pair_cap]
                merged = merged.index_select(0, keep)

            for cluster_a, cluster_b in merged.to("cpu").tolist():
                pair = (int(cluster_a), int(cluster_b))
                proposals.setdefault(
                    pair, MergeProposal(cluster_a=pair[0], cluster_b=pair[1])
                )
                if len(proposals) >= n_pairs:
                    break

        return list(proposals.values())[:n_pairs]

    def _rank_targets_for_cells_chunk(
        self,
        state: PartitionState,
        cell_ids: torch.Tensor,
        *,
        limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._sync_state_cache(state)
        cache = self._state_cache
        if cell_ids.numel() == 0:
            empty = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_2d = torch.empty((0, 0), dtype=torch.int64, device=self.device)
            empty_mask = torch.empty((0, 0), dtype=torch.bool, device=self.device)
            return empty, empty_2d, empty_mask

        n_active = int(cache.active_cluster_count)
        limit = min(max(1, int(limit)), max(1, n_active - 1))
        active_clusters = cache.active_cluster_ids_tensor[: cache.active_cluster_count]
        active_matrix = self.cluster_matrix.index_select(0, active_clusters)
        member_counts = (active_matrix > 0).sum(dim=0)

        source_cids = self.z_tensor[cell_ids]
        starts = self.X_indptr_tensor[cell_ids]
        ends = self.X_indptr_tensor[cell_ids + 1]
        lengths = ends - starts
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        if max_len == 0:
            top_clusters = active_clusters.unsqueeze(0).expand(cell_ids.numel(), -1)
            if top_clusters.size(1) > limit:
                top_clusters = top_clusters[:, :limit]
            valid = top_clusters != source_cids.unsqueeze(1)
            return source_cids, top_clusters, valid

        grid = torch.arange(max_len, device=self.device).expand(
            cell_ids.numel(), max_len
        )
        mask = grid < lengths.unsqueeze(1)

        index_tensor = starts.unsqueeze(1) + torch.arange(max_len, device=self.device)
        clamped_indices = torch.where(
            mask, index_tensor, torch.zeros_like(index_tensor)
        )
        genes_2d = torch.gather(
            self.X_indices_tensor, 0, clamped_indices.view(-1)
        ).view(cell_ids.numel(), max_len)
        counts_2d = torch.gather(self.X_data_tensor, 0, clamped_indices.view(-1)).view(
            cell_ids.numel(), max_len
        )

        genes_2d = torch.where(mask, genes_2d, torch.zeros_like(genes_2d))
        counts_2d = torch.where(mask, counts_2d, torch.zeros_like(counts_2d))

        idf_pad = torch.log(
            (float(n_active) + 1.0)
            / (
                member_counts.index_select(0, genes_2d.view(-1)).view(
                    cell_ids.numel(), max_len
                )
                + 1.0
            )
        )
        idf_pad = torch.where(mask, idf_pad, torch.zeros_like(idf_pad))

        target_counts = active_matrix.index_select(1, genes_2d.view(-1)).view(
            n_active, cell_ids.numel(), max_len
        )
        target_counts = target_counts.permute(1, 0, 2).contiguous()

        scores = torch.sum(
            torch.minimum(counts_2d.unsqueeze(1), target_counts) * idf_pad.unsqueeze(1),
            dim=2,
        )
        source_pos = self.active_cluster_lookup_tensor[source_cids]
        scores[
            torch.arange(cell_ids.numel(), device=self.device), source_pos
        ] = -torch.inf

        top_scores, top_idx = torch.topk(scores, k=limit, dim=1)
        top_clusters = active_clusters.index_select(0, top_idx.view(-1)).view(
            cell_ids.numel(), limit
        )
        valid = torch.isfinite(top_scores) & (top_clusters != source_cids.unsqueeze(1))
        return source_cids, top_clusters, valid

    def rank_target_clusters(
        self,
        state: PartitionState,
        source_cluster: int,
        genes: np.ndarray,
        values: np.ndarray,
        *,
        limit: int | None = None,
        candidate_targets: np.ndarray | None = None,
    ) -> list[int]:
        self._sync_state_cache(state)
        cache = self._state_cache
        n_active = int(cache.active_cluster_count)
        if n_active < 2:
            return []

        genes = np.asarray(genes, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if genes.size == 0 or values.size == 0:
            return []

        active_clusters = cache.active_cluster_ids_tensor[: cache.active_cluster_count]
        if candidate_targets is not None and np.asarray(candidate_targets).size:
            candidate_targets = np.asarray(candidate_targets, dtype=np.int64)
            valid_targets = [
                int(cluster_id)
                for cluster_id in candidate_targets.tolist()
                if 0 <= int(cluster_id) < cache.active_cluster_lookup_tensor.shape[0]
                and int(cache.active_cluster_lookup_tensor[int(cluster_id)].item()) >= 0
            ]
            if not valid_targets:
                return []
            target_clusters = torch.tensor(
                valid_targets, dtype=torch.int64, device=self.device
            )
        else:
            target_clusters = active_clusters

        target_clusters = target_clusters[target_clusters != int(source_cluster)]
        if target_clusters.numel() == 0:
            return []

        active_matrix = self.cluster_matrix.index_select(0, active_clusters)
        member_counts = (active_matrix > 0).sum(dim=0)
        gene_tensor = torch.tensor(genes, dtype=torch.int64, device=self.device)
        value_tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        idf = torch.log(
            (float(n_active) + 1.0) / (member_counts.index_select(0, gene_tensor) + 1.0)
        )
        target_matrix = self.cluster_matrix.index_select(0, target_clusters)
        target_counts = target_matrix.index_select(1, gene_tensor)
        scores = torch.sum(
            torch.minimum(value_tensor.unsqueeze(0), target_counts) * idf.unsqueeze(0),
            dim=1,
        )
        keep = torch.isfinite(scores)
        if not torch.any(keep):
            return []
        target_clusters = target_clusters[keep]
        scores = scores[keep]
        if target_clusters.numel() == 0:
            return []
        order = torch.argsort(scores, descending=True, stable=True)
        ranked = target_clusters.index_select(0, order)
        if limit is not None and int(limit) > 0 and ranked.numel() > int(limit):
            ranked = ranked[: int(limit)]
        return [int(cluster_id) for cluster_id in ranked.to("cpu").tolist()]

    def rank_targets_for_cells(
        self,
        state: PartitionState,
        cell_ids: np.ndarray,
        *,
        limit: int,
        uniform_targets: int = 0,
        seed: int | None = None,
    ) -> dict[int, list[int]]:
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        if cell_ids.size == 0:
            return {}
        self._sync_state_cache(state)
        cache = self._state_cache
        if int(cache.active_cluster_count) < 2:
            return {}

        rng = np.random.default_rng(seed)
        active_clusters = cache.active_cluster_ids_tensor[: cache.active_cluster_count]
        active_np = active_clusters.to("cpu").numpy()
        chunk_size = min(max(32, int(self._cuda_min_chunk_size // 8)), cell_ids.size)
        ranked: dict[int, list[int]] = {}
        for start in range(0, cell_ids.size, chunk_size):
            chunk_np = cell_ids[start : start + chunk_size]
            chunk_tensor = torch.tensor(chunk_np, dtype=torch.int64, device=self.device)
            source_cids, top_clusters, valid = self._rank_targets_for_cells_chunk(
                state,
                chunk_tensor,
                limit=max(1, int(limit)),
            )
            source_np = source_cids.to("cpu").numpy()
            top_np = top_clusters.to("cpu").numpy()
            valid_np = valid.to("cpu").numpy()
            for row, cell in enumerate(chunk_np.tolist()):
                targets = [
                    int(cluster_id)
                    for cluster_id, keep in zip(
                        top_np[row].tolist(), valid_np[row].tolist(), strict=True
                    )
                    if keep and int(cluster_id) != int(source_np[row])
                ]
                if uniform_targets > 0:
                    uniform_pool = active_np[active_np != int(source_np[row])]
                    if uniform_pool.size:
                        take = min(int(uniform_targets), int(uniform_pool.size))
                        extra = rng.choice(uniform_pool, size=take, replace=False)
                        targets.extend(int(cluster_id) for cluster_id in extra.tolist())
                deduped = [
                    int(cluster_id)
                    for cluster_id in dict.fromkeys(targets)
                    if int(cluster_id) != int(source_np[row])
                ]
                if deduped:
                    ranked[int(cell)] = deduped
        return ranked

    def sample_guided_move_proposals(
        self,
        state: PartitionState,
        n_proposals: int,
        *,
        uniform_prob: float = 0.05,
        limit: int = 16,
        max_unique_proposals: int | None = None,
        seed: int | None = None,
    ) -> list[MoveProposal]:
        n_proposals = max(0, int(n_proposals))
        if n_proposals <= 0 or torch is None or state.n_cells <= 0:
            return []

        self._sync_state_cache(state)
        cache = self._state_cache
        if int(cache.active_cluster_count) < 2:
            return []

        uniform_prob = max(0.0, min(1.0, float(uniform_prob)))
        rng = np.random.default_rng(seed)
        draw_seed = int(rng.integers(np.iinfo(np.int64).max))
        generator = torch.Generator(device=str(self.device))
        generator.manual_seed(draw_seed)

        active_clusters = cache.active_cluster_ids_tensor[: cache.active_cluster_count]
        active_np = active_clusters.to("cpu").numpy()
        draw_count = max(n_proposals * 2, n_proposals + 32)
        cell_ids = torch.randint(
            state.n_cells,
            (draw_count,),
            generator=generator,
            device=self.device,
        )
        unique: dict[tuple[int, int, int], MoveProposal] = {}
        rank_weights: dict[int, np.ndarray] = {}
        chunk_size = min(max(32, int(self._cuda_min_chunk_size // 8)), draw_count)
        for start in range(0, draw_count, chunk_size):
            chunk = cell_ids[start : start + chunk_size]
            source_cids, top_clusters, valid = self._rank_targets_for_cells_chunk(
                state,
                chunk,
                limit=max(1, int(limit)),
            )
            chunk_cells = chunk.to("cpu").tolist()
            source_np = source_cids.to("cpu").numpy()
            top_np = top_clusters.to("cpu").numpy()
            valid_np = valid.to("cpu").numpy()
            for row, cell in enumerate(chunk_cells):
                source_cluster = int(source_np[row])
                target_cluster: int | None = None
                ranked_targets = [
                    int(cluster_id)
                    for cluster_id, keep in zip(
                        top_np[row].tolist(), valid_np[row].tolist(), strict=True
                    )
                    if keep and int(cluster_id) != source_cluster
                ]
                if ranked_targets and rng.random() >= uniform_prob:
                    n_ranked = len(ranked_targets)
                    weights = rank_weights.get(n_ranked)
                    if weights is None:
                        weights = np.linspace(
                            float(n_ranked), 1.0, int(n_ranked), dtype=np.float64
                        )
                        weights = weights / float(weights.sum())
                        rank_weights[n_ranked] = weights
                    target_cluster = int(
                        rng.choice(np.asarray(ranked_targets), p=weights)
                    )
                if target_cluster is None:
                    uniform_pool = active_np[active_np != source_cluster]
                    if uniform_pool.size:
                        target_cluster = int(rng.choice(uniform_pool))
                if target_cluster is None or target_cluster == source_cluster:
                    continue
                key = (int(cell), source_cluster, int(target_cluster))
                unique.setdefault(
                    key,
                    MoveProposal(
                        cell=int(cell),
                        source_cluster=source_cluster,
                        target_cluster=int(target_cluster),
                    ),
                )
                if len(unique) >= n_proposals:
                    break
            if len(unique) >= n_proposals:
                break

        proposals = list(unique.values())
        if (
            max_unique_proposals is not None
            and int(max_unique_proposals) > 0
            and len(proposals) > int(max_unique_proposals)
        ):
            keep = rng.choice(
                len(proposals), size=int(max_unique_proposals), replace=False
            )
            proposals = [proposals[int(idx)] for idx in keep.tolist()]
        return proposals

    def select_nonconflicting_candidates(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        if torch is None:
            return list(candidates)

        touch_pairs: list[tuple[int, int]] = []
        scores = torch.empty(len(candidates), dtype=torch.float64, device=self.device)
        max_touch_id = -1
        for idx, candidate in enumerate(candidates):
            scores[idx] = float(candidate["delta"])
            touch_ids = candidate.get("touch_ids")
            if touch_ids is None:
                touch_ids = tuple(sorted(int(x) for x in candidate["touch_set"]))
            if not touch_ids:
                touch_pairs.append((-1, -1))
                continue
            first = int(touch_ids[0])
            second = int(touch_ids[1]) if len(touch_ids) > 1 else -1
            touch_pairs.append((first, second))
            if first >= 0:
                max_touch_id = max(max_touch_id, first)
            if second >= 0:
                max_touch_id = max(max_touch_id, second)

        if max_touch_id < 0:
            return []

        cluster_capacity = max(self.cluster_capacity, max_touch_id + 1)
        touch_tensor = torch.tensor(touch_pairs, dtype=torch.int64, device=self.device)
        touch_a = touch_tensor[:, 0]
        touch_b = touch_tensor[:, 1]
        finite = torch.isfinite(scores)
        if not torch.any(finite):
            return []

        order = torch.argsort(scores, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(
            len(candidates), dtype=torch.int64, device=self.device
        )

        remaining = finite.clone()
        accepted = torch.zeros(len(candidates), dtype=torch.bool, device=self.device)
        occupied = torch.zeros(cluster_capacity, dtype=torch.bool, device=self.device)

        while True:
            visible = remaining.clone()
            if torch.any(occupied):
                blocked = torch.zeros_like(visible)
                valid_a = touch_a >= 0
                if torch.any(valid_a):
                    blocked[valid_a] |= occupied[touch_a[valid_a]]
                valid_b = touch_b >= 0
                if torch.any(valid_b):
                    blocked[valid_b] |= occupied[touch_b[valid_b]]
                visible &= ~blocked
                remaining &= ~blocked
            if not torch.any(visible):
                break

            visible_ranks = ranks[visible]
            visible_touches = touch_tensor[visible]
            flat_ids = visible_touches.reshape(-1)
            flat_valid = flat_ids >= 0
            if not torch.any(flat_valid):
                break
            flat_ranks = visible_ranks.repeat_interleave(2)[flat_valid]
            cluster_best = torch.full(
                (cluster_capacity,),
                len(candidates),
                dtype=torch.int64,
                device=self.device,
            )
            cluster_best.scatter_reduce_(
                0, flat_ids[flat_valid], flat_ranks, reduce="amin", include_self=True
            )

            best_a = torch.ones_like(visible, dtype=torch.bool)
            valid_a = touch_a >= 0
            if torch.any(valid_a):
                best_a[valid_a] = ranks[valid_a] == cluster_best[touch_a[valid_a]]
            best_b = torch.ones_like(visible, dtype=torch.bool)
            valid_b = touch_b >= 0
            if torch.any(valid_b):
                best_b[valid_b] = ranks[valid_b] == cluster_best[touch_b[valid_b]]

            accept_now = visible & best_a & best_b
            if not torch.any(accept_now):
                break

            accepted |= accept_now
            remaining[accept_now] = False

            accepted_touches = touch_tensor[accept_now]
            accepted_flat = accepted_touches.reshape(-1)
            accepted_valid = accepted_flat >= 0
            if torch.any(accepted_valid):
                occupied[accepted_flat[accepted_valid]] = True

        accepted_indices = torch.nonzero(accepted, as_tuple=False).flatten()
        if accepted_indices.numel() == 0:
            return []
        accepted_indices = accepted_indices[
            torch.argsort(ranks.index_select(0, accepted_indices))
        ]
        return [candidates[int(idx)] for idx in accepted_indices.tolist()]

    def _chunked(
        self, indexed: list[tuple[int, Proposal]]
    ) -> list[list[tuple[int, Proposal]]]:
        if not indexed:
            return []
        chunk_size = int(self.chunk_size) if self.chunk_size is not None else 8192
        return [
            indexed[start : start + chunk_size]
            for start in range(0, len(indexed), chunk_size)
        ]

    def _proposal_width(self, state: PartitionState, proposal: Proposal) -> int:
        if isinstance(proposal, MergeProposal):
            vector_a = state.clusters[proposal.cluster_a]
            vector_b = state.clusters[proposal.cluster_b]
            return int(min(vector_a.nnz, vector_b.nnz))
        if isinstance(proposal, PeelProposal):
            return int(state.cell_nnz[int(proposal.cell)])
        if isinstance(proposal, MoveProposal):
            return int(state.cell_nnz[int(proposal.cell)])
        if isinstance(proposal, BlockPeelProposal):
            return int(proposal.block.indices.size)
        if isinstance(proposal, BlockMoveProposal):
            return int(proposal.block.indices.size)
        return 1

    def _cuda_memory_info(self) -> tuple[int, int] | None:
        if self.device.type != "cuda" or torch is None or not torch.cuda.is_available():
            return None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except TypeError:  # pragma: no cover - older torch variants
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:  # pragma: no cover - defensive
            return None
        return int(free_bytes), int(total_bytes)

    def _adaptive_chunk_size(
        self,
        width: int,
        *,
        memory_info: tuple[int, int] | None = None,
    ) -> int:
        if self.chunk_size is not None:
            return int(self.chunk_size)
        width = max(1, int(width))
        if memory_info is None:
            target_entries = 1_048_576
        else:
            free_bytes, _ = memory_info
            target_entries = int(
                max(
                    131_072,
                    min(
                        4_194_304,
                        (free_bytes * self._cuda_target_fraction)
                        / self._cuda_bytes_per_entry,
                    ),
                )
            )
        chunk_size = max(1, target_entries // width)
        return int(
            max(
                self._cuda_min_chunk_size,
                min(self._cuda_max_chunk_size, chunk_size),
            )
        )

    def _chunk_proposals(
        self, state: PartitionState, indexed: list[tuple[int, Proposal]]
    ) -> list[list[tuple[int, Proposal]]]:
        if not indexed:
            return []
        if self.chunk_size is not None or self.device.type != "cuda":
            return self._chunked(indexed)
        memory_info = self._cuda_memory_info()
        annotated = [
            (idx, proposal, self._proposal_width(state, proposal))
            for idx, proposal in indexed
        ]
        annotated.sort(key=lambda item: item[2], reverse=True)
        chunks: list[list[tuple[int, Proposal]]] = []
        start = 0
        while start < len(annotated):
            width = int(annotated[start][2])
            chunk_size = self._adaptive_chunk_size(width, memory_info=memory_info)
            end = min(len(annotated), start + chunk_size)
            chunks.append(
                [(idx, proposal) for idx, proposal, _ in annotated[start:end]]
            )
            start = end
        return chunks

    def _score_peels(
        self,
        state: PartitionState,
        indexed: list[tuple[int, Proposal]],
        scores: np.ndarray,
    ) -> None:
        batch = [proposal for _, proposal in indexed]
        cell_ids = torch.tensor(
            [proposal.cell for proposal in batch], dtype=torch.int64, device=self.device
        )
        source_cids = torch.tensor(
            [proposal.source_cluster for proposal in batch],
            dtype=torch.int64,
            device=self.device,
        )

        starts = self.X_indptr_tensor[cell_ids]
        ends = self.X_indptr_tensor[cell_ids + 1]
        lengths = ends - starts
        max_len = int(lengths.max().item()) if lengths.numel() else 0

        if max_len == 0:
            source_totals = self.cluster_totals[source_cids]
            cell_totals = self.cell_totals_tensor[cell_ids]
            deltas = torch.lgamma(source_totals + self.psi0_tensor) - torch.lgamma(
                source_totals - cell_totals + self.psi0_tensor
            )
            deltas += self.cell_ll_tensor[cell_ids]
            scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()
            return

        grid = torch.arange(max_len, device=self.device).expand(len(batch), max_len)
        mask = grid < lengths.unsqueeze(1)

        index_tensor = starts.unsqueeze(1) + torch.arange(max_len, device=self.device)
        clamped_indices = torch.where(
            mask, index_tensor, torch.zeros_like(index_tensor)
        )

        genes_2d = torch.gather(
            self.X_indices_tensor, 0, clamped_indices.view(-1)
        ).view(len(batch), max_len)
        counts_2d = torch.gather(self.X_data_tensor, 0, clamped_indices.view(-1)).view(
            len(batch), max_len
        )

        genes_2d = torch.where(mask, genes_2d, torch.zeros_like(genes_2d))
        counts_2d = torch.where(mask, counts_2d, torch.zeros_like(counts_2d))

        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(
            len(batch), max_len
        )
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))

        rows = source_cids.unsqueeze(1).expand(len(batch), max_len)
        source_pad = self.cluster_matrix[rows, genes_2d]
        source_pad = torch.where(mask, source_pad, torch.zeros_like(source_pad))

        source_totals = self.cluster_totals[source_cids]
        cell_totals = self.cell_totals_tensor[cell_ids]

        delta = torch.lgamma(source_totals + self.psi0_tensor)
        delta -= torch.lgamma(source_totals - cell_totals + self.psi0_tensor)
        delta += torch.sum(
            torch.lgamma(source_pad - counts_2d + psi_pad)
            - torch.lgamma(source_pad + psi_pad),
            dim=1,
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
        cell_ids = torch.tensor(
            [proposal.cell for proposal in batch], dtype=torch.int64, device=self.device
        )
        source_cids = torch.tensor(
            [proposal.source_cluster for proposal in batch],
            dtype=torch.int64,
            device=self.device,
        )
        target_cids = torch.tensor(
            [proposal.target_cluster for proposal in batch],
            dtype=torch.int64,
            device=self.device,
        )

        starts = self.X_indptr_tensor[cell_ids]
        ends = self.X_indptr_tensor[cell_ids + 1]
        lengths = ends - starts
        max_len = int(lengths.max().item()) if lengths.numel() else 0

        if max_len == 0:
            source_totals = self.cluster_totals[source_cids]
            target_totals = self.cluster_totals[target_cids]
            cell_totals = self.cell_totals_tensor[cell_ids]

            source_delta = torch.lgamma(
                source_totals + self.psi0_tensor
            ) - torch.lgamma(source_totals - cell_totals + self.psi0_tensor)
            target_delta = torch.lgamma(
                target_totals + self.psi0_tensor
            ) - torch.lgamma(target_totals + cell_totals + self.psi0_tensor)
            deltas = source_delta + target_delta
            scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()
            return

        grid = torch.arange(max_len, device=self.device).expand(len(batch), max_len)
        mask = grid < lengths.unsqueeze(1)

        index_tensor = starts.unsqueeze(1) + torch.arange(max_len, device=self.device)
        clamped_indices = torch.where(
            mask, index_tensor, torch.zeros_like(index_tensor)
        )

        genes_2d = torch.gather(
            self.X_indices_tensor, 0, clamped_indices.view(-1)
        ).view(len(batch), max_len)
        counts_2d = torch.gather(self.X_data_tensor, 0, clamped_indices.view(-1)).view(
            len(batch), max_len
        )

        genes_2d = torch.where(mask, genes_2d, torch.zeros_like(genes_2d))
        counts_2d = torch.where(mask, counts_2d, torch.zeros_like(counts_2d))

        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(
            len(batch), max_len
        )
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
            torch.lgamma(source_pad - counts_2d + psi_pad)
            - torch.lgamma(source_pad + psi_pad),
            dim=1,
        )

        target_delta = torch.lgamma(target_totals + self.psi0_tensor)
        target_delta -= torch.lgamma(target_totals + cell_totals + self.psi0_tensor)
        target_delta += torch.sum(
            torch.lgamma(target_pad + counts_2d + psi_pad)
            - torch.lgamma(target_pad + psi_pad),
            dim=1,
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

        max_len = int(lengths.max()) if lengths.size else 0
        if max_len == 0:
            large_totals = torch.tensor(
                large_totals_arr, dtype=torch.float64, device=self.device
            )
            small_totals = torch.tensor(
                small_totals_arr, dtype=torch.float64, device=self.device
            )
            small_ll = torch.tensor(
                small_ll_arr, dtype=torch.float64, device=self.device
            )

            deltas = (
                torch.lgamma(large_totals + self.psi0_tensor)
                - torch.lgamma(large_totals + small_totals + self.psi0_tensor)
                - small_ll
            )
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
        small_pad = torch.tensor(
            small_counts_np, dtype=torch.float64, device=self.device
        )
        mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)

        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(
            len(batch), max_len
        )
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))

        large_cids_tensor = torch.tensor(
            large_cids, dtype=torch.int64, device=self.device
        )
        rows = large_cids_tensor.unsqueeze(1).expand(len(batch), max_len)
        large_pad = self.cluster_matrix[rows, genes_2d]
        large_pad = torch.where(mask, large_pad, torch.zeros_like(large_pad))

        large_totals = torch.tensor(
            large_totals_arr, dtype=torch.float64, device=self.device
        )
        small_totals = torch.tensor(
            small_totals_arr, dtype=torch.float64, device=self.device
        )
        small_ll = torch.tensor(small_ll_arr, dtype=torch.float64, device=self.device)

        delta = torch.lgamma(large_totals + self.psi0_tensor)
        delta -= torch.lgamma(large_totals + small_totals + self.psi0_tensor)
        delta += torch.sum(
            torch.lgamma(large_pad + small_pad + psi_pad)
            - torch.lgamma(large_pad + psi_pad),
            dim=1,
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
        lengths = np.asarray(
            [proposal.block.indices.size for proposal in batch], dtype=np.int64
        )

        for row, proposal in enumerate(batch):
            block_ll_arr[row] = cluster_log_likelihood_from_sparse(
                proposal.block.indices,
                proposal.block.values,
                self.psi,
                psi0=self.psi0,
            )
            source_totals_arr[row] = state.cluster_total(proposal.source_cluster)
            block_totals_arr[row] = proposal.block.total

        max_len = int(lengths.max()) if lengths.size else 0
        if max_len == 0:
            source_totals = torch.tensor(
                source_totals_arr, dtype=torch.float64, device=self.device
            )
            block_totals = torch.tensor(
                block_totals_arr, dtype=torch.float64, device=self.device
            )
            block_ll = torch.tensor(
                block_ll_arr, dtype=torch.float64, device=self.device
            )

            deltas = (
                torch.lgamma(source_totals + self.psi0_tensor)
                - torch.lgamma(source_totals - block_totals + self.psi0_tensor)
                + block_ll
            )
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
        block_pad = torch.tensor(
            block_counts_np, dtype=torch.float64, device=self.device
        )
        mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)

        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(
            len(batch), max_len
        )
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))

        source_cids = torch.tensor(
            [proposal.source_cluster for proposal in batch],
            dtype=torch.int64,
            device=self.device,
        )
        rows = source_cids.unsqueeze(1).expand(len(batch), max_len)
        source_pad = self.cluster_matrix[rows, genes_2d]
        source_pad = torch.where(mask, source_pad, torch.zeros_like(source_pad))

        source_totals = torch.tensor(
            source_totals_arr, dtype=torch.float64, device=self.device
        )
        block_totals = torch.tensor(
            block_totals_arr, dtype=torch.float64, device=self.device
        )
        block_ll = torch.tensor(block_ll_arr, dtype=torch.float64, device=self.device)

        delta = torch.lgamma(source_totals + self.psi0_tensor)
        delta -= torch.lgamma(source_totals - block_totals + self.psi0_tensor)
        delta += torch.sum(
            torch.lgamma(source_pad - block_pad + psi_pad)
            - torch.lgamma(source_pad + psi_pad),
            dim=1,
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
        lengths = np.asarray(
            [proposal.block.indices.size for proposal in batch], dtype=np.int64
        )

        for row, proposal in enumerate(batch):
            source_totals_arr[row] = state.cluster_total(proposal.source_cluster)
            target_totals_arr[row] = state.cluster_total(proposal.target_cluster)
            block_totals_arr[row] = proposal.block.total

        max_len = int(lengths.max()) if lengths.size else 0
        if max_len == 0:
            source_totals = torch.tensor(
                source_totals_arr, dtype=torch.float64, device=self.device
            )
            target_totals = torch.tensor(
                target_totals_arr, dtype=torch.float64, device=self.device
            )
            block_totals = torch.tensor(
                block_totals_arr, dtype=torch.float64, device=self.device
            )

            deltas = torch.lgamma(source_totals + self.psi0_tensor) - torch.lgamma(
                source_totals - block_totals + self.psi0_tensor
            )
            deltas += torch.lgamma(target_totals + self.psi0_tensor) - torch.lgamma(
                target_totals + block_totals + self.psi0_tensor
            )
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
        block_pad = torch.tensor(
            block_counts_np, dtype=torch.float64, device=self.device
        )
        mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)

        psi_pad = torch.gather(self.psi_tensor, 0, genes_2d.view(-1)).view(
            len(batch), max_len
        )
        psi_pad = torch.where(mask, psi_pad, torch.ones_like(psi_pad))

        source_cids = torch.tensor(
            [proposal.source_cluster for proposal in batch],
            dtype=torch.int64,
            device=self.device,
        )
        rows_source = source_cids.unsqueeze(1).expand(len(batch), max_len)
        source_pad = self.cluster_matrix[rows_source, genes_2d]
        source_pad = torch.where(mask, source_pad, torch.zeros_like(source_pad))

        target_cids = torch.tensor(
            [proposal.target_cluster for proposal in batch],
            dtype=torch.int64,
            device=self.device,
        )
        rows_target = target_cids.unsqueeze(1).expand(len(batch), max_len)
        target_pad = self.cluster_matrix[rows_target, genes_2d]
        target_pad = torch.where(mask, target_pad, torch.zeros_like(target_pad))

        source_totals = torch.tensor(
            source_totals_arr, dtype=torch.float64, device=self.device
        )
        target_totals = torch.tensor(
            target_totals_arr, dtype=torch.float64, device=self.device
        )
        block_totals = torch.tensor(
            block_totals_arr, dtype=torch.float64, device=self.device
        )

        source_delta = torch.lgamma(source_totals + self.psi0_tensor)
        source_delta -= torch.lgamma(source_totals - block_totals + self.psi0_tensor)
        source_delta += torch.sum(
            torch.lgamma(source_pad - block_pad + psi_pad)
            - torch.lgamma(source_pad + psi_pad),
            dim=1,
        )

        target_delta = torch.lgamma(target_totals + self.psi0_tensor)
        target_delta -= torch.lgamma(target_totals + block_totals + self.psi0_tensor)
        target_delta += torch.sum(
            torch.lgamma(target_pad + block_pad + psi_pad)
            - torch.lgamma(target_pad + psi_pad),
            dim=1,
        )

        deltas = source_delta + target_delta
        scores[[idx for idx, _ in indexed]] = deltas.cpu().numpy()


class TorchCudaBackend(TorchDeviceBackend):
    def __init__(
        self,
        psi: np.ndarray,
        state: PartitionState,
        *,
        chunk_size: int | None = None,
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
        chunk_size: int | None = 8192,
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
    chunk_size: int | None = None,
) -> CPUBackend:
    backend = backend.lower()
    if backend in {"cpu", "numpy"}:
        return CPUBackend(psi, state, num_threads=num_threads)
    if backend in {"torch-cpu", "cpu-torch", "torch"}:
        return TorchCPUBackend(psi, state, num_threads=num_threads)
    if backend == "cuda":
        return TorchCudaBackend(
            psi,
            state,
            num_threads=num_threads,
            chunk_size=None if chunk_size is None else int(chunk_size),
        )
    if backend == "auto":
        if _torch_device_available("cuda") and _torch_device_supports_float64("cuda"):
            return TorchCudaBackend(
                psi,
                state,
                num_threads=num_threads,
                chunk_size=None if chunk_size is None else int(chunk_size),
            )
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
