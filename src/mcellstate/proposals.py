from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import time
from typing import Callable, Literal

import numpy as np

from .likelihood import delta_peel_cell
from .state import PartitionState


@dataclass(frozen=True)
class BlockPayload:
    cells: tuple[int, ...]
    indices: np.ndarray
    values: np.ndarray
    total: int

    @classmethod
    def from_cells(cls, state: PartitionState, cells: tuple[int, ...]) -> "BlockPayload":
        vector = state.block_vector_from_cells(cells)
        indices, values = vector.sorted_items()
        return cls(
            cells=tuple(int(cell) for cell in cells),
            indices=np.asarray(indices, dtype=np.int64),
            values=np.asarray(values, dtype=np.int64),
            total=int(vector.total),
        )


@dataclass(frozen=True)
class MergeProposal:
    cluster_a: int
    cluster_b: int
    kind: Literal["merge"] = "merge"

    def touch_set(self) -> frozenset[int]:
        return frozenset((int(self.cluster_a), int(self.cluster_b)))


@dataclass(frozen=True)
class PeelProposal:
    cell: int
    source_cluster: int
    kind: Literal["peel"] = "peel"

    def touch_set(self) -> frozenset[int]:
        return frozenset((int(self.source_cluster),))


@dataclass(frozen=True)
class MoveProposal:
    cell: int
    source_cluster: int
    target_cluster: int
    kind: Literal["move"] = "move"

    def touch_set(self) -> frozenset[int]:
        return frozenset((int(self.source_cluster), int(self.target_cluster)))


@dataclass(frozen=True)
class BlockPeelProposal:
    block: BlockPayload
    source_cluster: int
    kind: Literal["block_peel"] = "block_peel"

    def touch_set(self) -> frozenset[int]:
        return frozenset((int(self.source_cluster),))


@dataclass(frozen=True)
class BlockMoveProposal:
    block: BlockPayload
    source_cluster: int
    target_cluster: int
    kind: Literal["block_move"] = "block_move"

    def touch_set(self) -> frozenset[int]:
        return frozenset((int(self.source_cluster), int(self.target_cluster)))


Proposal = MergeProposal | PeelProposal | MoveProposal | BlockPeelProposal | BlockMoveProposal


class ProposalSampler:
    def __init__(
        self,
        *,
        pi_merge: float = 0.5,
        pi_peel: float = 0.15,
        pi_move: float = 0.20,
        pi_block_peel: float = 0.075,
        pi_block_move: float = 0.075,
        epsilon_uniform: float = 0.05,
        merge_uniform_prob: float | None = None,
        peel_uniform_prob: float | None = None,
        move_uniform_prob: float | None = None,
        block_peel_uniform_prob: float | None = None,
        block_move_uniform_prob: float | None = None,
        top_merge_neighbors: int = 8,
        deterministic_merge_ratio: float = 0.25,
        move_neighbor_limit: int = 16,
        merge_gene_cluster_cap: int = 48,
        signature_top_genes: int = 16,
        signature_pool_size: int = 48,
        block_size_min: int = 2,
        block_size_max: int = 12,
        block_candidate_cells: int = 32,
        low_fit_cache_size: int = 8,
        move_focus_genes: int = 4,
        move_candidate_cap_per_gene: int = 8,
        move_candidate_cap_total: int = 32,
        proposal_workers: int = 1,
        seed: int | None = None,
    ) -> None:
        weights = np.asarray(
            [pi_merge, pi_peel, pi_move, pi_block_peel, pi_block_move],
            dtype=np.float64,
        )
        if np.any(weights < 0):
            raise ValueError("proposal weights must be non-negative")
        if weights.sum() <= 0.0:
            raise ValueError("at least one proposal weight must be positive")
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
        self.family_names = ("merge", "peel", "move", "block_peel", "block_move")
        self.family_weights = weights
        self.epsilon_uniform = float(epsilon_uniform)
        self.merge_uniform_prob = float(epsilon_uniform if merge_uniform_prob is None else merge_uniform_prob)
        self.peel_uniform_prob = float(epsilon_uniform if peel_uniform_prob is None else peel_uniform_prob)
        self.move_uniform_prob = float(epsilon_uniform if move_uniform_prob is None else move_uniform_prob)
        self.block_peel_uniform_prob = float(
            epsilon_uniform if block_peel_uniform_prob is None else block_peel_uniform_prob,
        )
        self.block_move_uniform_prob = float(
            epsilon_uniform if block_move_uniform_prob is None else block_move_uniform_prob,
        )
        self.top_merge_neighbors = int(top_merge_neighbors)
        self.deterministic_merge_ratio = float(deterministic_merge_ratio)
        self.move_neighbor_limit = int(move_neighbor_limit)
        self.merge_gene_cluster_cap = int(merge_gene_cluster_cap)
        self.signature_top_genes = int(signature_top_genes)
        self.signature_pool_size = int(signature_pool_size)
        self.block_size_min = int(block_size_min)
        self.block_size_max = int(block_size_max)
        self.block_candidate_cells = int(block_candidate_cells)
        self.low_fit_cache_size = max(1, int(low_fit_cache_size))
        self.move_focus_genes = int(move_focus_genes)
        self.move_candidate_cap_per_gene = int(move_candidate_cap_per_gene)
        self.move_candidate_cap_total = int(move_candidate_cap_total)
        self.proposal_workers = max(1, int(proposal_workers))
        self.rng = np.random.default_rng(seed)

        self._prepared_version: int | None = None
        self._pending_signature_refresh: set[int] = set()
        self._pending_gene_refresh: set[int] = set()
        self._active_clusters = np.empty(0, dtype=np.int64)
        self._cluster_lookup = np.empty(0, dtype=np.int64)
        self._non_singletons = np.empty(0, dtype=np.int64)
        self._idf_weights = np.empty(0, dtype=np.float64)
        self._signature_pools: dict[int, np.ndarray] = {}
        self._signature_genes: dict[int, np.ndarray] = {}
        self._signature_weights: dict[int, np.ndarray] = {}
        self._signature_gene_sources: dict[int, set[int]] = {}
        self._merge_neighbors: dict[int, np.ndarray] = {}
        self._merge_neighbor_weights: dict[int, np.ndarray] = {}
        self._merge_neighbor_sources = np.empty(0, dtype=np.int64)
        self._merge_pair_scores: dict[tuple[int, int], float] = {}
        self._deterministic_merge_pairs: list[tuple[int, int]] = []
        self._non_singleton_probs = np.empty(0, dtype=np.float64)
        self._candidate_clusters_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._bad_cell_candidates: dict[int, np.ndarray] = {}
        self._psi: np.ndarray | None = None
        self._trace: Callable[[str], None] | None = None

    def set_scoring_context(self, psi: np.ndarray | None) -> None:
        self._psi = None if psi is None else np.asarray(psi, dtype=np.float64)

    def set_trace(self, callback: Callable[[str], None] | None) -> None:
        self._trace = callback

    def _emit_trace(self, message: str) -> None:
        if self._trace is not None:
            self._trace(message)

    def _parallel_map(self, items: list, worker):
        if len(items) <= 1 or self.proposal_workers <= 1:
            return [worker(item) for item in items]
        max_workers = min(self.proposal_workers, len(items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, item) for item in items]
            return [future.result() for future in futures]

    def set_family_weights(self, **weights: float) -> None:
        updated = []
        for family_name, old_weight in zip(self.family_names, self.family_weights, strict=True):
            updated.append(float(weights.get(family_name, old_weight)))
        array = np.asarray(updated, dtype=np.float64)
        if np.any(array < 0) or array.sum() <= 0.0:
            raise ValueError("family weights must stay non-negative with positive total mass")
        self.family_weights = array / array.sum()

    def notify_state_changed(
        self,
        state: PartitionState,
        touched_clusters: set[int] | None = None,
    ) -> None:
        if touched_clusters is None:
            touched_clusters = state.last_touched_clusters
        self._pending_signature_refresh.update(int(cluster_id) for cluster_id in touched_clusters)
        self._pending_gene_refresh.update(int(gene) for gene in np.asarray(state.last_touched_genes, dtype=np.int64))

    def prepare_round(self, state: PartitionState) -> None:
        if self._prepared_version == state.version:
            return
        start = time.perf_counter()
        self._emit_trace("sampler prepare_round start")
        self._active_clusters = state.active_cluster_array()
        self._refresh_cluster_lookup()
        self._non_singletons = np.asarray(sorted(state.non_singleton_cluster_ids()), dtype=np.int64)
        if self._non_singletons.size:
            sizes = np.asarray([state.cluster_size(cluster_id) for cluster_id in self._non_singletons], dtype=np.float64)
            total = float(sizes.sum())
            self._non_singleton_probs = sizes / total if total > 0.0 else np.full_like(sizes, 1.0 / float(len(sizes)))
        else:
            self._non_singleton_probs = np.empty(0, dtype=np.float64)
        full_neighbor_rebuild = self._prepared_version is None
        if full_neighbor_rebuild:
            self._candidate_clusters_cache = {}
        else:
            self._invalidate_candidate_cluster_cache(self._pending_gene_refresh)
        self._idf_weights = np.log(
            (len(self._active_clusters) + 1.0) / (state.gene_cluster_counts.astype(np.float64) + 1.0),
        )
        t1 = time.perf_counter()
        self._refresh_bad_cell_cache(state)
        t2 = time.perf_counter()
        self._refresh_signature_pools(state)
        t3 = time.perf_counter()
        self._refresh_current_signatures(state)
        t4 = time.perf_counter()
        self._rebuild_merge_neighbors(state, full_rebuild=full_neighbor_rebuild)
        t5 = time.perf_counter()
        self._prepared_version = state.version
        self._pending_signature_refresh.clear()
        self._pending_gene_refresh.clear()
        self._emit_trace(
            "sampler prepare_round done "
            f"total_s={t5 - start:.3f} "
            f"base_s={t1 - start:.3f} "
            f"bad_cell_cache_s={t2 - t1:.3f} "
            f"signature_pool_s={t3 - t2:.3f} "
            f"signature_current_s={t4 - t3:.3f} "
            f"merge_neighbors_s={t5 - t4:.3f} "
            f"active={self._active_clusters.size} non_singletons={self._non_singletons.size}",
        )

    def _refresh_bad_cell_cache(self, state: PartitionState) -> None:
        active_non_singletons = set(int(cluster_id) for cluster_id in self._non_singletons.tolist())
        stale_clusters = set(self._bad_cell_candidates).difference(active_non_singletons)
        for cluster_id in stale_clusters:
            self._bad_cell_candidates.pop(cluster_id, None)
        if not active_non_singletons:
            return
        if not self._bad_cell_candidates or self._prepared_version is None:
            refresh_clusters = active_non_singletons
        else:
            refresh_clusters = {
                int(cluster_id)
                for cluster_id in self._pending_signature_refresh
                if int(cluster_id) in active_non_singletons
            }
            refresh_clusters.update(active_non_singletons.difference(self._bad_cell_candidates))
        refresh_list = sorted(refresh_clusters)
        if not refresh_list:
            return
        refresh_seeds = self.rng.integers(np.iinfo(np.int64).max, size=len(refresh_list), dtype=np.int64)

        def build_bad_cells(task: tuple[int, int]) -> tuple[int, np.ndarray]:
            cluster_id, seed = task
            rng = np.random.default_rng(int(seed))
            membership = state.cells_by_cluster[int(cluster_id)]
            candidates = np.asarray(membership.cells, dtype=np.int64)
            if candidates.size == 0:
                return int(cluster_id), np.empty(0, dtype=np.int64)
            sample_size = min(self.block_candidate_cells, len(candidates))
            sampled = rng.choice(candidates, size=sample_size, replace=False)
            poor_fit_scores = np.asarray(
                [self._cell_poor_fit_score(state, int(cluster_id), int(cell)) for cell in sampled],
                dtype=np.float64,
            )
            order = np.argsort(poor_fit_scores)[::-1][: min(self.low_fit_cache_size, sampled.size)]
            return int(cluster_id), np.asarray(sampled[order], dtype=np.int64)

        tasks = list(zip(refresh_list, refresh_seeds.tolist(), strict=True))
        for cluster_id, cached_cells in self._parallel_map(tasks, build_bad_cells):
            self._bad_cell_candidates[int(cluster_id)] = cached_cells

    def _refresh_signature_pools(self, state: PartitionState) -> None:
        active_set = set(int(cluster_id) for cluster_id in self._active_clusters.tolist())
        stale_clusters = set(self._signature_pools).difference(active_set)
        for cluster_id in stale_clusters:
            self._signature_pools.pop(cluster_id, None)
            self._signature_genes.pop(cluster_id, None)
            self._signature_weights.pop(cluster_id, None)

        if not self._signature_pools:
            refresh_clusters = active_set
        else:
            refresh_clusters = {int(cluster_id) for cluster_id in active_set if int(cluster_id) not in self._signature_pools}
            refresh_clusters.update(
                int(cluster_id)
                for cluster_id in self._pending_signature_refresh
                if int(cluster_id) in active_set
            )
        refresh_list = sorted(int(cluster_id) for cluster_id in refresh_clusters)

        def build_pool(cluster_id: int) -> tuple[int, np.ndarray | None]:
            genes, counts = state.clusters[cluster_id].sorted_items()
            if genes.size == 0:
                return int(cluster_id), None
            order = np.argsort(counts)[::-1][: self.signature_pool_size]
            return int(cluster_id), np.asarray(genes[order], dtype=np.int64)

        for cluster_id, pool in self._parallel_map(refresh_list, build_pool):
            if pool is None:
                self._signature_pools.pop(cluster_id, None)
                continue
            self._signature_pools[cluster_id] = pool

    def _refresh_current_signatures(self, state: PartitionState) -> None:
        self._signature_genes = {}
        self._signature_weights = {}
        active_clusters = [int(cluster_id) for cluster_id in self._active_clusters.tolist()]

        def build_signature(cluster_id: int) -> tuple[int, np.ndarray | None, np.ndarray | None]:
            pool = self._signature_pools.get(int(cluster_id))
            if pool is None or pool.size == 0:
                return int(cluster_id), None, None
            pool_counts = state.clusters[int(cluster_id)].get_many(pool).astype(np.float64)
            pool_weights = pool_counts * np.maximum(self._idf_weights[pool], 0.0)
            positive = pool_weights > 0.0
            if not np.any(positive):
                genes, counts = state.clusters[int(cluster_id)].sorted_items()
                if genes.size == 0:
                    return int(cluster_id), None, None
                order = np.argsort(counts)[::-1][: self.signature_top_genes]
                chosen_genes = np.asarray(genes[order], dtype=np.int64)
                chosen_weights = np.asarray(counts[order], dtype=np.float64)
            else:
                candidate_genes = pool[positive]
                candidate_weights = pool_weights[positive]
                order = np.argsort(candidate_weights)[::-1][: self.signature_top_genes]
                chosen_genes = np.asarray(candidate_genes[order], dtype=np.int64)
                chosen_weights = np.asarray(candidate_weights[order], dtype=np.float64)
            return int(cluster_id), chosen_genes, chosen_weights

        signature_gene_sources: dict[int, set[int]] = defaultdict(set)
        for cluster_id, chosen_genes, chosen_weights in self._parallel_map(active_clusters, build_signature):
            if chosen_genes is None or chosen_weights is None:
                continue
            self._signature_genes[int(cluster_id)] = chosen_genes
            self._signature_weights[int(cluster_id)] = chosen_weights
            for gene in chosen_genes.tolist():
                signature_gene_sources[int(gene)].add(int(cluster_id))
        self._signature_gene_sources = dict(signature_gene_sources)

    def _refresh_cluster_lookup(self) -> None:
        if self._active_clusters.size == 0:
            self._cluster_lookup = np.empty(0, dtype=np.int64)
            return
        max_cluster_id = int(np.max(self._active_clusters))
        self._cluster_lookup = np.full(max_cluster_id + 1, -1, dtype=np.int64)
        self._cluster_lookup[self._active_clusters] = np.arange(self._active_clusters.size, dtype=np.int64)

    def _invalidate_candidate_cluster_cache(self, genes: set[int]) -> None:
        for gene in genes:
            self._candidate_clusters_cache.pop(int(gene), None)

    def _candidate_cluster_data_for_gene(
        self,
        state: PartitionState,
        gene: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._candidate_clusters_cache.get(int(gene))
        if cached is not None:
            cluster_ids, counts = cached
        else:
            clusters = state.gene_to_clusters.get(int(gene))
            if not clusters:
                result = (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                )
                self._candidate_clusters_cache[int(gene)] = (
                    result[0],
                    result[2],
                )
                return result
            cluster_ids = np.fromiter((int(cluster_id) for cluster_id in clusters), dtype=np.int64, count=len(clusters))
            counts = np.asarray([state.clusters[int(cluster_id)].get(int(gene)) for cluster_id in cluster_ids], dtype=np.int64)
            order = np.argsort(counts)[::-1]
            cluster_ids = cluster_ids[order]
            counts = counts[order]
            if cluster_ids.size > self.merge_gene_cluster_cap:
                cluster_ids = np.asarray(cluster_ids[: self.merge_gene_cluster_cap], dtype=np.int64)
                counts = np.asarray(counts[: self.merge_gene_cluster_cap], dtype=np.int64)
            self._candidate_clusters_cache[int(gene)] = (cluster_ids, counts)

        valid = cluster_ids < self._cluster_lookup.size
        cluster_ids = cluster_ids[valid]
        counts = counts[valid]
        if cluster_ids.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        positions = self._cluster_lookup[cluster_ids]
        valid_positions = positions >= 0
        return (
            np.asarray(cluster_ids[valid_positions], dtype=np.int64),
            np.asarray(positions[valid_positions], dtype=np.int64),
            np.asarray(counts[valid_positions], dtype=np.int64),
        )

    def _candidate_clusters_for_gene(self, state: PartitionState, gene: int) -> np.ndarray:
        cluster_ids, _, _ = self._candidate_cluster_data_for_gene(state, gene)
        return cluster_ids

    def _top_cluster_ids_from_scores(
        self,
        scores: np.ndarray,
        *,
        limit: int | None,
        exclude_cluster: int | None = None,
    ) -> list[int]:
        candidate_positions = np.flatnonzero(scores > 0.0)
        if exclude_cluster is not None and 0 <= int(exclude_cluster) < self._cluster_lookup.size:
            source_pos = int(self._cluster_lookup[int(exclude_cluster)])
            if source_pos >= 0:
                candidate_positions = candidate_positions[candidate_positions != source_pos]
        if candidate_positions.size == 0:
            return []
        if limit is not None and candidate_positions.size > int(limit):
            top_rel = np.argpartition(scores[candidate_positions], -int(limit))[-int(limit):]
            candidate_positions = candidate_positions[top_rel]
        order = np.argsort(scores[candidate_positions])[::-1]
        return [int(self._active_clusters[pos]) for pos in candidate_positions[order]]

    def _build_merge_neighbors_for_source(
        self,
        state: PartitionState,
        source_cluster: int,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]]]:
        signature_genes = self._signature_genes.get(int(source_cluster))
        signature_weights = self._signature_weights.get(int(source_cluster))
        if signature_genes is None or signature_weights is None or signature_genes.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), []
        scores = np.zeros(self._active_clusters.size, dtype=np.float64)
        source_pos = int(self._cluster_lookup[int(source_cluster)]) if int(source_cluster) < self._cluster_lookup.size else -1
        for gene, source_weight in zip(signature_genes, signature_weights, strict=True):
            idf = max(float(self._idf_weights[int(gene)]), 0.0)
            if idf <= 0.0:
                continue
            _, positions, counts = self._candidate_cluster_data_for_gene(state, int(gene))
            if positions.size == 0:
                continue
            valid = positions != source_pos
            if not np.any(valid):
                continue
            gene_positions = positions[valid]
            target_weights = counts[valid].astype(np.float64) * idf
            scores[gene_positions] += np.minimum(float(source_weight), target_weights)
        ranked_ids = self._top_cluster_ids_from_scores(scores, limit=self.top_merge_neighbors, exclude_cluster=source_cluster)
        if not ranked_ids:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), []
        neighbors = np.asarray(ranked_ids, dtype=np.int64)
        neighbor_positions = self._cluster_lookup[neighbors]
        neighbor_scores = scores[neighbor_positions]
        weights = neighbor_scores / neighbor_scores.sum()
        ranked = [(int(cluster_id), float(score)) for cluster_id, score in zip(neighbors, neighbor_scores, strict=True)]
        return neighbors, weights, ranked

    def _remove_merge_source(self, source_cluster: int) -> None:
        source_cluster = int(source_cluster)
        self._merge_neighbors.pop(source_cluster, None)
        self._merge_neighbor_weights.pop(source_cluster, None)
        stale_pairs = [pair for pair in self._merge_pair_scores if source_cluster in pair]
        for pair in stale_pairs:
            del self._merge_pair_scores[pair]

    def _rebuild_merge_neighbors(self, state: PartitionState, *, full_rebuild: bool) -> None:
        active_set = set(int(cluster_id) for cluster_id in self._active_clusters.tolist())
        stale_sources = set(self._merge_neighbors).difference(active_set)
        for cluster_id in stale_sources:
            self._remove_merge_source(int(cluster_id))

        if full_rebuild:
            sources_to_rebuild = [int(cluster_id) for cluster_id in self._active_clusters.tolist()]
            self._merge_neighbors = {}
            self._merge_neighbor_weights = {}
            self._merge_pair_scores = {}
        else:
            impacted_sources = set(int(cluster_id) for cluster_id in self._pending_signature_refresh if int(cluster_id) in active_set)
            impacted_sources.update(active_set.difference(self._merge_neighbors))
            for gene in self._pending_gene_refresh:
                impacted_sources.update(int(cluster_id) for cluster_id in self._signature_gene_sources.get(int(gene), ()))
                cluster_ids, _, _ = self._candidate_cluster_data_for_gene(state, int(gene))
                impacted_sources.update(int(cluster_id) for cluster_id in cluster_ids.tolist())
            sources_to_rebuild = sorted(impacted_sources)

        rebuild_list = [int(cluster_id) for cluster_id in sources_to_rebuild]
        for cluster_id in rebuild_list:
            self._remove_merge_source(int(cluster_id))

        def build_source(cluster_id: int) -> tuple[int, np.ndarray, np.ndarray, list[tuple[int, float]]]:
            if int(cluster_id) not in active_set:
                return int(cluster_id), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), []
            neighbors, weights, ranked = self._build_merge_neighbors_for_source(state, int(cluster_id))
            return int(cluster_id), neighbors, weights, ranked

        for cluster_id, neighbors, weights, ranked in self._parallel_map(rebuild_list, build_source):
            if neighbors.size == 0:
                continue
            self._merge_neighbors[int(cluster_id)] = neighbors
            self._merge_neighbor_weights[int(cluster_id)] = weights
            for target_cluster, score in ranked:
                pair = tuple(sorted((int(cluster_id), int(target_cluster))))
                self._merge_pair_scores[pair] = self._merge_pair_scores.get(pair, 0.0) + float(score)

        ordered_pairs = sorted(self._merge_pair_scores.items(), key=lambda item: item[1], reverse=True)
        self._merge_neighbor_sources = (
            np.asarray(sorted(self._merge_neighbors), dtype=np.int64)
            if self._merge_neighbors
            else np.empty(0, dtype=np.int64)
        )
        self._deterministic_merge_pairs = [pair for pair, _ in ordered_pairs]

    def sample_batch(self, state: PartitionState, n_proposals: int) -> list[Proposal]:
        if n_proposals <= 0:
            return []
        start = time.perf_counter()
        self.prepare_round(state)
        after_prepare = time.perf_counter()
        family_counts = self.rng.multinomial(int(n_proposals), self.family_weights)
        proposals: list[Proposal] = []

        deterministic_budget = min(
            len(self._deterministic_merge_pairs),
            int(round(int(family_counts[0]) * self.deterministic_merge_ratio)),
        )
        proposals.extend(self._deterministic_merge_proposals(deterministic_budget))
        self._emit_trace(
            "sampler sample_batch start "
            f"prepare_s={after_prepare - start:.3f} "
            f"merge={int(family_counts[0])} peel={int(family_counts[1])} move={int(family_counts[2])} "
            f"block_peel={int(family_counts[3])} block_move={int(family_counts[4])} "
            f"deterministic_merge={deterministic_budget} "
            f"workers={self.proposal_workers}",
        )

        family_tasks: list[tuple[str, int]] = []
        for family_idx, family_name in enumerate(self.family_names):
            count = int(family_counts[family_idx])
            if family_name == "merge":
                count = max(0, count - deterministic_budget)
            family_tasks.append((family_name, count))
        sub_tasks = self._expand_family_tasks(family_tasks)
        if self.proposal_workers > 1 and len(sub_tasks) > 1:
            seeds = self.rng.integers(np.iinfo(np.int64).max, size=len(sub_tasks), dtype=np.int64)
            max_workers = min(self.proposal_workers, len(sub_tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._sample_family_parallel,
                        state,
                        family_name,
                        count,
                        int(seed),
                    )
                    for (family_name, count), seed in zip(sub_tasks, seeds, strict=True)
                ]
                results = [future.result() for future in futures]
            aggregated: dict[str, dict[str, object]] = {}
            for family_name, count, family_props, elapsed in results:
                record = aggregated.setdefault(
                    family_name,
                    {"count": 0, "proposals": [], "elapsed": 0.0},
                )
                record["count"] = int(record["count"]) + int(count)
                record["elapsed"] = float(record["elapsed"]) + float(elapsed)
                cast_props = record["proposals"]
                assert isinstance(cast_props, list)
                cast_props.extend(family_props)
            for family_name, _ in family_tasks:
                record = aggregated.get(family_name)
                if record is None:
                    continue
                family_props = record["proposals"]
                assert isinstance(family_props, list)
                proposals.extend(family_props)
                self._emit_trace(
                    f"sampler family={family_name} requested={int(record['count'])} produced={len(family_props)} time_s={float(record['elapsed']):.3f}",
                )
        else:
            for family_name, count in family_tasks:
                family_start = time.perf_counter()
                family_props = self._sample_family(state, family_name, count)
                proposals.extend(family_props)
                family_end = time.perf_counter()
                self._emit_trace(
                    f"sampler family={family_name} requested={count} produced={len(family_props)} time_s={family_end - family_start:.3f}",
                )
        self._emit_trace(f"sampler sample_batch done total_props={len(proposals)} total_s={time.perf_counter() - start:.3f}")
        return proposals

    def _expand_family_tasks(self, family_tasks: list[tuple[str, int]]) -> list[tuple[str, int]]:
        expanded: list[tuple[str, int]] = []
        for family_name, count in family_tasks:
            if count <= 0:
                continue
            if family_name not in {"move", "block_move"} or self.proposal_workers <= 1 or count < max(32, self.proposal_workers * 4):
                expanded.append((family_name, int(count)))
                continue
            n_chunks = min(self.proposal_workers, int(count))
            base = count // n_chunks
            rem = count % n_chunks
            for chunk_idx in range(n_chunks):
                chunk_count = base + (1 if chunk_idx < rem else 0)
                if chunk_count > 0:
                    expanded.append((family_name, int(chunk_count)))
        return expanded

    def _sample_family_parallel(
        self,
        state: PartitionState,
        family: str,
        count: int,
        seed: int,
    ) -> tuple[str, int, list[Proposal], float]:
        worker = copy.copy(self)
        worker.rng = np.random.default_rng(seed)
        worker._trace = None
        start = time.perf_counter()
        proposals = worker._sample_family(state, family, count)
        return family, count, proposals, time.perf_counter() - start

    def _deterministic_merge_proposals(self, budget: int) -> list[MergeProposal]:
        return [
            MergeProposal(cluster_a=int(cluster_a), cluster_b=int(cluster_b))
            for cluster_a, cluster_b in self._deterministic_merge_pairs[:budget]
        ]

    def _sample_family(self, state: PartitionState, family: str, count: int) -> list[Proposal]:
        sampled: list[Proposal] = []
        for _ in range(count):
            proposal: Proposal | None
            if family == "move":
                proposal = self._sample_move_uniform(state)
            elif family == "block_move":
                proposal = self._sample_block_move_uniform(state)
            else:
                if family == "merge":
                    uniform_prob = self.merge_uniform_prob
                elif family == "peel":
                    uniform_prob = self.peel_uniform_prob
                elif family == "block_peel":
                    uniform_prob = self.block_peel_uniform_prob
                else:
                    raise ValueError(f"unknown proposal family: {family!r}")
                use_uniform = self.rng.random() < uniform_prob
                if family == "merge":
                    proposal = self._sample_merge_uniform(state) if use_uniform else self._sample_merge_biased(state)
                else:
                    proposal = self._sample_peel_uniform(state) if use_uniform else self._sample_peel_biased(state)
            if proposal is not None:
                sampled.append(proposal)
        return sampled

    def merge_neighbors_for(self, cluster_id: int, limit: int | None = None) -> np.ndarray:
        neighbors = self._merge_neighbors.get(int(cluster_id), np.empty(0, dtype=np.int64))
        if limit is None:
            return neighbors
        return neighbors[: int(limit)]

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
        self.prepare_round(state)
        scores = np.zeros(self._active_clusters.size, dtype=np.float64)
        candidate_mask = np.zeros(self._active_clusters.size, dtype=bool)
        if candidate_targets is not None and np.asarray(candidate_targets, dtype=np.int64).size:
            candidate_targets = np.asarray(candidate_targets, dtype=np.int64)
            valid = candidate_targets < self._cluster_lookup.size
            candidate_targets = candidate_targets[valid]
            candidate_positions = self._cluster_lookup[candidate_targets] if candidate_targets.size else np.empty(0, dtype=np.int64)
            candidate_positions = candidate_positions[candidate_positions >= 0]
            if candidate_positions.size:
                candidate_mask[candidate_positions] = True
        source_pos = int(self._cluster_lookup[int(source_cluster)]) if int(source_cluster) < self._cluster_lookup.size else -1
        for gene, value in zip(np.asarray(genes, dtype=np.int64), np.asarray(values, dtype=np.int64), strict=True):
            idf = max(float(self._idf_weights[int(gene)]), 0.0)
            if idf <= 0.0:
                continue
            _, positions, counts = self._candidate_cluster_data_for_gene(state, int(gene))
            if positions.size == 0:
                continue
            valid_positions = positions != source_pos
            if candidate_targets is not None:
                valid_positions &= candidate_mask[positions]
            if not np.any(valid_positions):
                continue
            gene_positions = positions[valid_positions]
            scores[gene_positions] += np.minimum(float(value), counts[valid_positions].astype(np.float64)) * idf
        ranked_ids = self._top_cluster_ids_from_scores(scores, limit=limit or self.move_neighbor_limit, exclude_cluster=source_cluster)
        if not ranked_ids:
            neighbors = self.merge_neighbors_for(int(source_cluster), limit or self.move_neighbor_limit)
            return [int(cluster_id) for cluster_id in neighbors]
        return ranked_ids

    def _sample_merge_uniform(self, state: PartitionState) -> MergeProposal | None:
        del state
        if len(self._active_clusters) < 2:
            return None
        choice = self.rng.choice(self._active_clusters, size=2, replace=False)
        cluster_a, cluster_b = sorted((int(choice[0]), int(choice[1])))
        return MergeProposal(cluster_a=cluster_a, cluster_b=cluster_b)

    def _sample_merge_biased(self, state: PartitionState) -> MergeProposal | None:
        if not self._merge_neighbors:
            return self._sample_merge_uniform(state)
        source_cluster = int(self.rng.choice(self._merge_neighbor_sources))
        neighbors = self._merge_neighbors.get(source_cluster)
        if neighbors is None or neighbors.size == 0:
            return self._sample_merge_uniform(state)
        probs = self._merge_neighbor_weights[source_cluster]
        target_cluster = int(self.rng.choice(neighbors, p=probs))
        cluster_a, cluster_b = sorted((source_cluster, target_cluster))
        return MergeProposal(cluster_a=cluster_a, cluster_b=cluster_b)

    def _sample_non_singleton_cluster(self, state: PartitionState) -> int | None:
        del state
        if self._non_singletons.size == 0:
            return None
        return int(self.rng.choice(self._non_singletons, p=self._non_singleton_probs))

    def _sample_non_singleton_cell(self, state: PartitionState) -> tuple[int, int] | None:
        cluster_id = self._sample_non_singleton_cluster(state)
        if cluster_id is None:
            return None
        return state.cells_by_cluster[cluster_id].sample(self.rng), cluster_id

    def _sample_low_fit_cell(self, state: PartitionState, source_cluster: int) -> int:
        cached = self._bad_cell_candidates.get(int(source_cluster))
        if cached is not None and cached.size:
            if cached.size == 1:
                return int(cached[0])
            weights = np.linspace(float(cached.size), 1.0, int(cached.size), dtype=np.float64)
            weights = weights / weights.sum()
            return int(self.rng.choice(cached, p=weights))
        membership = state.cells_by_cluster[source_cluster]
        candidates = np.asarray(membership.cells, dtype=np.int64)
        sample_size = min(self.block_candidate_cells, len(candidates))
        sampled = self.rng.choice(candidates, size=sample_size, replace=False)
        poor_fit_scores = np.asarray(
            [self._cell_poor_fit_score(state, source_cluster, int(cell)) for cell in sampled],
            dtype=np.float64,
        )
        return int(sampled[int(np.argmax(poor_fit_scores))])

    def _sample_peel_uniform(self, state: PartitionState) -> PeelProposal | None:
        sampled = self._sample_non_singleton_cell(state)
        if sampled is None:
            return None
        cell, cluster_id = sampled
        return PeelProposal(cell=int(cell), source_cluster=int(cluster_id))

    def _sample_peel_biased(self, state: PartitionState) -> PeelProposal | None:
        cluster_id = self._sample_non_singleton_cluster(state)
        if cluster_id is None:
            return None
        cell = self._sample_low_fit_cell(state, cluster_id)
        return PeelProposal(cell=cell, source_cluster=cluster_id)

    def _sample_move_uniform(self, state: PartitionState) -> MoveProposal | None:
        if len(self._active_clusters) < 2 or state.n_cells == 0:
            return None
        cell = int(self.rng.integers(state.n_cells))
        source_cluster = int(state.z[cell])
        target_cluster = self._sample_uniform_target(source_cluster)
        if target_cluster is None:
            return None
        return MoveProposal(cell=cell, source_cluster=source_cluster, target_cluster=target_cluster)

    def _sample_move_biased(self, state: PartitionState) -> MoveProposal | None:
        return self._sample_move_uniform(state)

    def _sample_block_peel_uniform(self, state: PartitionState) -> BlockPeelProposal | None:
        sampled = self._sample_block_source(state, biased=False)
        if sampled is None:
            return None
        source_cluster, block = sampled
        return BlockPeelProposal(block=block, source_cluster=source_cluster)

    def _sample_block_peel_biased(self, state: PartitionState) -> BlockPeelProposal | None:
        sampled = self._sample_block_source(state, biased=True)
        if sampled is None:
            return None
        source_cluster, block = sampled
        return BlockPeelProposal(block=block, source_cluster=source_cluster)

    def _sample_block_move_uniform(self, state: PartitionState) -> BlockMoveProposal | None:
        sampled = self._sample_block_source(state, biased=False)
        if sampled is None:
            return None
        source_cluster, block = sampled
        target_cluster = self._sample_uniform_target(source_cluster)
        if target_cluster is None:
            return None
        return BlockMoveProposal(block=block, source_cluster=source_cluster, target_cluster=target_cluster)

    def _sample_block_move_biased(self, state: PartitionState) -> BlockMoveProposal | None:
        sampled = self._sample_block_source(state, biased=True)
        if sampled is None:
            return None
        source_cluster, block = sampled
        candidate_targets = self._candidate_targets_for_payload(state, source_cluster, block.indices, block.values)
        targets = self.rank_target_clusters(
            state,
            source_cluster,
            block.indices,
            block.values,
            limit=self.move_neighbor_limit,
            candidate_targets=candidate_targets,
        )
        if not targets:
            return self._sample_block_move_uniform(state)
        target_cluster = int(targets[0] if len(targets) == 1 else self.rng.choice(np.asarray(targets, dtype=np.int64)))
        return BlockMoveProposal(block=block, source_cluster=source_cluster, target_cluster=target_cluster)

    def _candidate_targets_for_payload(
        self,
        state: PartitionState,
        source_cluster: int,
        genes: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray | None:
        candidates: list[np.ndarray] = []
        merge_neighbors = self.merge_neighbors_for(int(source_cluster), self.move_neighbor_limit)
        if merge_neighbors.size:
            candidates.append(np.asarray(merge_neighbors, dtype=np.int64))
        if genes.size:
            scores = values.astype(np.float64) * np.maximum(self._idf_weights[np.asarray(genes, dtype=np.int64)], 0.0)
            focus = min(self.move_focus_genes, genes.size)
            focus_order = np.argsort(scores)[::-1][:focus]
            for gene in np.asarray(genes, dtype=np.int64)[focus_order]:
                gene_candidates = self._candidate_clusters_for_gene(state, int(gene))
                if gene_candidates.size:
                    candidates.append(np.asarray(gene_candidates[: self.move_candidate_cap_per_gene], dtype=np.int64))
        if not candidates:
            return None
        ordered = np.concatenate(candidates)
        pooled = np.fromiter(
            (cluster_id for cluster_id in dict.fromkeys(int(cluster_id) for cluster_id in ordered.tolist()) if cluster_id != int(source_cluster)),
            dtype=np.int64,
        )
        if pooled.size == 0:
            return None
        if pooled.size > self.move_candidate_cap_total:
            pooled = pooled[: self.move_candidate_cap_total]
        return np.asarray(pooled, dtype=np.int64)

    def _sample_block_source(
        self,
        state: PartitionState,
        *,
        biased: bool,
    ) -> tuple[int, BlockPayload] | None:
        source_cluster = self._sample_non_singleton_cluster(state)
        if source_cluster is None:
            return None
        cluster_size = state.cluster_size(source_cluster)
        max_block = min(self.block_size_max, cluster_size - 1)
        if max_block < self.block_size_min:
            return None
        target_size = int(self.rng.integers(self.block_size_min, max_block + 1))
        if biased:
            block = self._biased_block_payload(state, source_cluster, target_size)
        else:
            cells = np.asarray(state.cells_by_cluster[source_cluster].cells, dtype=np.int64)
            selected = tuple(sorted(int(cell) for cell in self.rng.choice(cells, size=target_size, replace=False)))
            block = BlockPayload.from_cells(state, selected)
        if len(block.cells) == 0 or len(block.cells) >= cluster_size:
            return None
        return source_cluster, block

    def _biased_block_payload(
        self,
        state: PartitionState,
        source_cluster: int,
        target_size: int,
    ) -> BlockPayload:
        membership = state.cells_by_cluster[source_cluster]
        candidates = np.asarray(membership.cells, dtype=np.int64)
        sample_size = min(self.block_candidate_cells, len(candidates))
        sampled = self.rng.choice(candidates, size=sample_size, replace=False)
        poor_fit_scores = np.asarray(
            [self._cell_poor_fit_score(state, source_cluster, int(cell)) for cell in sampled],
            dtype=np.float64,
        )
        seed = int(sampled[int(np.argmax(poor_fit_scores))])
        seed_genes, seed_values = state.cell_counts(seed)

        similarity_scores = np.asarray(
            [self._shared_rare_overlap(state, seed_genes, seed_values, int(cell)) for cell in sampled],
            dtype=np.float64,
        )
        combined = self._rank01(similarity_scores) + self._rank01(poor_fit_scores)
        order = np.argsort(combined)[::-1][:target_size]
        picked = sampled[order]
        if seed not in picked:
            picked = np.asarray([seed, *picked[:-1]], dtype=np.int64)
        selected = tuple(sorted(int(cell) for cell in np.unique(picked)))
        return BlockPayload.from_cells(state, selected)

    def _sample_uniform_target(self, source_cluster: int) -> int | None:
        candidate_targets = self._active_clusters[self._active_clusters != int(source_cluster)]
        if candidate_targets.size == 0:
            return None
        return int(self.rng.choice(candidate_targets))

    def _cell_fit_score(self, state: PartitionState, source_cluster: int, cell: int) -> float:
        genes, values = state.cell_counts(cell)
        if genes.size == 0:
            return 0.0
        cluster = state.clusters[source_cluster]
        cluster_total = max(float(cluster.total), 1.0)
        cluster_counts = cluster.get_many(genes).astype(np.float64)
        probs = (cluster_counts + 1.0) / (cluster_total + float(genes.size))
        return float(np.sum(values.astype(np.float64) * np.log(probs)))

    def _cell_poor_fit_score(self, state: PartitionState, source_cluster: int, cell: int) -> float:
        if self._psi is None:
            return -self._cell_fit_score(state, source_cluster, cell)
        return delta_peel_cell(state, self._psi, cell, source_cluster)

    @staticmethod
    def _rank01(scores: np.ndarray) -> np.ndarray:
        if scores.size <= 1:
            return np.ones(scores.shape, dtype=np.float64)
        ranks = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort").astype(np.float64)
        return ranks / float(scores.size - 1)

    def _shared_rare_overlap(
        self,
        state: PartitionState,
        seed_genes: np.ndarray,
        seed_values: np.ndarray,
        cell: int,
    ) -> float:
        genes, values = state.cell_counts(cell)
        if genes.size == 0 or seed_genes.size == 0:
            return 0.0
        seed_map = {int(gene): int(value) for gene, value in zip(seed_genes, seed_values, strict=True)}
        score = 0.0
        for gene, value in zip(genes, values, strict=True):
            gene = int(gene)
            if gene not in seed_map:
                continue
            score += min(float(value), float(seed_map[gene])) * max(float(self._idf_weights[gene]), 0.0)
        return float(score)
