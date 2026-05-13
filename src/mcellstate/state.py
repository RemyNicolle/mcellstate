from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse


@dataclass
class CellMembership:
    cells: list[int]
    positions: dict[int, int]

    @classmethod
    def from_cells(cls, cells: Iterable[int]) -> "CellMembership":
        ordered = [int(cell) for cell in cells]
        return cls(ordered, {cell: idx for idx, cell in enumerate(ordered)})

    def copy(self) -> "CellMembership":
        return CellMembership(list(self.cells), dict(self.positions))

    def __len__(self) -> int:
        return len(self.cells)

    def add(self, cell: int) -> None:
        cell = int(cell)
        if cell in self.positions:
            raise ValueError(f"cell {cell} is already present")
        self.positions[cell] = len(self.cells)
        self.cells.append(cell)

    def extend(self, other: "CellMembership") -> None:
        for cell in other.cells:
            self.add(cell)

    def remove(self, cell: int) -> None:
        cell = int(cell)
        idx = self.positions.pop(cell)
        last = self.cells.pop()
        if idx < len(self.cells):
            self.cells[idx] = last
            self.positions[last] = idx

    def sample(self, rng: np.random.Generator) -> int:
        if not self.cells:
            raise ValueError("cannot sample from an empty membership")
        return int(self.cells[int(rng.integers(len(self.cells)))])


@dataclass
class SparseCountVector:
    counts: dict[int, int]
    total: int
    _sorted_indices: np.ndarray | None = None
    _sorted_values: np.ndarray | None = None

    @classmethod
    def from_indices(
        cls,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> "SparseCountVector":
        counts: dict[int, int] = {}
        total = 0
        for gene, value in zip(indices, values, strict=True):
            gene = int(gene)
            value = int(value)
            if value < 0:
                raise ValueError("negative counts are not allowed")
            if value == 0:
                continue
            counts[gene] = counts.get(gene, 0) + value
            total += value
        return cls(counts=counts, total=total)

    def copy(self) -> "SparseCountVector":
        indices, values = self.sorted_items()
        return SparseCountVector(
            counts=dict(self.counts),
            total=int(self.total),
            _sorted_indices=indices.copy(),
            _sorted_values=values.copy(),
        )

    @property
    def nnz(self) -> int:
        return len(self.counts)

    def invalidate(self) -> None:
        self._sorted_indices = None
        self._sorted_values = None

    def get(self, gene: int) -> int:
        return int(self.counts.get(int(gene), 0))

    def get_many(self, genes: np.ndarray) -> np.ndarray:
        return np.asarray([self.get(int(gene)) for gene in genes], dtype=np.int64)

    def sorted_items(self) -> tuple[np.ndarray, np.ndarray]:
        if self._sorted_indices is None or self._sorted_values is None:
            if not self.counts:
                self._sorted_indices = np.empty(0, dtype=np.int64)
                self._sorted_values = np.empty(0, dtype=np.int64)
            else:
                genes = np.fromiter(self.counts.keys(), dtype=np.int64, count=len(self.counts))
                order = np.argsort(genes, kind="mergesort")
                values = np.fromiter(
                    (self.counts[int(gene)] for gene in genes),
                    dtype=np.int64,
                    count=len(self.counts),
                )
                self._sorted_indices = genes[order]
                self._sorted_values = values[order]
        return self._sorted_indices, self._sorted_values

    def add_inplace(self, indices: np.ndarray, values: np.ndarray) -> list[int]:
        added_genes: list[int] = []
        delta_total = 0
        for gene, value in zip(indices, values, strict=True):
            gene = int(gene)
            value = int(value)
            if value < 0:
                raise ValueError("negative counts are not allowed")
            if value == 0:
                continue
            previous = self.counts.get(gene, 0)
            if previous == 0:
                added_genes.append(gene)
            self.counts[gene] = previous + value
            delta_total += value
        self.total += delta_total
        self.invalidate()
        return added_genes

    def add_vector_inplace(self, other: "SparseCountVector") -> list[int]:
        indices, values = other.sorted_items()
        return self.add_inplace(indices, values)

    def subtract_inplace(self, indices: np.ndarray, values: np.ndarray) -> list[int]:
        removed_genes: list[int] = []
        delta_total = 0
        for gene, value in zip(indices, values, strict=True):
            gene = int(gene)
            value = int(value)
            if value < 0:
                raise ValueError("negative counts are not allowed")
            if value == 0:
                continue
            previous = self.counts.get(gene, 0)
            updated = previous - value
            if updated < 0:
                raise ValueError(
                    f"subtracting {value} from gene {gene} would create a negative count",
                )
            if updated == 0:
                if gene in self.counts:
                    del self.counts[gene]
                removed_genes.append(gene)
            else:
                self.counts[gene] = updated
            delta_total += value
        self.total -= delta_total
        if self.total < 0:
            raise ValueError("cluster total became negative")
        self.invalidate()
        return removed_genes


class PartitionState:
    def __init__(
        self,
        X: sparse.csr_matrix,
        z: np.ndarray,
        clusters: dict[int, SparseCountVector],
        cells_by_cluster: dict[int, CellMembership],
        active_cluster_ids: set[int],
        next_cluster_id: int,
        gene_to_clusters: dict[int, set[int]],
        gene_cluster_counts: np.ndarray,
        *,
        version: int = 0,
        last_touched_clusters: Iterable[int] | None = None,
        last_touched_genes: np.ndarray | None = None,
    ) -> None:
        self.X = X
        self.z = z.astype(np.int64, copy=True)
        self.n_cells, self.n_genes = self.X.shape
        self.clusters = clusters
        self.cells_by_cluster = cells_by_cluster
        self.active_cluster_ids = active_cluster_ids
        self.next_cluster_id = int(next_cluster_id)
        self.gene_to_clusters = gene_to_clusters
        self.gene_cluster_counts = np.asarray(gene_cluster_counts, dtype=np.int64).copy()
        self.cell_totals = np.asarray(self.X.sum(axis=1)).ravel().astype(np.int64, copy=False)
        self.cell_nnz = np.diff(self.X.indptr).astype(np.int64, copy=False)
        self.version = int(version)
        self.last_touched_clusters = set(
            active_cluster_ids if last_touched_clusters is None else (int(x) for x in last_touched_clusters),
        )
        self.last_touched_genes = (
            np.arange(self.n_genes, dtype=np.int64)
            if last_touched_genes is None
            else np.asarray(last_touched_genes, dtype=np.int64)
        )
        self._likelihood_cache: dict[int, float] = {}
        self._likelihood_cache_psi_id: int | None = None
        self._cached_total_log_likelihood: float | None = None

    @classmethod
    def from_csr(
        cls,
        X: sparse.csr_matrix,
        init: str | np.ndarray = "singletons",
        *,
        seed: int | None = None,
        n_clusters: int | None = None,
    ) -> "PartitionState":
        if not sparse.isspmatrix_csr(X):
            raise TypeError("X must be a scipy CSR matrix")
        X = X.copy()
        X.sum_duplicates()
        X.sort_indices()
        if np.any(X.data < 0):
            raise ValueError("negative counts are not allowed")
        X.data = X.data.astype(np.int64, copy=False)

        if isinstance(init, str):
            if init == "singletons":
                z = np.arange(X.shape[0], dtype=np.int64)
            elif init == "one_cluster":
                z = np.zeros(X.shape[0], dtype=np.int64)
            elif init == "random":
                rng = np.random.default_rng(seed)
                if n_clusters is None:
                    n_clusters = max(1, min(X.shape[0], int(np.sqrt(max(X.shape[0], 1)))))
                z = rng.integers(0, int(n_clusters), size=X.shape[0], dtype=np.int64)
            elif init == "leiden_overclustered":
                from .warm_start import overclustered_leiden_labels

                z = overclustered_leiden_labels(
                    X,
                    seed=seed,
                    target_clusters=n_clusters,
                )
            else:
                raise ValueError(f"unknown init mode: {init}")
        else:
            z = np.asarray(init, dtype=np.int64)
            if z.shape != (X.shape[0],):
                raise ValueError("z must have shape (n_cells,)")

        return cls.from_assignment(X, z)

    @classmethod
    def from_assignment(cls, X: sparse.csr_matrix, z: np.ndarray) -> "PartitionState":
        z = np.asarray(z, dtype=np.int64)
        unique, inverse = np.unique(z, return_inverse=True)
        del unique
        z = inverse.astype(np.int64, copy=False)

        clusters: dict[int, SparseCountVector] = {}
        cells_by_cluster: dict[int, CellMembership] = defaultdict(lambda: CellMembership([], {}))
        gene_to_clusters: dict[int, set[int]] = defaultdict(set)
        active_cluster_ids: set[int] = set()
        gene_cluster_counts = np.zeros(X.shape[1], dtype=np.int64)

        for cell in range(X.shape[0]):
            cluster_id = int(z[cell])
            active_cluster_ids.add(cluster_id)
            if cluster_id not in clusters:
                clusters[cluster_id] = SparseCountVector({}, 0)
            cells_by_cluster[cluster_id].add(cell)
            start = X.indptr[cell]
            end = X.indptr[cell + 1]
            indices = X.indices[start:end]
            values = X.data[start:end]
            added = clusters[cluster_id].add_inplace(indices, values)
            for gene in added:
                gene_to_clusters[gene].add(cluster_id)
                gene_cluster_counts[gene] += 1

        return cls(
            X=X,
            z=z,
            clusters=clusters,
            cells_by_cluster=dict(cells_by_cluster),
            active_cluster_ids=active_cluster_ids,
            next_cluster_id=(max(active_cluster_ids) + 1) if active_cluster_ids else 0,
            gene_to_clusters=dict(gene_to_clusters),
            gene_cluster_counts=gene_cluster_counts,
        )

    def copy(self) -> "PartitionState":
        copied = PartitionState(
            X=self.X,
            z=self.z.copy(),
            clusters={cluster_id: vector.copy() for cluster_id, vector in self.clusters.items()},
            cells_by_cluster={
                cluster_id: membership.copy()
                for cluster_id, membership in self.cells_by_cluster.items()
            },
            active_cluster_ids=set(self.active_cluster_ids),
            next_cluster_id=self.next_cluster_id,
            gene_to_clusters={
                gene: set(cluster_ids) for gene, cluster_ids in self.gene_to_clusters.items()
            },
            gene_cluster_counts=self.gene_cluster_counts.copy(),
            version=self.version,
            last_touched_clusters=set(self.last_touched_clusters),
            last_touched_genes=self.last_touched_genes.copy(),
        )
        if self._likelihood_cache_psi_id is not None:
            copied._likelihood_cache = dict(self._likelihood_cache)
            copied._likelihood_cache_psi_id = self._likelihood_cache_psi_id
            copied._cached_total_log_likelihood = self._cached_total_log_likelihood
        return copied

    def cell_counts(self, cell: int) -> tuple[np.ndarray, np.ndarray]:
        cell = int(cell)
        start = self.X.indptr[cell]
        end = self.X.indptr[cell + 1]
        return self.X.indices[start:end], self.X.data[start:end]

    def block_vector_from_cells(self, cells: Iterable[int]) -> SparseCountVector:
        vector = SparseCountVector({}, 0)
        for cell in cells:
            indices, values = self.cell_counts(int(cell))
            vector.add_inplace(indices, values)
        return vector

    def cluster_size(self, cluster_id: int) -> int:
        return len(self.cells_by_cluster[int(cluster_id)])

    def cluster_total(self, cluster_id: int) -> int:
        return int(self.clusters[int(cluster_id)].total)

    def non_singleton_cluster_ids(self) -> list[int]:
        return [cluster_id for cluster_id in self.active_cluster_ids if self.cluster_size(cluster_id) > 1]

    def active_cluster_array(self) -> np.ndarray:
        return np.asarray(sorted(self.active_cluster_ids), dtype=np.int64)

    def cluster_log_likelihood_cached(self, cluster_id: int, psi: np.ndarray) -> float:
        self._ensure_likelihood_cache_identity(psi)
        cluster_id = int(cluster_id)
        cached = self._likelihood_cache.get(cluster_id)
        if cached is not None:
            return cached
        from .likelihood import cluster_log_likelihood

        value = float(cluster_log_likelihood(self.clusters[cluster_id], psi))
        self._likelihood_cache[cluster_id] = value
        if self._cached_total_log_likelihood is not None:
            self._cached_total_log_likelihood += value
        return value

    def total_log_likelihood_cached(self, psi: np.ndarray) -> float:
        self._ensure_likelihood_cache_identity(psi)
        if self._cached_total_log_likelihood is None:
            self._cached_total_log_likelihood = float(
                sum(self.cluster_log_likelihood_cached(cluster_id, psi) for cluster_id in self.active_cluster_ids),
            )
        return float(self._cached_total_log_likelihood)

    def initialize_likelihood_cache(self, psi: np.ndarray) -> None:
        from .likelihood import cluster_log_likelihood

        self._likelihood_cache = {
            int(cluster_id): float(cluster_log_likelihood(vector, psi))
            for cluster_id, vector in self.clusters.items()
        }
        self._likelihood_cache_psi_id = id(psi)
        self._cached_total_log_likelihood = float(sum(self._likelihood_cache.values()))

    def clear_likelihood_cache(self) -> None:
        self._likelihood_cache = {}
        self._likelihood_cache_psi_id = None
        self._cached_total_log_likelihood = None

    def _ensure_likelihood_cache_identity(self, psi: np.ndarray) -> None:
        if self._likelihood_cache_psi_id != id(psi):
            self.initialize_likelihood_cache(psi)

    def _invalidate_likelihood_entries(self, cluster_ids: Iterable[int]) -> None:
        if self._likelihood_cache_psi_id is None:
            return
        for cluster_id in cluster_ids:
            cluster_id = int(cluster_id)
            self._likelihood_cache.pop(cluster_id, None)
        self._cached_total_log_likelihood = None

    def _note_state_change(self, touched_clusters: Iterable[int], touched_genes: Iterable[int]) -> None:
        self.version += 1
        self.last_touched_clusters = {int(cluster_id) for cluster_id in touched_clusters}
        self.last_touched_genes = np.asarray(sorted({int(gene) for gene in touched_genes}), dtype=np.int64)

    def _remove_cluster_gene_links(self, cluster_id: int, genes: Iterable[int]) -> None:
        cluster_id = int(cluster_id)
        for gene in genes:
            gene = int(gene)
            cluster_ids = self.gene_to_clusters.get(gene)
            if cluster_ids is None:
                continue
            if cluster_id in cluster_ids:
                cluster_ids.remove(cluster_id)
                self.gene_cluster_counts[gene] -= 1
            if not cluster_ids:
                del self.gene_to_clusters[gene]

    def _add_cluster_gene_links(self, cluster_id: int, genes: Iterable[int]) -> None:
        cluster_id = int(cluster_id)
        for gene in genes:
            gene = int(gene)
            cluster_ids = self.gene_to_clusters.setdefault(gene, set())
            if cluster_id not in cluster_ids:
                cluster_ids.add(cluster_id)
                self.gene_cluster_counts[gene] += 1

    def _drop_cluster(self, cluster_id: int) -> None:
        cluster_id = int(cluster_id)
        if cluster_id not in self.active_cluster_ids:
            return
        vector = self.clusters[cluster_id]
        self._remove_cluster_gene_links(cluster_id, vector.counts.keys())
        del self.clusters[cluster_id]
        del self.cells_by_cluster[cluster_id]
        self.active_cluster_ids.remove(cluster_id)
        self._invalidate_likelihood_entries([cluster_id])

    def merge_clusters(self, cluster_a: int, cluster_b: int) -> int:
        cluster_a = int(cluster_a)
        cluster_b = int(cluster_b)
        if cluster_a == cluster_b:
            raise ValueError("cannot merge a cluster with itself")
        if cluster_a not in self.active_cluster_ids or cluster_b not in self.active_cluster_ids:
            raise KeyError("merge requested on an inactive cluster")

        vector_a = self.clusters[cluster_a]
        vector_b = self.clusters[cluster_b]
        block_genes = set(vector_b.counts.keys())
        added_genes = vector_a.add_vector_inplace(vector_b)
        self._add_cluster_gene_links(cluster_a, added_genes)
        self._remove_cluster_gene_links(cluster_b, vector_b.counts.keys())

        membership_b = self.cells_by_cluster[cluster_b]
        for cell in membership_b.cells:
            self.z[cell] = cluster_a
        self.cells_by_cluster[cluster_a].extend(membership_b)

        del self.clusters[cluster_b]
        del self.cells_by_cluster[cluster_b]
        self.active_cluster_ids.remove(cluster_b)
        self._invalidate_likelihood_entries([cluster_a, cluster_b])
        self._note_state_change([cluster_a, cluster_b], block_genes.union(added_genes))
        return cluster_a

    def peel_cell_to_new_cluster(self, cell: int, source_cluster: int) -> int:
        cell = int(cell)
        source_cluster = int(source_cluster)
        if self.z[cell] != source_cluster:
            raise ValueError("cell does not belong to the requested source cluster")
        if self.cluster_size(source_cluster) <= 1:
            raise ValueError("cannot peel from a singleton cluster")

        indices, values = self.cell_counts(cell)
        source_vector = self.clusters[source_cluster]
        removed_genes = source_vector.subtract_inplace(indices, values)
        self._remove_cluster_gene_links(source_cluster, removed_genes)
        self.cells_by_cluster[source_cluster].remove(cell)

        new_cluster = self.next_cluster_id
        self.next_cluster_id += 1
        singleton = SparseCountVector.from_indices(indices, values)
        self.clusters[new_cluster] = singleton
        self.cells_by_cluster[new_cluster] = CellMembership.from_cells([cell])
        self.active_cluster_ids.add(new_cluster)
        self._add_cluster_gene_links(new_cluster, singleton.counts.keys())
        self.z[cell] = new_cluster
        self._invalidate_likelihood_entries([source_cluster, new_cluster])
        self._note_state_change([source_cluster, new_cluster], indices)
        return new_cluster

    def move_cell(self, cell: int, source_cluster: int, target_cluster: int) -> None:
        cell = int(cell)
        source_cluster = int(source_cluster)
        target_cluster = int(target_cluster)
        if source_cluster == target_cluster:
            raise ValueError("source and target clusters must differ")
        if self.z[cell] != source_cluster:
            raise ValueError("cell does not belong to the requested source cluster")
        if source_cluster not in self.active_cluster_ids or target_cluster not in self.active_cluster_ids:
            raise KeyError("move requested on an inactive cluster")

        indices, values = self.cell_counts(cell)
        source_vector = self.clusters[source_cluster]
        removed_genes = source_vector.subtract_inplace(indices, values)
        self._remove_cluster_gene_links(source_cluster, removed_genes)
        self.cells_by_cluster[source_cluster].remove(cell)

        target_vector = self.clusters[target_cluster]
        added_genes = target_vector.add_inplace(indices, values)
        self._add_cluster_gene_links(target_cluster, added_genes)
        self.cells_by_cluster[target_cluster].add(cell)
        self.z[cell] = target_cluster

        if self.cluster_size(source_cluster) == 0:
            del self.clusters[source_cluster]
            del self.cells_by_cluster[source_cluster]
            self.active_cluster_ids.remove(source_cluster)
        self._invalidate_likelihood_entries([source_cluster, target_cluster])
        self._note_state_change([source_cluster, target_cluster], indices)

    def peel_block_to_new_cluster(
        self,
        cells: Iterable[int],
        source_cluster: int,
        *,
        block_indices: np.ndarray | None = None,
        block_values: np.ndarray | None = None,
    ) -> int:
        selected = tuple(sorted(int(cell) for cell in cells))
        source_cluster = int(source_cluster)
        if not selected:
            raise ValueError("cannot peel an empty block")
        if len(selected) >= self.cluster_size(source_cluster):
            raise ValueError("cannot peel an entire cluster into a new cluster")
        for cell in selected:
            if self.z[cell] != source_cluster:
                raise ValueError("a requested cell does not belong to the source cluster")

        if block_indices is None or block_values is None:
            block_vector = self.block_vector_from_cells(selected)
        else:
            block_vector = SparseCountVector.from_indices(block_indices, block_values)
        block_indices, block_values = block_vector.sorted_items()
        source_vector = self.clusters[source_cluster]
        removed_genes = source_vector.subtract_inplace(block_indices, block_values)
        self._remove_cluster_gene_links(source_cluster, removed_genes)
        for cell in selected:
            self.cells_by_cluster[source_cluster].remove(cell)

        new_cluster = self.next_cluster_id
        self.next_cluster_id += 1
        self.clusters[new_cluster] = block_vector
        self.cells_by_cluster[new_cluster] = CellMembership.from_cells(selected)
        self.active_cluster_ids.add(new_cluster)
        self._add_cluster_gene_links(new_cluster, block_vector.counts.keys())
        for cell in selected:
            self.z[cell] = new_cluster
        self._invalidate_likelihood_entries([source_cluster, new_cluster])
        self._note_state_change([source_cluster, new_cluster], block_indices)
        return new_cluster

    def move_block(
        self,
        cells: Iterable[int],
        source_cluster: int,
        target_cluster: int,
        *,
        block_indices: np.ndarray | None = None,
        block_values: np.ndarray | None = None,
    ) -> None:
        selected = tuple(sorted(int(cell) for cell in cells))
        source_cluster = int(source_cluster)
        target_cluster = int(target_cluster)
        if source_cluster == target_cluster:
            raise ValueError("source and target clusters must differ")
        if not selected:
            raise ValueError("cannot move an empty block")
        if source_cluster not in self.active_cluster_ids or target_cluster not in self.active_cluster_ids:
            raise KeyError("move requested on an inactive cluster")
        for cell in selected:
            if self.z[cell] != source_cluster:
                raise ValueError("a requested cell does not belong to the source cluster")

        if block_indices is None or block_values is None:
            block_vector = self.block_vector_from_cells(selected)
        else:
            block_vector = SparseCountVector.from_indices(block_indices, block_values)
        block_indices, block_values = block_vector.sorted_items()
        source_vector = self.clusters[source_cluster]
        removed_genes = source_vector.subtract_inplace(block_indices, block_values)
        self._remove_cluster_gene_links(source_cluster, removed_genes)
        for cell in selected:
            self.cells_by_cluster[source_cluster].remove(cell)

        target_vector = self.clusters[target_cluster]
        added_genes = target_vector.add_inplace(block_indices, block_values)
        self._add_cluster_gene_links(target_cluster, added_genes)
        for cell in selected:
            self.cells_by_cluster[target_cluster].add(cell)
            self.z[cell] = target_cluster

        if self.cluster_size(source_cluster) == 0:
            del self.clusters[source_cluster]
            del self.cells_by_cluster[source_cluster]
            self.active_cluster_ids.remove(source_cluster)
        self._invalidate_likelihood_entries([source_cluster, target_cluster])
        self._note_state_change([source_cluster, target_cluster], block_indices)

    def validate(self) -> None:
        if self.z.shape != (self.n_cells,):
            raise AssertionError("z has the wrong shape")
        if set(self.clusters) != self.active_cluster_ids:
            raise AssertionError("cluster storage and active cluster ids disagree")
        if set(self.cells_by_cluster) != self.active_cluster_ids:
            raise AssertionError("cell memberships and active cluster ids disagree")

        reconstructed_z = np.empty_like(self.z)
        for cluster_id in self.active_cluster_ids:
            membership = self.cells_by_cluster[cluster_id]
            if len(membership) == 0:
                raise AssertionError(f"cluster {cluster_id} is empty")
            if len(membership.positions) != len(membership.cells):
                raise AssertionError(f"cluster {cluster_id} membership positions are inconsistent")
            for idx, cell in enumerate(membership.cells):
                if membership.positions[cell] != idx:
                    raise AssertionError(f"cluster {cluster_id} membership indexing is inconsistent")
                reconstructed_z[cell] = cluster_id
                if self.z[cell] != cluster_id:
                    raise AssertionError(f"z disagrees with membership for cell {cell}")

            rebuilt = SparseCountVector({}, 0)
            for cell in membership.cells:
                indices, values = self.cell_counts(cell)
                rebuilt.add_inplace(indices, values)
            vector = self.clusters[cluster_id]
            if rebuilt.total != vector.total or rebuilt.counts != vector.counts:
                raise AssertionError(f"cluster {cluster_id} counts are inconsistent")
            if vector.total < 0:
                raise AssertionError(f"cluster {cluster_id} total is negative")

        if not np.array_equal(reconstructed_z, self.z):
            raise AssertionError("reconstructed z disagrees with stored z")

        rebuilt_gene_to_clusters: dict[int, set[int]] = defaultdict(set)
        rebuilt_gene_cluster_counts = np.zeros(self.n_genes, dtype=np.int64)
        for cluster_id, vector in self.clusters.items():
            for gene in vector.counts:
                rebuilt_gene_to_clusters[gene].add(cluster_id)
        for gene, cluster_ids in rebuilt_gene_to_clusters.items():
            rebuilt_gene_cluster_counts[gene] = len(cluster_ids)
        if rebuilt_gene_to_clusters != self.gene_to_clusters:
            raise AssertionError("gene-to-cluster index is inconsistent")
        if not np.array_equal(rebuilt_gene_cluster_counts, self.gene_cluster_counts):
            raise AssertionError("gene cluster counts are inconsistent")
