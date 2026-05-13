from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse


def overclustered_leiden_labels(
    X: sparse.csr_matrix,
    *,
    seed: int | None = None,
    target_clusters: int | None = None,
    k_neighbors: int = 16,
    top_genes_per_cell: int = 24,
    prefer_external: bool = True,
    resolution_grid: tuple[float, ...] | None = None,
    max_local_passes: int = 8,
    max_outer_rounds: int = 6,
) -> np.ndarray:
    if not sparse.isspmatrix_csr(X):
        raise TypeError("X must be a CSR matrix")
    n_cells = int(X.shape[0])
    if n_cells == 0:
        return np.empty(0, dtype=np.int64)
    if n_cells == 1:
        return np.zeros(1, dtype=np.int64)

    graph_neighbors, graph_weights = _build_overlap_graph(
        X,
        k_neighbors=int(k_neighbors),
        top_genes_per_cell=int(top_genes_per_cell),
    )
    if target_clusters is None:
        target_clusters = max(8, min(max(16, n_cells // 3), int(8.0 * np.sqrt(n_cells))))
    if prefer_external:
        labels = _external_leiden_partition(
            graph_neighbors,
            graph_weights,
            target_clusters=int(target_clusters),
            seed=seed,
            resolution_grid=resolution_grid,
        )
        if labels is not None:
            return _relabel_consecutive(labels)

    best_labels = np.arange(n_cells, dtype=np.int64)
    best_score = np.inf
    grid = _default_resolution_grid(n_cells, int(target_clusters)) if resolution_grid is None else resolution_grid
    for idx, resolution in enumerate(grid):
        local_rng = np.random.default_rng(None if seed is None else int(seed) + idx)
        labels = _leiden_style_partition(
            graph_neighbors,
            graph_weights,
            resolution=float(resolution),
            rng=local_rng,
            max_local_passes=int(max_local_passes),
            max_outer_rounds=int(max_outer_rounds),
        )
        n_found = int(np.unique(labels).size)
        score = abs(n_found - int(target_clusters)) + (0.05 if n_found < int(target_clusters) else 0.0)
        if score < best_score:
            best_labels = labels
            best_score = float(score)
        if abs(n_found - int(target_clusters)) <= max(2, int(target_clusters) // 20):
            break
    return _relabel_consecutive(best_labels)


def _external_leiden_partition(
    graph_neighbors: list[np.ndarray],
    graph_weights: list[np.ndarray],
    *,
    target_clusters: int,
    seed: int | None,
    resolution_grid: tuple[float, ...] | None,
) -> np.ndarray | None:
    try:
        import igraph as ig  # type: ignore[import-not-found]
        import leidenalg  # type: ignore[import-not-found]
    except Exception:
        return None

    n_cells = len(graph_neighbors)
    edge_weights: dict[tuple[int, int], float] = {}
    for source, (neighbors, weights) in enumerate(zip(graph_neighbors, graph_weights, strict=True)):
        for target, weight in zip(neighbors, weights, strict=True):
            target = int(target)
            if source == target:
                continue
            edge = tuple(sorted((int(source), target)))
            edge_weights[edge] = max(edge_weights.get(edge, 0.0), float(weight))
    if not edge_weights:
        return np.arange(n_cells, dtype=np.int64)

    edges = list(edge_weights)
    weights = [float(edge_weights[edge]) for edge in edges]
    graph = ig.Graph(n=n_cells, edges=edges, directed=False)
    graph.es["weight"] = weights

    grid = _default_resolution_grid(n_cells, int(target_clusters)) if resolution_grid is None else resolution_grid
    best_labels: np.ndarray | None = None
    best_score = np.inf
    for resolution in grid:
        labels = _run_leiden_once(
            leidenalg,
            graph,
            resolution=float(resolution),
            seed=seed,
        )
        if labels is None:
            continue
        n_found = int(np.unique(labels).size)
        # Prefer matching the requested overclustered scale; when tied, err on splitting.
        score = abs(n_found - int(target_clusters)) + (0.05 if n_found < int(target_clusters) else 0.0)
        if score < best_score:
            best_score = float(score)
            best_labels = labels
        if n_found >= int(target_clusters) and abs(n_found - int(target_clusters)) <= max(2, target_clusters // 20):
            break
    return best_labels


def _run_leiden_once(
    leidenalg: object,
    graph: object,
    *,
    resolution: float,
    seed: int | None,
) -> np.ndarray | None:
    partition_types = []
    for name in ("RBConfigurationVertexPartition", "CPMVertexPartition"):
        partition_type = getattr(leidenalg, name, None)
        if partition_type is not None:
            partition_types.append(partition_type)
    for partition_type in partition_types:
        kwargs = {
            "weights": "weight",
            "resolution_parameter": float(resolution),
            "n_iterations": -1,
        }
        if seed is not None:
            kwargs["seed"] = int(seed)
        try:
            partition = leidenalg.find_partition(graph, partition_type, **kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            try:
                partition = leidenalg.find_partition(graph, partition_type, **kwargs)
            except TypeError:
                kwargs.pop("n_iterations", None)
                try:
                    partition = leidenalg.find_partition(graph, partition_type, **kwargs)
                except Exception:
                    continue
            except Exception:
                continue
        except Exception:
            continue
        return np.asarray(partition.membership, dtype=np.int64)
    return None


def _default_resolution_grid(n_cells: int, target_clusters: int) -> tuple[float, ...]:
    del target_clusters
    if n_cells <= 50:
        values = np.geomspace(0.02, 8.0, 12)
    else:
        values = np.geomspace(0.01, 32.0, 16)
    values = np.unique(np.asarray([*values.tolist(), 1.0], dtype=np.float64))
    return tuple(float(value) for value in values)


def _build_overlap_graph(
    X: sparse.csr_matrix,
    *,
    k_neighbors: int,
    top_genes_per_cell: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_cells, n_genes = X.shape
    detected_per_gene = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.int64, copy=False)
    idf = np.log((n_cells + 1.0) / (detected_per_gene.astype(np.float64) + 1.0))

    cell_genes: list[np.ndarray] = []
    cell_weights: list[np.ndarray] = []
    inverted: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for cell in range(n_cells):
        start = X.indptr[cell]
        end = X.indptr[cell + 1]
        genes = X.indices[start:end]
        counts = X.data[start:end].astype(np.float64, copy=False)
        if genes.size == 0:
            selected_genes = np.empty(0, dtype=np.int64)
            selected_weights = np.empty(0, dtype=np.float64)
        else:
            weights = counts * idf[genes]
            order = np.argsort(weights)[::-1][:top_genes_per_cell]
            selected_genes = genes[order].astype(np.int64, copy=False)
            selected_weights = weights[order].astype(np.float64, copy=False)
        cell_genes.append(selected_genes)
        cell_weights.append(selected_weights)
        for gene, weight in zip(selected_genes, selected_weights, strict=True):
            if weight > 0.0:
                inverted[int(gene)].append((cell, float(weight)))

    raw_neighbors: list[dict[int, float]] = [defaultdict(float) for _ in range(n_cells)]
    for cell in range(n_cells):
        scores: dict[int, float] = defaultdict(float)
        for gene, weight in zip(cell_genes[cell], cell_weights[cell], strict=True):
            for other_cell, other_weight in inverted.get(int(gene), []):
                if other_cell == cell:
                    continue
                scores[int(other_cell)] += min(float(weight), float(other_weight))
        if scores:
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k_neighbors]
            for other_cell, score in ranked:
                raw_neighbors[cell][int(other_cell)] = float(score)

    symmetric: list[dict[int, float]] = [defaultdict(float) for _ in range(n_cells)]
    for cell, neighbors in enumerate(raw_neighbors):
        for other_cell, weight in neighbors.items():
            symmetric[cell][other_cell] = max(symmetric[cell].get(other_cell, 0.0), weight)
            symmetric[other_cell][cell] = max(symmetric[other_cell].get(cell, 0.0), weight)

    graph_neighbors: list[np.ndarray] = []
    graph_weights: list[np.ndarray] = []
    for neighbors in symmetric:
        if not neighbors:
            graph_neighbors.append(np.empty(0, dtype=np.int64))
            graph_weights.append(np.empty(0, dtype=np.float64))
            continue
        ordered = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[:k_neighbors]
        graph_neighbors.append(np.asarray([node for node, _ in ordered], dtype=np.int64))
        graph_weights.append(np.asarray([weight for _, weight in ordered], dtype=np.float64))
    return graph_neighbors, graph_weights


def _leiden_style_partition(
    graph_neighbors: list[np.ndarray],
    graph_weights: list[np.ndarray],
    *,
    resolution: float,
    rng: np.random.Generator,
    max_local_passes: int,
    max_outer_rounds: int,
) -> np.ndarray:
    n_cells = len(graph_neighbors)
    labels = np.arange(n_cells, dtype=np.int64)
    previous_labels: np.ndarray | None = None

    for _ in range(max_outer_rounds):
        labels = _local_move(
            graph_neighbors,
            graph_weights,
            labels,
            resolution=resolution,
            rng=rng,
            max_passes=max_local_passes,
        )
        refined = _split_disconnected_components(graph_neighbors, labels)
        refined = _relabel_consecutive(refined)
        if previous_labels is not None and np.array_equal(refined, previous_labels):
            break
        previous_labels = labels
        labels = refined
    return _relabel_consecutive(labels)


def _local_move(
    graph_neighbors: list[np.ndarray],
    graph_weights: list[np.ndarray],
    labels: np.ndarray,
    *,
    resolution: float,
    rng: np.random.Generator,
    max_passes: int,
) -> np.ndarray:
    labels = labels.copy()
    next_label = int(labels.max(initial=-1)) + 1
    community_sizes: dict[int, int] = defaultdict(int)
    for label in labels:
        community_sizes[int(label)] += 1

    for _ in range(max_passes):
        moved = False
        for node in rng.permutation(len(labels)):
            node = int(node)
            old_comm = int(labels[node])
            comm_weights: dict[int, float] = defaultdict(float)
            for neighbor, weight in zip(graph_neighbors[node], graph_weights[node], strict=True):
                comm_weights[int(labels[int(neighbor)])] += float(weight)
            old_weight = float(comm_weights.get(old_comm, 0.0))

            best_delta = 0.0
            best_comm: int | None = old_comm
            if community_sizes[old_comm] > 1:
                singleton_delta = -old_weight + float(resolution) * float(community_sizes[old_comm] - 1)
                if singleton_delta > best_delta + 1e-12:
                    best_delta = singleton_delta
                    best_comm = None

            for candidate_comm, weight_to_comm in comm_weights.items():
                if candidate_comm == old_comm:
                    continue
                delta = (
                    float(weight_to_comm)
                    - old_weight
                    + float(resolution) * float(community_sizes[old_comm] - 1 - community_sizes[candidate_comm])
                )
                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_comm = int(candidate_comm)

            if best_comm == old_comm:
                continue
            community_sizes[old_comm] -= 1
            if community_sizes[old_comm] == 0:
                del community_sizes[old_comm]
            if best_comm is None:
                best_comm = next_label
                next_label += 1
            labels[node] = int(best_comm)
            community_sizes[int(best_comm)] += 1
            moved = True
        if not moved:
            break
    return labels


def _split_disconnected_components(
    graph_neighbors: list[np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    refined = -np.ones_like(labels)
    members_by_comm: dict[int, list[int]] = defaultdict(list)
    for node, community in enumerate(labels):
        members_by_comm[int(community)].append(int(node))

    next_label = 0
    for community, members in members_by_comm.items():
        del community
        member_set = set(members)
        for start in members:
            if refined[start] != -1:
                continue
            stack = [int(start)]
            refined[start] = next_label
            while stack:
                node = stack.pop()
                for neighbor in graph_neighbors[node]:
                    neighbor = int(neighbor)
                    if neighbor not in member_set or refined[neighbor] != -1:
                        continue
                    refined[neighbor] = next_label
                    stack.append(neighbor)
            next_label += 1
    return refined


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64, copy=False)
