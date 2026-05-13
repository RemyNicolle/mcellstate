from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmread


def normalize_count_matrix(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        data = np.asarray(matrix.data)
        if np.issubdtype(data.dtype, np.floating):
            rounded = np.rint(data)
            if not np.allclose(data, rounded):
                raise ValueError("sparse matrix contains non-integer values")
            matrix.data = rounded.astype(np.int64, copy=False)
        else:
            matrix.data = data.astype(np.int64, copy=False)
        if np.any(matrix.data < 0):
            raise ValueError("negative counts are not allowed")
        return matrix

    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("dense input must be two-dimensional")
    if np.issubdtype(array.dtype, np.floating):
        rounded = np.rint(array)
        if not np.allclose(array, rounded):
            raise ValueError("dense matrix contains non-integer values")
        array = rounded
    if np.any(array < 0):
        raise ValueError("negative counts are not allowed")
    return sparse.csr_matrix(array.astype(np.int64, copy=False))


def load_count_matrix(path: Path) -> sparse.csr_matrix:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npz"):
        matrix = sparse.load_npz(path)
    elif suffixes.endswith(".mtx") or suffixes.endswith(".mtx.gz"):
        matrix = mmread(path)
    elif suffixes.endswith(".npy"):
        matrix = np.load(path)
    elif suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        raise ValueError("TSV inputs must be converted first with `mcellstate convert`")
    else:
        raise ValueError(f"unsupported input format: {path.suffixes!r}")
    return normalize_count_matrix(matrix)


def tsv_to_csr(path: Path) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[int] = []

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if len(header) < 2:
            raise ValueError(f"{path} has no cell columns")
        n_cells = len(header) - 1
        n_genes = 0
        for record in reader:
            if not record:
                continue
            if len(record) != n_cells + 1:
                raise ValueError(f"{path} line {n_genes + 2} has {len(record)} columns, expected {n_cells + 1}")
            for cell_idx, value in enumerate(record[1:]):
                umi = int(float(value))
                if umi:
                    rows.append(cell_idx)
                    cols.append(n_genes)
                    vals.append(umi)
            n_genes += 1

    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_genes), dtype=np.int64)
    matrix.sum_duplicates()
    matrix.sort_indices()
    gene_mask = np.asarray(matrix.sum(axis=0)).ravel() > 0
    return matrix[:, gene_mask]


def convert_counts(input_path: Path, output_path: Path) -> sparse.csr_matrix:
    matrix = tsv_to_csr(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_path, matrix)
    return matrix
