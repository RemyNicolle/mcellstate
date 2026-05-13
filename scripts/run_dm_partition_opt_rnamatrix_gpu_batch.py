from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse


def find_rna_matrices(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("RNAmatrix*.tsv") if path.is_file())


def tsv_to_csr(tsv_path: Path) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[int] = []

    with tsv_path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if len(header) < 2:
            raise ValueError(f"{tsv_path} has no cell columns")

        n_cells = len(header) - 1
        n_genes = 0
        for record in reader:
            if not record:
                continue
            if len(record) != n_cells + 1:
                raise ValueError(
                    f"{tsv_path} line {n_genes + 2} has {len(record)} columns, expected {n_cells + 1}",
                )
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


def convert_to_npz(tsv_path: Path, npz_path: Path, *, force: bool) -> Path:
    if npz_path.exists() and not force:
        return npz_path
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = tsv_to_csr(tsv_path)
    sparse.save_npz(npz_path, matrix)
    return npz_path


def run_fit(
    npz_path: Path,
    labels_path: Path,
    *,
    optimizer_mode: str,
    backend: str,
    seed: int,
    restarts: int,
    n_proposals: int,
    max_rounds: int,
    stall_rounds: int,
    improvement_window: int,
    eta: float,
    overwrite: bool,
    progress: bool,
    verbose: bool,
) -> None:
    summary_path = labels_path.with_suffix(".json")
    if labels_path.exists() and summary_path.exists() and not overwrite:
        print(f"skip existing result: {labels_path}", flush=True)
        return

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mcellstate",
        "fit",
        "--input",
        str(npz_path),
        "--output",
        str(labels_path),
        "--summary-json",
        str(summary_path),
        "--optimizer-mode",
        optimizer_mode,
        "--backend",
        backend,
        "--seed",
        str(seed),
        "--restarts",
        str(restarts),
        "--n-proposals",
        str(n_proposals),
        "--max-rounds",
        str(max_rounds),
        "--stall-rounds",
        str(stall_rounds),
        "--improvement-window",
        str(improvement_window),
        "--eta",
        str(eta),
    ]
    if progress:
        cmd.append("--progress")
    if verbose:
        cmd.append("--verbose")
    print("run:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert RNAmatrix TSV files to sparse NPZ and run mcellstate on CUDA.",
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing RNAmatrix*.tsv files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for labels and summary JSON outputs.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache/rnamatrix_npz"),
        help="Directory for converted NPZ matrices.",
    )
    parser.add_argument(
        "--optimizer-mode",
        default="gpu-heavy",
        choices=("effective", "gpu-heavy"),
        help="effective keeps the current mixed search; gpu-heavy reduces CPU-heavy refinement phases.",
    )
    parser.add_argument("--backend", default="cuda", help="mcellstate backend. Use cuda for GPU.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--n-proposals", type=int, default=4000)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--stall-rounds", type=int, default=1)
    parser.add_argument("--improvement-window", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--progress", dest="progress", action="store_true", help="Print per-round timing output.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable per-round timing output.")
    parser.set_defaults(progress=True)
    parser.add_argument("--verbose", action="store_true", help="Print every optimizer step as it runs.")
    parser.add_argument("--force-convert", action="store_true", help="Rebuild cached NPZ files.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun fits even if labels and JSON already exist.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrices = find_rna_matrices(args.input_root)
    if not matrices:
        raise SystemExit(f"no RNAmatrix*.tsv files found under {args.input_root}")

    print(f"found {len(matrices)} RNAmatrix TSV files", flush=True)
    for tsv_path in matrices:
        relative = tsv_path.relative_to(args.input_root)
        stem = tsv_path.stem
        npz_path = args.cache_root / relative.with_suffix(".npz")
        labels_path = args.output_root / relative.parent / f"{stem}.labels.npy"
        start_time = datetime.now()
        print(f"[START] {tsv_path} {start_time.isoformat()}", flush=True)
        convert_to_npz(tsv_path, npz_path, force=bool(args.force_convert))
        end_time = datetime.now()
        print(f"[END] {tsv_path} {end_time.isoformat()} duration={(end_time - start_time).total_seconds():.2f}s", flush=True)
        run_fit(
            npz_path,
            labels_path,
            optimizer_mode=str(args.optimizer_mode),
            backend=str(args.backend),
            seed=int(args.seed),
            restarts=int(args.restarts),
            n_proposals=int(args.n_proposals),
            max_rounds=int(args.max_rounds),
            stall_rounds=int(args.stall_rounds),
            improvement_window=int(args.improvement_window),
            eta=float(args.eta),
            overwrite=bool(args.overwrite),
            progress=bool(args.progress),
            verbose=bool(args.verbose),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
