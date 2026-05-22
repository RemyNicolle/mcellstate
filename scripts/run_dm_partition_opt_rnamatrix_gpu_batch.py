from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def find_rna_matrices(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("RNAmatrix*.tsv") if path.is_file())


def tsv_to_csr(tsv_path: Path) -> sparse.csr_matrix:
    # 1. Read header to find cell names and n_cells
    with tsv_path.open("r") as f:
        header = f.readline().strip().split("\t")
    if len(header) < 2:
        raise ValueError(f"{tsv_path} has no cell columns")
    n_cells = len(header) - 1

    # 2. Read in chunks of genes to keep memory low
    chunk_list = []
    # Using chunksize=5000 is memory efficient and extremely fast.
    # index_col=0 treats the first column (gene names) as row index.
    for chunk in pd.read_csv(tsv_path, sep="\t", index_col=0, chunksize=5000):
        # chunk is shape (chunk_genes, n_cells)
        # Handle NaN values by filling with 0, cast to int64, transpose to (n_cells, chunk_genes)
        dense_chunk = chunk.fillna(0.0).values.T.astype(np.int64)
        sparse_chunk = sparse.csr_matrix(dense_chunk)
        chunk_list.append(sparse_chunk)

    if not chunk_list:
        return sparse.csr_matrix((n_cells, 0), dtype=np.int64)

    # Combine all chunks column-wise (since each chunk is a subset of genes)
    matrix = sparse.hstack(chunk_list, format="csr", dtype=np.int64)
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
    preset: str,
    optimizer_mode: str | None,
    backend: str,
    proposal_workers: int | None,
    seed: int,
    restarts: int | None,
    n_proposals: int,
    max_scored_proposals: int | None,
    random_proposals: bool | None,
    random_accept_prob: float | None,
    random_accept_max_fraction: float | None,
    proposal_batch_size: int | None,
    cuda_chunk_size: int | None,
    recompute_ll_each_round: bool | None,
    cuda_empty_cache: bool,
    max_rounds: int,
    stall_rounds: int | None,
    improvement_window: int | None,
    eta: float | None,
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
        *(["--optimizer-mode", optimizer_mode] if optimizer_mode is not None else []),
        *(["--preset", preset] if optimizer_mode is None else []),
        "--backend",
        backend,
        *(
            []
            if proposal_workers is None
            else ["--proposal-workers", str(proposal_workers)]
        ),
        "--seed",
        str(seed),
        *([] if restarts is None else ["--restarts", str(restarts)]),
        "--n-proposals",
        str(n_proposals),
        *(
            []
            if max_scored_proposals is None
            else ["--max-scored-proposals", str(max_scored_proposals)]
        ),
        *(
            []
            if random_proposals is None
            else (
                ["--random-proposals"] if random_proposals else ["--guided-proposals"]
            )
        ),
        *(
            []
            if random_accept_prob is None
            else ["--random-accept-prob", str(random_accept_prob)]
        ),
        *(
            []
            if random_accept_max_fraction is None
            else ["--random-accept-max-fraction", str(random_accept_max_fraction)]
        ),
        *(
            []
            if proposal_batch_size is None
            else ["--proposal-batch-size", str(proposal_batch_size)]
        ),
        *(
            []
            if cuda_chunk_size is None
            else ["--cuda-chunk-size", str(cuda_chunk_size)]
        ),
        *(
            []
            if recompute_ll_each_round is None
            else (
                ["--recompute-ll-each-round"]
                if recompute_ll_each_round
                else ["--fast-ll-tracking"]
            )
        ),
        *(["--cuda-empty-cache"] if cuda_empty_cache else []),
        "--max-rounds",
        str(max_rounds),
        *([] if stall_rounds is None else ["--stall-rounds", str(stall_rounds)]),
        *(
            []
            if improvement_window is None
            else ["--improvement-window", str(improvement_window)]
        ),
        *([] if eta is None else ["--eta", str(eta)]),
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
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing RNAmatrix*.tsv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for labels and summary JSON outputs.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache/rnamatrix_npz"),
        help="Directory for converted NPZ matrices.",
    )
    parser.add_argument(
        "--preset",
        choices=("balanced", "cpu", "gpu", "quality", "benchmark"),
        default="gpu",
        help="Fit preset. Defaults to gpu, which now uses the structured CUDA search when backend=cuda.",
    )
    parser.add_argument(
        "--optimizer-mode",
        default=None,
        choices=("effective", "gpu", "cpu-only"),
        help="Legacy low-level override. Prefer --preset; --optimizer-mode gpu keeps the legacy weak GPU search policy.",
    )
    parser.add_argument(
        "--backend", default="cuda", help="mcellstate backend. Use cuda for GPU."
    )
    parser.add_argument(
        "--proposal-workers",
        type=int,
        default=None,
        help="Parallel proposal-family worker count. Defaults to the optimizer preset when omitted.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--restarts",
        type=int,
        default=None,
        help="Restart count. Omit to use the preset default; gpu preset currently defaults to multiple restarts.",
    )
    parser.add_argument("--n-proposals", type=int, default=100_000)
    parser.add_argument(
        "--max-scored-proposals",
        type=int,
        default=None,
        help="Optional cap after deduplication. Omit to use the preset/backend default.",
    )
    parser.add_argument(
        "--random-proposals", dest="random_proposals", action="store_true"
    )
    parser.add_argument(
        "--guided-proposals", dest="random_proposals", action="store_false"
    )
    parser.set_defaults(random_proposals=None)
    parser.add_argument("--random-accept-prob", type=float, default=None)
    parser.add_argument("--random-accept-max-fraction", type=float, default=None)
    parser.add_argument("--proposal-batch-size", type=int, default=None)
    parser.add_argument("--cuda-chunk-size", type=int, default=None)
    parser.add_argument(
        "--recompute-ll-each-round",
        dest="recompute_ll_each_round",
        action="store_true",
    )
    parser.add_argument(
        "--fast-ll-tracking",
        dest="recompute_ll_each_round",
        action="store_false",
    )
    parser.set_defaults(recompute_ll_each_round=None)
    parser.add_argument("--cuda-empty-cache", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument(
        "--stall-rounds",
        type=int,
        default=None,
        help="Stop after this many non-improving rounds. Omit to use the preset default.",
    )
    parser.add_argument(
        "--improvement-window",
        type=int,
        default=None,
        help="Relative improvement window. Omit to use the preset default.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Relative improvement threshold. Omit to use the preset default.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Print per-round timing output.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable per-round timing output.",
    )
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--verbose", action="store_true", help="Print every optimizer step as it runs."
    )
    parser.add_argument(
        "--force-convert", action="store_true", help="Rebuild cached NPZ files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun fits even if labels and JSON already exist.",
    )
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
        print(
            f"[END] {tsv_path} {end_time.isoformat()} duration={(end_time - start_time).total_seconds():.2f}s",
            flush=True,
        )
        run_fit(
            npz_path,
            labels_path,
            preset=str(args.preset),
            optimizer_mode=(
                None if args.optimizer_mode is None else str(args.optimizer_mode)
            ),
            backend=str(args.backend),
            proposal_workers=None
            if args.proposal_workers is None
            else int(args.proposal_workers),
            seed=int(args.seed),
            restarts=None if args.restarts is None else int(args.restarts),
            n_proposals=int(args.n_proposals),
            max_scored_proposals=None
            if args.max_scored_proposals is None
            else int(args.max_scored_proposals),
            random_proposals=args.random_proposals,
            random_accept_prob=None
            if args.random_accept_prob is None
            else float(args.random_accept_prob),
            random_accept_max_fraction=(
                None
                if args.random_accept_max_fraction is None
                else float(args.random_accept_max_fraction)
            ),
            proposal_batch_size=(
                None
                if args.proposal_batch_size is None
                else int(args.proposal_batch_size)
            ),
            cuda_chunk_size=None
            if args.cuda_chunk_size is None
            else int(args.cuda_chunk_size),
            recompute_ll_each_round=args.recompute_ll_each_round,
            cuda_empty_cache=bool(args.cuda_empty_cache),
            max_rounds=int(args.max_rounds),
            stall_rounds=None if args.stall_rounds is None else int(args.stall_rounds),
            improvement_window=(
                None
                if args.improvement_window is None
                else int(args.improvement_window)
            ),
            eta=None if args.eta is None else float(args.eta),
            overwrite=bool(args.overwrite),
            progress=bool(args.progress),
            verbose=bool(args.verbose),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
