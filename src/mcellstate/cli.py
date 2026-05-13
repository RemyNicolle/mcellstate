from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from mcellstate_doublet import AuditThresholds, audit_doublets as run_doublet_audit
from mcellstate_doublet import make_prior as make_doublet_prior

from . import Optimizer, PartitionState, __version__, make_prior
from .io import convert_counts, load_count_matrix
from .presets import FIT_PRESETS, resolve_fit_preset


def _auto_tau(n_umis: float, n_cells: int) -> float:
    if n_umis <= 0.0:
        return 1.0
    ratio = max(n_umis / float(max(n_cells, 1)), 1e-12)
    return float(2.0 ** round(math.log2(ratio)))


def _default_target_clusters(n_cells: int) -> int:
    return max(1, min(int(n_cells), max(16, int(2.0 * np.sqrt(max(int(n_cells), 1))))))


def _default_leiden_target(n_cells: int) -> int:
    if n_cells <= 1:
        return int(n_cells)
    target = max(128, min(int(n_cells // 3), int(8.0 * np.sqrt(max(int(n_cells), 1)))))
    return max(1, min(int(n_cells), target))


def _load_labels(path: Path) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        labels = np.load(path)
    else:
        labels = np.loadtxt(path, dtype=np.int64)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    return labels.astype(np.int64, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcellstate",
        description="Collapsed Dirichlet-multinomial partitioning and doublet auditing for scRNA-seq UMI counts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert RNAmatrix TSV input to sparse NPZ.")
    convert.add_argument("--input", required=True, type=Path, help="Input RNAmatrix TSV path.")
    convert.add_argument("--output", required=True, type=Path, help="Output sparse NPZ path.")

    fit = subparsers.add_parser("fit", help="Fit a hard partition from a raw count matrix.")
    fit.add_argument("--input", required=True, type=Path, help="Input count matrix (.npz, .mtx, or .npy).")
    fit.add_argument("--output", required=True, type=Path, help="Path to the output labels .npy file.")
    fit.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary path.")
    fit.add_argument(
        "--preset",
        choices=FIT_PRESETS,
        default="balanced",
        help="balanced is the default; cpu keeps the search on CPU; gpu reduces CPU-heavy refinement; gpu-full shifts harder toward GPU-scoreable proposals; quality is more exhaustive; benchmark matches comparison runs.",
    )
    fit.add_argument(
        "--optimizer-mode",
        choices=Optimizer.OPTIMIZER_MODES,
        default=None,
        help="Optional low-level override. Prefer --preset for normal usage.",
    )
    fit.add_argument(
        "--init",
        choices=["singletons", "one_cluster", "random", "leiden_overclustered"],
        default="leiden_overclustered",
        help="Initial partition strategy.",
    )
    fit.add_argument("--n-clusters", type=int, default=None, help="Optional initialization cluster count.")
    fit.add_argument("--tau", type=float, default=None, help="Dirichlet concentration. Auto if omitted.")
    fit.add_argument("--backend", default="auto", help="Backend to use: auto, cpu, torch-cpu, or cuda.")
    fit.add_argument("--threads", type=int, default=None, help="Thread count for CPU torch backends.")
    fit.add_argument("--proposal-workers", type=int, default=None, help="Worker count for parallel proposal-family sampling.")
    fit.add_argument("--seed", type=int, default=1, help="Random seed.")
    fit.add_argument("--restarts", type=int, default=1, help="Number of restarts to run.")
    fit.add_argument("--n-proposals", type=int, default=4000, help="Proposals per optimization round.")
    fit.add_argument("--max-rounds", type=int, default=0, help="Use 0 or negative to run until convergence.")
    fit.add_argument("--stall-rounds", type=int, default=1, help="Stop after this many non-improving rounds.")
    fit.add_argument("--improvement-window", type=int, default=5, help="Relative improvement window.")
    fit.add_argument("--eta", type=float, default=0.0, help="Relative improvement threshold.")
    fit.add_argument("--target-clusters", type=int, default=None, help="Optional staged-search target.")
    fit.add_argument("--validate-batches", action="store_true", help="Validate batch gains exactly.")
    fit.add_argument("--update-psi", action="store_true", help="Optionally update tau during optimization.")
    fit.add_argument("--progress", action="store_true", help="Print per-round timing and progress output.")
    fit.add_argument("--verbose", action="store_true", help="Print every optimizer step as it runs.")

    audit = subparsers.add_parser("audit-doublets", help="Audit an existing partition for likely doublets.")
    audit.add_argument("--input", required=True, type=Path, help="Input count matrix (.npz, .mtx, or .npy).")
    audit.add_argument("--labels", required=True, type=Path, help="Cluster labels (.npy or text).")
    audit.add_argument("--output", required=True, type=Path, help="Output cluster table TSV path.")
    audit.add_argument("--tau", type=float, default=1.0, help="Total prior mass.")
    audit.add_argument("--doublet-rate", type=float, default=0.05)
    audit.add_argument("--top-m-parents", type=int, default=50)
    audit.add_argument("--lambda-grid-size", type=int, default=101)
    audit.add_argument("--include-homotypic", action="store_true")
    audit.add_argument("--return-cell-lambda", action="store_true")
    audit.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary path.")
    return parser


def run_convert(args: argparse.Namespace) -> dict:
    matrix = convert_counts(args.input, args.output)
    return {
        "status": "completed",
        "input_path": str(args.input),
        "output_path": str(args.output),
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "total_umis": int(matrix.sum()),
    }


def run_fit(args: argparse.Namespace) -> dict:
    input_path: Path = args.input
    output_path: Path = args.output
    summary_path: Path = args.summary_json if args.summary_json is not None else output_path.with_suffix(".json")

    X = load_count_matrix(input_path)
    n_cells = int(X.shape[0])
    tau = _auto_tau(float(X.sum()), n_cells) if args.tau is None else float(args.tau)
    psi = make_prior(X, tau=tau)

    preset = resolve_fit_preset(args.preset)
    optimizer_mode = str(args.optimizer_mode or preset["optimizer_mode"])
    effective_backend = str(args.backend)
    if optimizer_mode == Optimizer.CPU_ONLY_MODE and effective_backend not in {"cpu", "numpy", "torch-cpu", "cpu-torch", "torch"}:
        effective_backend = "torch-cpu"
    target_clusters = int(args.target_clusters) if args.target_clusters is not None else _default_target_clusters(n_cells)
    leiden_target = _default_leiden_target(n_cells)
    init_n_clusters = args.n_clusters if args.n_clusters is not None else leiden_target

    state = PartitionState.from_csr(X, init=args.init, seed=args.seed, n_clusters=init_n_clusters)
    if args.verbose:
        print(
            " ".join(
                [
                    f"[mcellstate] version={__version__}",
                    f"input={input_path}",
                    f"backend={effective_backend}",
                    f"preset={args.preset}",
                    f"optimizer_mode={optimizer_mode}",
                    f"n_cells={n_cells}",
                    f"n_genes={int(X.shape[1])}",
                    f"n_proposals={int(args.n_proposals)}",
                    f"proposal_workers={args.proposal_workers if args.proposal_workers is not None else 'auto'}",
                ]
            ),
            flush=True,
        )
    optimizer_kwargs = dict(preset["optimizer_kwargs"])
    optimizer_kwargs.update(
        {
            "state": state,
            "psi": psi,
            "optimizer_mode": optimizer_mode,
            "backend": effective_backend,
            "backend_threads": args.threads,
            "proposal_workers": args.proposal_workers,
            "n_proposals": args.n_proposals,
            "seed": args.seed,
            "validate_batches": bool(args.validate_batches) or bool(optimizer_kwargs.get("validate_batches", False)),
            "staged_search": True,
            "target_clusters": target_clusters,
            "leiden_restart_targets": (leiden_target, min(n_cells, leiden_target * 2), max(16, leiden_target // 2)),
        }
    )
    optimizer = Optimizer(**optimizer_kwargs)

    fit_max_rounds = None if int(args.max_rounds) <= 0 else int(args.max_rounds)
    start = time.perf_counter()
    result = optimizer.fit(
        max_rounds=fit_max_rounds,
        update_psi=bool(args.update_psi),
        restarts=int(args.restarts),
        stall_rounds=int(args.stall_rounds),
        improvement_window=int(args.improvement_window),
        eta=float(args.eta),
        progress=bool(args.progress),
        verbose=bool(args.verbose),
    )
    elapsed = time.perf_counter() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result.z.astype(np.int32, copy=False))

    summary = {
        "status": "completed",
        "version": __version__,
        "input_path": str(input_path),
        "labels_path": str(output_path),
        "summary_json": str(summary_path),
        "elapsed_s": float(elapsed),
        "n_cells": n_cells,
        "n_genes": int(X.shape[1]),
        "n_clusters": int(len(np.unique(result.z))),
        "log_likelihood": float(result.log_likelihood),
        "preset": str(args.preset),
        "optimizer_mode": optimizer_mode,
        "backend": str(effective_backend),
        "threads": None if args.threads is None else int(args.threads),
        "proposal_workers": None if args.proposal_workers is None else int(args.proposal_workers),
        "seed": int(args.seed),
        "restarts": int(args.restarts),
        "max_rounds": fit_max_rounds,
        "tau": float(tau),
        "init": str(args.init),
        "n_proposals": int(args.n_proposals),
        "target_clusters": int(target_clusters),
        "validate_batches": bool(args.validate_batches),
        "update_psi": bool(args.update_psi),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_audit_doublets(args: argparse.Namespace) -> dict:
    X = load_count_matrix(args.input)
    labels = _load_labels(args.labels)
    psi = make_doublet_prior(X, tau=float(args.tau))
    result = run_doublet_audit(
        X,
        labels,
        psi=psi,
        doublet_rate=float(args.doublet_rate),
        top_m_parents=int(args.top_m_parents),
        lambda_grid_size=int(args.lambda_grid_size),
        include_homotypic=bool(args.include_homotypic),
        return_cell_lambda=bool(args.return_cell_lambda),
        thresholds=AuditThresholds(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.cluster_table.to_csv(args.output, sep="\t", index=False)
    summary_path = args.summary_json if args.summary_json is not None else args.output.with_suffix(".json")
    summary = {
        "status": "completed",
        "input_path": str(args.input),
        "labels_path": str(args.labels),
        "output_path": str(args.output),
        "summary_json": str(summary_path),
        "n_clusters": int(result.state.n_clusters),
        "n_likely_doublet": int((result.cluster_table["call"] == "likely_doublet").sum()),
        "n_possible_doublet": int((result.cluster_table["call"] == "possible_doublet").sum()),
        "tau": float(args.tau),
        "doublet_rate": float(args.doublet_rate),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        summary = run_convert(args)
        print(
            f"wrote {summary['output_path']} ({summary['n_cells']} cells, {summary['n_genes']} genes, {summary['nnz']} nnz)",
            flush=True,
        )
        return 0

    if args.command == "fit":
        summary = run_fit(args)
        print(
            f"wrote {summary['labels_path']} ({summary['n_clusters']} clusters, {summary['elapsed_s']:.2f}s, log-likelihood {summary['log_likelihood']:.2f})",
            flush=True,
        )
        print(f"wrote {summary['summary_json']}", flush=True)
        return 0

    if args.command == "audit-doublets":
        summary = run_audit_doublets(args)
        print(f"wrote {summary['output_path']}", flush=True)
        print(f"wrote {summary['summary_json']}", flush=True)
        return 0

    raise ValueError(f"unknown command: {args.command}")
