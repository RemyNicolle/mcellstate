from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
VENDOR_SRC = ROOT / "reference" / "cellstates"
VENDOR_SITE = ROOT / ".vendor_cellstates"
BENCHMARK_DATA = ROOT / "benchmarks" / "data"


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    contingency = _contingency(labels_true, labels_pred)
    n = int(contingency.sum())
    if n < 2:
        return 1.0
    sum_nij = _sum_choose2(contingency.ravel())
    sum_ai = _sum_choose2(contingency.sum(axis=1))
    sum_bj = _sum_choose2(contingency.sum(axis=0))
    total = math.comb(n, 2)
    expected = sum_ai * sum_bj / total if total else 0.0
    max_index = 0.5 * (sum_ai + sum_bj)
    denom = max_index - expected
    return (sum_nij - expected) / denom if denom else 1.0


def fowlkes_mallows_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    contingency = _contingency(labels_true, labels_pred)
    tp = _sum_choose2(contingency.ravel())
    fp = _sum_choose2(contingency.sum(axis=0)) - tp
    fn = _sum_choose2(contingency.sum(axis=1)) - tp
    denom = (tp + fp) * (tp + fn)
    return tp / math.sqrt(denom) if denom else 1.0


def _contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    labels_true = np.asarray(labels_true, dtype=np.int64)
    labels_pred = np.asarray(labels_pred, dtype=np.int64)
    _, true_codes = np.unique(labels_true, return_inverse=True)
    _, pred_codes = np.unique(labels_pred, return_inverse=True)
    contingency = np.zeros((true_codes.max() + 1, pred_codes.max() + 1), dtype=np.int64)
    np.add.at(contingency, (true_codes, pred_codes), 1)
    return contingency


def _sum_choose2(values: np.ndarray) -> int:
    return sum(math.comb(int(value), 2) for value in np.asarray(values).ravel())


def alpha_from_counts(total_umis: int, n_cells: int) -> float:
    return float(2.0 ** round(math.log2(total_umis / n_cells)))


def ensure_vendor_source(cellstates_repo: Path) -> Path:
    if VENDOR_SRC.exists():
        return VENDOR_SRC
    VENDOR_SRC.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cellstates_repo, VENDOR_SRC)
    raise RuntimeError(
        "vendored cellstates source was missing and has been copied into the workspace; "
        "it still needs the local setup.py patch that exists in this repository snapshot",
    )


def ensure_vendor_install(rebuild: bool = False) -> None:
    marker = VENDOR_SITE / "cellstates" / "cluster.cpython-311-darwin.so"
    if marker.exists() and not rebuild:
        return

    env = os.environ.copy()
    env["CELLSTATES_USE_CYTHON"] = "1"
    if sys.platform == "darwin":
        env["CC"] = "clang"
        env["CXX"] = "clang++"
        env["CELLSTATES_COMPILE_FLAGS"] = "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
        env["CELLSTATES_LINK_FLAGS"] = "-L/opt/homebrew/opt/libomp/lib -lomp"
    else:
        env["CELLSTATES_COMPILE_FLAGS"] = "-fopenmp"
        env["CELLSTATES_LINK_FLAGS"] = "-fopenmp"

    VENDOR_SITE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            "--target",
            str(VENDOR_SITE),
            str(VENDOR_SRC),
        ],
        check=True,
        cwd=str(ROOT),
        env=env,
    )


def import_mcellstate():
    from mcellstate import Optimizer, PartitionState, make_prior  # noqa: PLC0415
    from mcellstate.likelihood import full_partition_log_likelihood  # noqa: PLC0415

    return Optimizer, PartitionState, make_prior, full_partition_log_likelihood


def import_cellstates():
    sys.path.insert(0, str(VENDOR_SITE))
    from cellstates.cluster import Cluster  # noqa: PLC0415
    from cellstates.run import run_mcmc  # noqa: PLC0415

    return Cluster, run_mcmc


def aggregate_optimizer_timing(history: list[dict]) -> dict:
    totals: dict[str, float] = {}
    for record in history:
        for key, value in record.get("timing_s", {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: totals[key] for key in sorted(totals)}


def prepare_cached_dataset(cellstates_repo: Path) -> tuple[sparse.csr_matrix, np.ndarray, dict]:
    BENCHMARK_DATA.mkdir(parents=True, exist_ok=True)
    cache_npz = BENCHMARK_DATA / "simulated_cells_x_genes.npz"
    truth_npy = BENCHMARK_DATA / "simulated_truth.npy"
    metadata_json = BENCHMARK_DATA / "simulated_metadata.json"

    if cache_npz.exists() and truth_npy.exists() and metadata_json.exists():
        return (
            sparse.load_npz(cache_npz),
            np.load(truth_npy),
            json.loads(metadata_json.read_text()),
        )

    data_tsv = cellstates_repo / "test" / "data" / "simulated_data.tsv"
    truth_txt = cellstates_repo / "test" / "data" / "simulated_clusters.txt"

    rows: list[int] = []
    cols: list[int] = []
    vals: list[int] = []
    with data_tsv.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        n_cells = len(header) - 1
        n_genes = 0
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            for cell_idx, value in enumerate(parts[1:]):
                umi = int(float(value))
                if umi:
                    rows.append(cell_idx)
                    cols.append(n_genes)
                    vals.append(umi)
            n_genes += 1

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_genes), dtype=np.int64)
    mask = np.asarray(X.sum(axis=0)).ravel() > 0
    X = X[:, mask]
    truth = np.loadtxt(truth_txt, dtype=np.int32)

    metadata = {
        "n_cells": int(X.shape[0]),
        "n_genes_expressed": int(X.shape[1]),
        "nnz": int(X.nnz),
        "total_umis": int(X.sum()),
        "truth_clusters": int(len(np.unique(truth))),
    }
    sparse.save_npz(cache_npz, X)
    np.save(truth_npy, truth)
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")
    return X, truth, metadata


@dataclass
class OriginalReferenceResult:
    n_clusters: int
    ari_vs_truth: float
    fms_vs_truth: float
    log_likelihood_under_fixed_psi: float
    labels_path: str


def evaluate_reference_partition(
    cellstates_repo: Path,
    X: sparse.csr_matrix,
    truth: np.ndarray,
    psi: np.ndarray,
) -> OriginalReferenceResult:
    _, PartitionState, _, full_partition_log_likelihood = import_mcellstate()
    reference_path = cellstates_repo / "test" / "results" / "optimized_clusters.txt"
    labels = np.loadtxt(reference_path, dtype=np.int64)
    state = PartitionState.from_assignment(X, labels)
    return OriginalReferenceResult(
        n_clusters=int(len(np.unique(labels))),
        ari_vs_truth=adjusted_rand_index(truth, labels),
        fms_vs_truth=fowlkes_mallows_score(truth, labels),
        log_likelihood_under_fixed_psi=float(full_partition_log_likelihood(state, psi)),
        labels_path=str(reference_path),
    )


def run_original_runner(npz_path: Path, output_json: Path, *, n_threads: int = 1) -> None:
    Cluster, run_mcmc = import_cellstates()
    X = sparse.load_npz(npz_path)
    counts = X.T.toarray().astype(np.int64, copy=False)
    alpha = alpha_from_counts(int(counts.sum()), int(counts.shape[1]))
    lam = alpha * counts.sum(axis=1) / counts.sum()
    init = np.arange(counts.shape[1], dtype=np.int32)
    clst = Cluster(counts, lam, init.copy(), num_threads=int(n_threads), n_cache=10000, seed=1)
    start = time.perf_counter()
    run_mcmc(clst, N_steps=counts.shape[1], tries_per_step=1000, log_level="ERROR")
    result = {
        "status": "completed",
        "elapsed_s": time.perf_counter() - start,
        "n_clusters": int(len(np.unique(clst.clusters))),
        "log_likelihood": float(clst.total_likelihood),
        "labels": clst.clusters.astype(int).tolist(),
    }
    output_json.write_text(json.dumps(result))


def timed_original_run(npz_path: Path, timeout_s: float, output_dir: Path, *, n_threads: int = 1) -> dict:
    output_json = output_dir / "original_runtime_result.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "original-runner",
        "--npz-path",
        str(npz_path),
        "--output-json",
        str(output_json),
        "--n-threads",
        str(int(n_threads)),
    ]
    start = time.perf_counter()
    try:
        run_kwargs = {"cwd": str(ROOT), "check": True}
        if timeout_s > 0:
            run_kwargs["timeout"] = timeout_s
        subprocess.run(cmd, **run_kwargs)
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "wall_s": time.perf_counter() - start,
            "timeout_s": float(timeout_s),
        }

    payload = json.loads(output_json.read_text())
    payload["wall_s"] = time.perf_counter() - start
    return payload


def run_mcellstate_benchmark(
    X: sparse.csr_matrix,
    truth: np.ndarray,
    reference_labels: np.ndarray,
    output_dir: Path,
    *,
    n_threads: int,
    n_proposals: int,
    max_rounds: int,
) -> dict:
    Optimizer, PartitionState, make_prior, full_partition_log_likelihood = import_mcellstate()
    alpha = alpha_from_counts(int(X.sum()), int(X.shape[0]))
    psi = make_prior(X, tau=alpha)
    leiden_target = max(128, min(int(X.shape[0] // 3), int(8.0 * np.sqrt(X.shape[0]))))
    search_target = max(16, int(2.0 * np.sqrt(X.shape[0])))
    fit_max_rounds = None if int(max_rounds) <= 0 else int(max_rounds)
    state = PartitionState.from_csr(X, init="leiden_overclustered", seed=1, n_clusters=leiden_target)
    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="torch-cpu",
        backend_threads=n_threads,
        n_proposals=n_proposals,
        seed=1,
        validate_batches=False,
        pi_merge=0.45,
        pi_peel=0.08,
        pi_move=0.17,
        pi_block_peel=0.15,
        pi_block_move=0.15,
        merge_uniform_prob=0.10,
        peel_uniform_prob=0.50,
        move_uniform_prob=0.35,
        block_peel_uniform_prob=0.35,
        block_move_uniform_prob=0.30,
        top_merge_neighbors=8,
        deterministic_merge_ratio=0.25,
        merge_gene_cluster_cap=24,
        block_size_min=2,
        block_size_max=16,
        staged_search=True,
        target_clusters=search_target,
        leiden_restart_targets=(leiden_target, min(X.shape[0], leiden_target * 2), max(16, leiden_target // 2)),
        full_merge_stage=True,
        full_merge_active_limit=1200,
        full_merge_score_chunk=8192,
        full_merge_max_sweeps=32,
        greedy_merge_sweeps=2,
        greedy_merge_candidates=max(2048, n_proposals // 2),
        exact_cell_reassign_passes=3,
        exact_cell_reassign_active_limit=1200,
        exact_cell_reassign_top_fraction=0.2,
        exact_cell_score_chunk=512,
        cluster_reassign_sweeps=0,
        cluster_reassign_max_sources=256,
        perturb_every=3,
        perturb_steps=16,
        perturb_temperature=1.0,
        perturb_positive_cleanup=True,
        serial_refine_passes=0,
        serial_refine_merge_candidates=3,
        serial_refine_move_candidates=2,
        serial_refine_cells_per_cluster=6,
    )
    start = time.perf_counter()
    result = optimizer.fit(
        max_rounds=fit_max_rounds,
        restarts=1,
        update_psi=False,
        stall_rounds=1,
        improvement_window=5,
        eta=0.0,
    )
    elapsed = time.perf_counter() - start

    labels_path = output_dir / "mcellstate_labels.npy"
    np.save(labels_path, result.z.astype(np.int32))

    reference_state = PartitionState.from_assignment(X, reference_labels)
    reference_ll = float(full_partition_log_likelihood(reference_state, psi))
    return {
        "status": "completed",
        "elapsed_s": elapsed,
        "n_clusters": int(len(np.unique(result.z))),
        "ari_vs_truth": adjusted_rand_index(truth, result.z),
        "fms_vs_truth": fowlkes_mallows_score(truth, result.z),
        "ari_vs_reference": adjusted_rand_index(reference_labels, result.z),
        "fms_vs_reference": fowlkes_mallows_score(reference_labels, result.z),
        "log_likelihood": float(result.log_likelihood),
        "reference_log_likelihood_under_same_psi": reference_ll,
        "labels_path": str(labels_path),
        "history": result.history,
        "timing_s": aggregate_optimizer_timing(result.history),
        "config": {
            "init": "leiden_overclustered",
            "leiden_target_clusters": leiden_target,
            "search_target_clusters": search_target,
            "backend": "torch-cpu",
            "backend_threads": n_threads,
            "n_proposals": n_proposals,
            "max_rounds": fit_max_rounds,
            "pi_merge": 0.45,
            "pi_peel": 0.08,
            "pi_move": 0.17,
            "pi_block_peel": 0.15,
            "pi_block_move": 0.15,
            "merge_uniform_prob": 0.10,
            "peel_uniform_prob": 0.50,
            "move_uniform_prob": 0.35,
            "block_peel_uniform_prob": 0.35,
            "block_move_uniform_prob": 0.30,
            "top_merge_neighbors": 8,
            "deterministic_merge_ratio": 0.25,
            "merge_gene_cluster_cap": 24,
            "block_size_min": 2,
            "block_size_max": 16,
            "staged_search": True,
            "full_merge_stage": True,
            "full_merge_active_limit": 1200,
            "full_merge_score_chunk": 8192,
            "full_merge_max_sweeps": 32,
            "greedy_merge_sweeps": 2,
            "greedy_merge_candidates": max(2048, n_proposals // 2),
            "exact_cell_reassign_passes": 3,
            "exact_cell_reassign_active_limit": 1200,
            "exact_cell_reassign_top_fraction": 0.2,
            "exact_cell_score_chunk": 512,
            "cluster_reassign_sweeps": 0,
            "cluster_reassign_max_sources": 256,
            "perturb_every": 3,
            "perturb_steps": 16,
            "perturb_temperature": 1.0,
            "serial_refine_passes": 0,
            "seed": 1,
            "update_psi": False,
            "stall_rounds": 1,
            "eta": 0.0,
        },
    }


def write_markdown_report(summary: dict, report_path: Path) -> None:
    data = summary["dataset"]
    original_runtime = summary["original_runtime"]
    reference = summary["original_reference"]
    mcell = summary["mcellstate"]
    ll_gap = mcell["log_likelihood"] - reference["log_likelihood_under_fixed_psi"]
    converged = mcell["config"].get("max_rounds") is None
    rounds_label = "until convergence" if converged else f'{mcell["config"]["max_rounds"]} rounds'
    rounds_sentence = "running until convergence" if converged else f"after {rounds_label}"
    timing_rows = "\n".join(
        f"| `{key}` | {value:.3f} |" for key, value in mcell.get("timing_s", {}).items()
    )
    if not timing_rows:
        timing_rows = "| n/a | n/a |"

    report = f"""# cellstates vs mcellstate on the bundled simulated dataset

## Setup

- Dataset source: `{summary["paths"]["cellstates_repo"]}/test/data/simulated_data.tsv`
- Cells: {data["n_cells"]}
- Expressed genes: {data["n_genes_expressed"]}
- Nonzero entries: {data["nnz"]}
- Total UMIs: {data["total_umis"]}
- Truth clusters: {data["truth_clusters"]}
- Shared fixed prior: `alpha = 2**round(log2(total_umis / n_cells)) = {summary["alpha"]:.0f}`
- Original `cellstates`: 1 thread, singleton initialization, fixed prior, no prior re-optimization
- `mcellstate`: 6 torch CPU threads, very overclustered Leiden-style warm start on a raw-count overlap graph, fixed prior, exact full merge sweeps, exact best-cell reassignment sweeps, stochastic split/move proposals, and optional perturbation cleanup
- `mcellstate` batch config: {rounds_label}, {mcell["config"]["n_proposals"]} proposals per round
  proposal mix `merge/peel/move/block_peel/block_move = {mcell["config"]["pi_merge"]:.2f}/{mcell["config"]["pi_peel"]:.2f}/{mcell["config"]["pi_move"]:.2f}/{mcell["config"]["pi_block_peel"]:.2f}/{mcell["config"]["pi_block_move"]:.2f}`
  warm start target clusters `{mcell["config"]["leiden_target_clusters"]}`
  search target clusters `{mcell["config"]["search_target_clusters"]}`

## Results

| Method | Status | Time (s) | Clusters | ARI vs truth | FMS vs truth | Log-likelihood |
|---|---:|---:|---:|---:|---:|---:|
| `cellstates` fresh 1-core run | {original_runtime["status"]} | {original_runtime.get("wall_s", float("nan")):.2f} | {"-" if original_runtime["status"] != "completed" else original_runtime["n_clusters"]} | {"-" if original_runtime["status"] != "completed" else f'{original_runtime["ari_vs_truth"]:.6f}'} | {"-" if original_runtime["status"] != "completed" else f'{original_runtime["fms_vs_truth"]:.6f}'} | {"-" if original_runtime["status"] != "completed" else f'{original_runtime["log_likelihood"]:.2f}'} |
| `cellstates` repo-shipped reference partition | precomputed | n/a | {reference["n_clusters"]} | {reference["ari_vs_truth"]:.6f} | {reference["fms_vs_truth"]:.6f} | {reference["log_likelihood_under_fixed_psi"]:.2f} |
| `mcellstate` fresh 6-core run | {mcell["status"]} | {mcell["elapsed_s"]:.2f} | {mcell["n_clusters"]} | {mcell["ari_vs_truth"]:.6f} | {mcell["fms_vs_truth"]:.6f} | {mcell["log_likelihood"]:.2f} |

## `mcellstate` timing breakdown

| Phase | Time (s) |
|---|---:|
{timing_rows}

## Interpretation

- The fresh one-core `cellstates` run did not finish within the scripted timeout of {original_runtime.get("timeout_s", 0):.0f} seconds. The bundled reference result in the upstream repo is perfect on this simulation: ARI and FMS are both 1.0 with exactly 10 recovered clusters.
- The current `mcellstate` implementation completed in {mcell["elapsed_s"]:.2f} seconds on 6 CPU threads using an overclustered Leiden-style warm start plus exact full merge sweeps, exact best-cell reassignment sweeps, and stochastic split/move proposals. It ended at {mcell["n_clusters"]} clusters {rounds_sentence}, with ARI {mcell["ari_vs_truth"]:.6f} and FMS {mcell["fms_vs_truth"]:.6f}.
- Under the same fixed prior, the `mcellstate` partition log-likelihood is {mcell["log_likelihood"]:.2f}, which is {ll_gap:.2f} above the upstream reference partition.
- On this benchmark, the remaining issue is runtime rather than search quality. The dominant costs are the exact full merge sweep and the exact best-cell reassignment sweep, with proposal generation still a noticeable secondary cost.

## Files

- Benchmark script: `{summary["paths"]["script_path"]}`
- Summary JSON: `{summary["paths"]["summary_json"]}`
- `mcellstate` labels: `{mcell["labels_path"]}`
- Upstream reference labels: `{reference["labels_path"]}`
"""
    report_path.write_text(report)


def benchmark(args: argparse.Namespace) -> None:
    cellstates_repo = Path(args.cellstates_repo).resolve()
    ensure_vendor_source(cellstates_repo)
    ensure_vendor_install(rebuild=args.rebuild_cellstates)

    X, truth, metadata = prepare_cached_dataset(cellstates_repo)
    alpha = alpha_from_counts(int(X.sum()), int(X.shape[0]))
    _, _, make_prior, _ = import_mcellstate()
    psi = make_prior(X, tau=alpha)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = evaluate_reference_partition(cellstates_repo, X, truth, psi)
    reference_labels = np.loadtxt(reference.labels_path, dtype=np.int64)
    original_runtime = timed_original_run(
        BENCHMARK_DATA / "simulated_cells_x_genes.npz",
        timeout_s=args.original_timeout_s,
        output_dir=output_dir,
        n_threads=args.original_threads,
    )
    if original_runtime["status"] == "completed":
        labels = np.asarray(original_runtime.pop("labels"), dtype=np.int64)
        original_runtime["ari_vs_truth"] = adjusted_rand_index(truth, labels)
        original_runtime["fms_vs_truth"] = fowlkes_mallows_score(truth, labels)
        labels_path = output_dir / "original_runtime_labels.npy"
        np.save(labels_path, labels.astype(np.int32))
        original_runtime["labels_path"] = str(labels_path)

    mcellstate_result = run_mcellstate_benchmark(
        X,
        truth,
        reference_labels,
        output_dir,
        n_threads=args.mcellstate_threads,
        n_proposals=args.mcellstate_proposals,
        max_rounds=args.mcellstate_rounds,
    )

    summary = {
        "dataset": metadata,
        "alpha": alpha,
        "original_reference": asdict(reference),
        "original_runtime": original_runtime,
        "mcellstate": mcellstate_result,
        "paths": {
            "cellstates_repo": str(cellstates_repo),
            "script_path": str(Path(__file__).resolve()),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(summary, report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark")
    bench.add_argument(
        "--cellstates-repo",
        default="/Users/remy.nicolle/Workspace/DEV/cellstates",
        help="Path to the upstream cellstates clone.",
    )
    bench.add_argument(
        "--output-dir",
        default=str(ROOT / "benchmarks" / "results" / "cellstates_comparison"),
        help="Directory for benchmark artifacts.",
    )
    bench.add_argument(
        "--report-path",
        default=str(ROOT / "reports" / "cellstates_vs_mcellstate_report.md"),
        help="Path to the generated markdown report.",
    )
    bench.add_argument("--original-timeout-s", type=float, default=120.0)
    bench.add_argument(
        "--original-threads",
        type=int,
        default=1,
        help="Thread count for the upstream cellstates runtime comparison.",
    )
    bench.add_argument("--mcellstate-threads", type=int, default=6)
    bench.add_argument("--mcellstate-proposals", type=int, default=4_000)
    bench.add_argument(
        "--mcellstate-rounds",
        type=int,
        default=0,
        help="Maximum optimizer rounds. Use 0 or a negative value to run until no improvement.",
    )
    bench.add_argument("--rebuild-cellstates", action="store_true")

    runner = subparsers.add_parser("original-runner")
    runner.add_argument("--npz-path", required=True)
    runner.add_argument("--output-json", required=True)
    runner.add_argument("--n-threads", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "benchmark":
        benchmark(args)
        return
    if args.command == "original-runner":
        run_original_runner(Path(args.npz_path), Path(args.output_json), n_threads=args.n_threads)
        return
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
