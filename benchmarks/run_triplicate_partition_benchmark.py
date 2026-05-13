from __future__ import annotations

import json
from multiprocessing import get_context
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
SRC = THIS_DIR.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compare_cellstates_simulation import (
    BENCHMARK_DATA,
    ROOT,
    adjusted_rand_index,
    aggregate_optimizer_timing,
    alpha_from_counts,
    ensure_vendor_install,
    ensure_vendor_source,
    evaluate_reference_partition,
    fowlkes_mallows_score,
    import_cellstates,
    import_mcellstate,
    prepare_cached_dataset,
)


CELLSTATES_REPO = Path("/Users/remy.nicolle/Workspace/DEV/cellstates").resolve()
OUTPUT_DIR = ROOT / "benchmarks" / "results" / "triplicate_partition_benchmark"
REPORT_PATH = ROOT / "benchmarks" / "reports" / "triplicate_partition_benchmark.md"
CELLSTATES_TIMEOUT_S = 1800.0


def run_cellstates_once(
    X: sparse.csr_matrix,
    truth: np.ndarray,
    *,
    n_threads: int,
    seed: int,
    alpha: float,
    output_dir: Path,
) -> dict:
    Cluster, run_mcmc = import_cellstates()
    counts = X.T.toarray().astype(np.int64, copy=False)
    lam = alpha * counts.sum(axis=1) / counts.sum()
    init = np.arange(counts.shape[1], dtype=np.int32)
    clst = Cluster(counts, lam, init.copy(), num_threads=n_threads, n_cache=10000, seed=seed)
    start = time.perf_counter()
    run_mcmc(clst, N_steps=counts.shape[1], tries_per_step=1000, log_level="ERROR")
    elapsed = time.perf_counter() - start
    labels = clst.clusters.astype(np.int32, copy=False)
    labels_path = output_dir / f"cellstates_{n_threads}core_seed{seed}_labels.npy"
    np.save(labels_path, labels)
    return {
        "method": f"cellstates_{n_threads}core",
        "seed": int(seed),
        "status": "completed",
        "elapsed_s": float(elapsed),
        "ari_vs_truth": float(adjusted_rand_index(truth, labels)),
        "fms_vs_truth": float(fowlkes_mallows_score(truth, labels)),
        "n_clusters": int(len(np.unique(labels))),
        "log_likelihood": float(clst.total_likelihood),
        "labels_path": str(labels_path),
    }


def _run_cellstates_worker(
    npz_path: str,
    truth_path: str,
    *,
    n_threads: int,
    seed: int,
    alpha: float,
    output_dir: str,
    result_path: str,
) -> None:
    os.setsid()
    X = sparse.load_npz(npz_path)
    truth = np.load(truth_path)
    result = run_cellstates_once(
        X,
        truth,
        n_threads=n_threads,
        seed=seed,
        alpha=alpha,
        output_dir=Path(output_dir),
    )
    Path(result_path).write_text(json.dumps(result))


def run_cellstates_once_with_timeout(
    *,
    n_threads: int,
    seed: int,
    alpha: float,
    output_dir: Path,
    timeout_s: float,
) -> dict:
    npz_path = BENCHMARK_DATA / "simulated_cells_x_genes.npz"
    truth_path = BENCHMARK_DATA / "simulated_truth.npy"
    result_path = output_dir / f"cellstates_{n_threads}core_seed{seed}_result.json"
    ctx = get_context("fork")
    proc = ctx.Process(
        target=_run_cellstates_worker,
        kwargs={
            "npz_path": str(npz_path),
            "truth_path": str(truth_path),
            "n_threads": int(n_threads),
            "seed": int(seed),
            "alpha": float(alpha),
            "output_dir": str(output_dir),
            "result_path": str(result_path),
        },
    )
    start = time.perf_counter()
    proc.start()
    proc.join(timeout_s)
    wall_s = time.perf_counter() - start
    if proc.is_alive():
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.join(10.0)
        if proc.is_alive():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.join(5.0)
        return {
            "method": f"cellstates_{n_threads}core",
            "seed": int(seed),
            "status": "timeout",
            "elapsed_s": float(wall_s),
            "ari_vs_truth": None,
            "fms_vs_truth": None,
            "n_clusters": None,
            "log_likelihood": None,
            "labels_path": None,
        }
    if not result_path.exists():
        return {
            "method": f"cellstates_{n_threads}core",
            "seed": int(seed),
            "status": "failed",
            "elapsed_s": float(wall_s),
            "ari_vs_truth": None,
            "fms_vs_truth": None,
            "n_clusters": None,
            "log_likelihood": None,
            "labels_path": None,
        }
    result = json.loads(result_path.read_text())
    result["elapsed_s"] = float(wall_s)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_mcellstate_once(
    X: sparse.csr_matrix,
    truth: np.ndarray,
    *,
    seed: int,
    alpha: float,
    output_dir: Path,
) -> dict:
    Optimizer, PartitionState, make_prior, _ = import_mcellstate()
    psi = make_prior(X, tau=alpha)
    leiden_target = max(128, min(int(X.shape[0] // 3), int(8.0 * np.sqrt(X.shape[0]))))
    search_target = max(16, int(2.0 * np.sqrt(X.shape[0])))
    state = PartitionState.from_csr(X, init="leiden_overclustered", seed=seed, n_clusters=leiden_target)
    optimizer = Optimizer(
        state=state,
        psi=psi,
        backend="torch-cpu",
        backend_threads=6,
        n_proposals=4000,
        seed=seed,
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
        greedy_merge_candidates=2048,
        exact_cell_reassign_passes=3,
        exact_cell_reassign_active_limit=1200,
        exact_cell_reassign_top_fraction=0.2,
        exact_cell_score_chunk=512,
        cluster_reassign_sweeps=0,
        perturb_every=3,
        perturb_steps=16,
        perturb_temperature=1.0,
        perturb_positive_cleanup=True,
        serial_refine_passes=0,
    )
    start = time.perf_counter()
    result = optimizer.fit(max_rounds=None, restarts=1, update_psi=False, stall_rounds=1, eta=0.0)
    elapsed = time.perf_counter() - start
    labels = result.z.astype(np.int32, copy=False)
    labels_path = output_dir / f"mcellstate_seed{seed}_labels.npy"
    np.save(labels_path, labels)
    return {
        "method": "mcellstate",
        "seed": int(seed),
        "status": "completed",
        "elapsed_s": float(elapsed),
        "ari_vs_truth": float(adjusted_rand_index(truth, labels)),
        "fms_vs_truth": float(fowlkes_mallows_score(truth, labels)),
        "n_clusters": int(len(np.unique(labels))),
        "log_likelihood": float(result.log_likelihood),
        "labels_path": str(labels_path),
        "timing_s": aggregate_optimizer_timing(result.history),
    }


def summarize_runs(runs: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    methods = sorted({run["method"] for run in runs})
    for method in methods:
        method_runs = [run for run in runs if run["method"] == method and run.get("status") == "completed"]
        if not method_runs:
            summaries.append(
                {
                    "method": method,
                    "n_runs": 0,
                    "mean_elapsed_s": None,
                    "mean_ari_vs_truth": None,
                    "mean_n_clusters": None,
                    "mean_log_likelihood": None,
                },
            )
            continue
        summaries.append(
            {
                "method": method,
                "n_runs": len(method_runs),
                "mean_elapsed_s": float(np.mean([run["elapsed_s"] for run in method_runs])),
                "mean_ari_vs_truth": float(np.mean([run["ari_vs_truth"] for run in method_runs])),
                "mean_n_clusters": float(np.mean([run["n_clusters"] for run in method_runs])),
                "mean_log_likelihood": float(np.mean([run["log_likelihood"] for run in method_runs])),
            },
        )
    return summaries


def write_report(summary: dict, report_path: Path) -> None:
    dataset = summary["dataset"]
    def fmt_metric(value: float | int | None, *, precision: int = 2) -> str:
        if value is None:
            return "-"
        if isinstance(value, int):
            return str(value)
        return f"{float(value):.{precision}f}"

    rows = "\n".join(
        f"| `{run['method']}` | {run['seed']} | {run['status']} | {fmt_metric(run['elapsed_s'])} | "
        f"{fmt_metric(run['ari_vs_truth'], precision=6)} | "
        f"{fmt_metric(run['n_clusters'], precision=0)} | "
        f"{fmt_metric(run['log_likelihood'])} |"
        for run in summary["runs"]
    )
    mean_rows = "\n".join(
        f"| `{row['method']}` | {row['n_runs']} | "
        f"{fmt_metric(row['mean_elapsed_s'])} | "
        f"{fmt_metric(row['mean_ari_vs_truth'], precision=6)} | "
        f"{fmt_metric(row['mean_n_clusters'])} | "
        f"{fmt_metric(row['mean_log_likelihood'])} |"
        for row in summary["method_summaries"]
    )
    report = f"""# Triplicate Partition Benchmark

## Setup

- Dataset source: `{summary["cellstates_repo"]}/test/data/simulated_data.tsv`
- Cells: {dataset["n_cells"]}
- Expressed genes: {dataset["n_genes_expressed"]}
- Truth clusters: {dataset["truth_clusters"]}
- Shared fixed prior magnitude: `{summary["alpha"]:.0f}`
- Requested repeats per method: `3`
- `cellstates` timeout per run: `{summary["cellstates_timeout_s"]:.0f}s`

## Per-run results

| Method | Seed | Status | Time (s) | ARI vs truth | Clusters | Log-likelihood |
|---|---:|---|---:|---:|---:|---:|
{rows}

## Mean by method

| Method | Runs | Mean time (s) | Mean ARI | Mean clusters | Mean log-likelihood |
|---|---:|---:|---:|---:|---:|
{mean_rows}
"""
    report_path.write_text(report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ensure_vendor_source(CELLSTATES_REPO)
    ensure_vendor_install(rebuild=False)
    X, truth, metadata = prepare_cached_dataset(CELLSTATES_REPO)
    alpha = alpha_from_counts(int(X.sum()), int(X.shape[0]))
    _, _, make_prior, _ = import_mcellstate()
    reference = evaluate_reference_partition(CELLSTATES_REPO, X, truth, make_prior(X, tau=alpha))

    runs: list[dict] = []

    for seed in (1, 2, 3):
        print(f"running cellstates 1 core, seed={seed}", flush=True)
        result = run_cellstates_once_with_timeout(
            n_threads=1,
            seed=seed,
            alpha=alpha,
            output_dir=OUTPUT_DIR,
            timeout_s=CELLSTATES_TIMEOUT_S,
        )
        runs.append(result)
        print(json.dumps(result, indent=2), flush=True)
        if result["status"] != "completed":
            break

    for seed in (1, 2, 3):
        print(f"running cellstates 6 cores, seed={seed}", flush=True)
        result = run_cellstates_once_with_timeout(
            n_threads=6,
            seed=seed,
            alpha=alpha,
            output_dir=OUTPUT_DIR,
            timeout_s=CELLSTATES_TIMEOUT_S,
        )
        runs.append(result)
        print(json.dumps(result, indent=2), flush=True)
        if result["status"] != "completed":
            break

    for seed in (1, 2, 3):
        print(f"running mcellstate, seed={seed}", flush=True)
        result = run_mcellstate_once(X, truth, seed=seed, alpha=alpha, output_dir=OUTPUT_DIR)
        runs.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "timing_s"}, indent=2), flush=True)

    summary = {
        "cellstates_repo": str(CELLSTATES_REPO),
        "dataset": metadata,
        "alpha": float(alpha),
        "cellstates_timeout_s": CELLSTATES_TIMEOUT_S,
        "reference_partition": reference.__dict__,
        "runs": runs,
        "method_summaries": summarize_runs(runs),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, REPORT_PATH)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
