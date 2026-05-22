from __future__ import annotations

import argparse
import gc
import json
import sys
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcellstate.io import tsv_to_csr  # noqa: E402
from mcellstate.likelihood import full_partition_log_likelihood_from_assignment  # noqa: E402
from mcellstate.prior import make_prior  # noqa: E402


DEFAULT_ROOT = ROOT / "benchmarks" / "results" / "lastRun"
DEFAULT_MARKDOWN = ROOT / "benchmarks" / "reports" / "last_run_compact_report.md"
DEFAULT_JSON = DEFAULT_ROOT / "compact_report.json"


@dataclass
class MethodSummary:
    method: str
    labels_path: str
    n_clusters: int
    prior_source: str
    prior_mass: float
    recorded_log_likelihood: float | None
    recomputed_mcellstate_log_likelihood: float
    recomputed_cellstates_log_likelihood: float
    parity_abs_diff: float
    parity_matches: bool
    recorded_abs_diff: float | None
    recorded_matches: bool | None


@dataclass
class SampleSummary:
    sample: str
    input_path: str
    n_cells: int
    n_genes: int
    current: MethodSummary
    original: MethodSummary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a compact likelihood parity report for benchmarks/results/lastRun."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Benchmark result root containing raw TSVs, currentMCellstate, and originalCellstate.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_MARKDOWN,
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON,
        help="Machine-readable JSON output path.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample basenames to restrict the report, for example RNAmatrix_Hajk_S01.",
    )
    parser.add_argument(
        "--cellstates-threads",
        type=int,
        default=1,
        help="Thread count passed to upstream cellstates.Cluster.",
    )
    parser.add_argument(
        "--cellstates-n-cache",
        type=int,
        default=128,
        help="lgamma cache size passed to upstream cellstates.Cluster.",
    )
    parser.add_argument(
        "--ll-atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for parity checks between mcellstate and upstream cellstates.",
    )
    return parser


def _load_header_cell_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    if len(header) < 2:
        raise ValueError(f"{path} has no cell columns")
    return header[1:]


def _load_text_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _load_labels(path: Path) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        labels = np.load(path)
    else:
        labels = np.loadtxt(path, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"{path} does not contain a one-dimensional label vector")
    return labels


def align_labels_to_cells(
    target_cell_ids: list[str],
    source_cell_ids: list[str],
    labels: np.ndarray,
) -> np.ndarray:
    if len(source_cell_ids) != len(labels):
        raise ValueError(
            f"source cell id count {len(source_cell_ids)} does not match label count {len(labels)}"
        )
    if target_cell_ids == source_cell_ids:
        return labels.astype(np.int64, copy=False)
    if len(target_cell_ids) != len(source_cell_ids):
        raise ValueError(
            f"target cell id count {len(target_cell_ids)} does not match source count {len(source_cell_ids)}"
        )
    source_index = {cell_id: idx for idx, cell_id in enumerate(source_cell_ids)}
    if len(source_index) != len(source_cell_ids):
        raise ValueError("duplicate cell ids in source ordering")
    try:
        order = np.asarray([source_index[cell_id] for cell_id in target_cell_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"target cell id {exc.args[0]!r} is missing from source ordering") from exc
    return labels[order].astype(np.int64, copy=False)


def _find_reference_build_path() -> Path | None:
    build_root = ROOT / "reference" / "cellstates" / "build"
    candidates = sorted(build_root.glob("lib.*"))
    return candidates[0] if candidates else None


def import_reference_cluster() -> type[Any]:
    try:
        from cellstates.cluster import Cluster  # type: ignore

        return Cluster
    except Exception:
        build_path = _find_reference_build_path()
        if build_path is None:
            raise RuntimeError(
                "could not import upstream cellstates and no vendored build path was found under "
                f"{ROOT / 'reference' / 'cellstates' / 'build'}"
            ) from None

    if str(build_path) not in sys.path:
        sys.path.insert(0, str(build_path))

    restore_matplotlib = "matplotlib" not in sys.modules
    if restore_matplotlib:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")
    try:
        from cellstates.cluster import Cluster  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "failed to import upstream cellstates Cluster from the vendored build path "
            f"{build_path}"
        ) from exc
    finally:
        if restore_matplotlib:
            sys.modules.pop("matplotlib", None)
    return Cluster


def cellstates_log_likelihood_from_dense(
    dense_counts_genes_by_cells: np.ndarray,
    labels: np.ndarray,
    pseudocounts: np.ndarray,
    *,
    num_threads: int,
    n_cache: int,
) -> float:
    Cluster = import_reference_cluster()
    cluster = Cluster(
        np.asarray(dense_counts_genes_by_cells, dtype=np.int64, order="C"),
        np.asarray(pseudocounts, dtype=np.float64),
        np.asarray(labels, dtype=np.int32),
        num_threads=int(num_threads),
        n_cache=int(n_cache),
        seed=1,
    )
    return float(cluster.total_likelihood)


def _method_summary(
    *,
    method: str,
    labels_path: Path,
    labels: np.ndarray,
    prior_source: str,
    pseudocounts: np.ndarray,
    dense_counts_genes_by_cells: np.ndarray,
    X,
    cellstates_threads: int,
    cellstates_n_cache: int,
    ll_atol: float,
    recorded_log_likelihood: float | None = None,
) -> MethodSummary:
    ll_mcellstate = float(
        full_partition_log_likelihood_from_assignment(
            X,
            labels,
            np.asarray(pseudocounts, dtype=np.float64),
        )
    )
    ll_cellstates = float(
        cellstates_log_likelihood_from_dense(
            dense_counts_genes_by_cells,
            labels,
            pseudocounts,
            num_threads=cellstates_threads,
            n_cache=cellstates_n_cache,
        )
    )
    parity_abs_diff = abs(ll_mcellstate - ll_cellstates)
    recorded_abs_diff = (
        None if recorded_log_likelihood is None else abs(float(recorded_log_likelihood) - ll_mcellstate)
    )
    return MethodSummary(
        method=method,
        labels_path=str(labels_path),
        n_clusters=int(np.unique(labels).size),
        prior_source=prior_source,
        prior_mass=float(np.asarray(pseudocounts, dtype=np.float64).sum()),
        recorded_log_likelihood=None
        if recorded_log_likelihood is None
        else float(recorded_log_likelihood),
        recomputed_mcellstate_log_likelihood=ll_mcellstate,
        recomputed_cellstates_log_likelihood=ll_cellstates,
        parity_abs_diff=parity_abs_diff,
        parity_matches=bool(parity_abs_diff <= ll_atol),
        recorded_abs_diff=recorded_abs_diff,
        recorded_matches=None if recorded_abs_diff is None else bool(recorded_abs_diff <= ll_atol),
    )


def sample_summary(
    sample_name: str,
    root: Path,
    *,
    cellstates_threads: int,
    cellstates_n_cache: int,
    ll_atol: float,
) -> SampleSummary:
    input_path = root / f"{sample_name}.tsv"
    current_dir = root / "currentMCellstate"
    original_dir = root / "originalCellstate" / sample_name

    current_json_path = current_dir / f"{sample_name}.labels.json"
    current_labels_path = current_dir / f"{sample_name}.labels.npy"
    original_labels_path = original_dir / "optimized_clusters.txt"
    original_cellid_path = original_dir / "CellID.txt"
    original_pseudocounts_path = original_dir / "dirichlet_pseudocounts.txt"

    target_cell_ids = _load_header_cell_ids(input_path)
    X = tsv_to_csr(input_path)
    if int(X.shape[0]) != len(target_cell_ids):
        raise ValueError(
            f"{input_path} has {len(target_cell_ids)} header cell ids but converts to {int(X.shape[0])} rows"
        )

    current_meta = json.loads(current_json_path.read_text(encoding="utf-8"))
    current_labels = _load_labels(current_labels_path)
    if len(current_labels) != len(target_cell_ids):
        raise ValueError(
            f"{current_labels_path} has {len(current_labels)} labels but {input_path} has {len(target_cell_ids)} cells"
        )
    current_tau = float(current_meta["tau"])
    current_prior = make_prior(X, tau=current_tau)

    original_source_cell_ids = _load_text_lines(original_cellid_path)
    original_labels = align_labels_to_cells(
        target_cell_ids,
        original_source_cell_ids,
        _load_labels(original_labels_path),
    )
    original_pseudocounts = np.asarray(
        np.loadtxt(original_pseudocounts_path, dtype=np.float64),
        dtype=np.float64,
    )
    if current_prior.shape[0] != int(X.shape[1]):
        raise ValueError(
            f"current prior length {current_prior.shape[0]} does not match {input_path} gene count {int(X.shape[1])}"
        )
    if original_pseudocounts.shape[0] != int(X.shape[1]):
        raise ValueError(
            f"original pseudocount length {original_pseudocounts.shape[0]} does not match "
            f"{input_path} gene count {int(X.shape[1])}"
        )

    dense_counts = X.transpose().toarray().astype(np.int64, copy=False)
    try:
        current = _method_summary(
            method="currentMCellstate",
            labels_path=current_labels_path,
            labels=current_labels,
            prior_source=f"make_prior(tau={current_tau:g})",
            pseudocounts=current_prior,
            dense_counts_genes_by_cells=dense_counts,
            X=X,
            cellstates_threads=cellstates_threads,
            cellstates_n_cache=cellstates_n_cache,
            ll_atol=ll_atol,
            recorded_log_likelihood=float(current_meta["log_likelihood"]),
        )
        original = _method_summary(
            method="originalCellstate",
            labels_path=original_labels_path,
            labels=original_labels,
            prior_source="dirichlet_pseudocounts.txt",
            pseudocounts=original_pseudocounts,
            dense_counts_genes_by_cells=dense_counts,
            X=X,
            cellstates_threads=cellstates_threads,
            cellstates_n_cache=cellstates_n_cache,
            ll_atol=ll_atol,
            recorded_log_likelihood=None,
        )
    finally:
        del dense_counts
        gc.collect()

    return SampleSummary(
        sample=sample_name,
        input_path=str(input_path),
        n_cells=int(X.shape[0]),
        n_genes=int(X.shape[1]),
        current=current,
        original=original,
    )


def select_samples(root: Path, requested: list[str] | None) -> list[str]:
    discovered = sorted(path.stem for path in root.glob("RNAmatrix_*.tsv"))
    if requested is None or len(requested) == 0:
        return discovered
    requested_set = set(requested)
    missing = sorted(requested_set.difference(discovered))
    if missing:
        raise ValueError(f"requested samples are missing under {root}: {', '.join(missing)}")
    return [sample for sample in discovered if sample in requested_set]


def format_markdown_report(
    root: Path,
    summaries: list[SampleSummary],
    *,
    ll_atol: float,
) -> str:
    lines = [
        "# Last Run Compact Report",
        "",
        f"- Root: `{root}`",
        f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Likelihood parity absolute tolerance: `{ll_atol:g}`",
        "",
        "| Sample | Cells | Genes | Current clusters | Current tau | Current saved LL | Current recomputed LL (`mcellstate`) | Current recomputed LL (`cellstates`) | Current parity diff | Original clusters | Original lambda sum | Original recomputed LL (`mcellstate`) | Original recomputed LL (`cellstates`) | Original parity diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.sample,
                    str(summary.n_cells),
                    str(summary.n_genes),
                    str(summary.current.n_clusters),
                    f"{summary.current.prior_mass:.0f}",
                    f"{summary.current.recorded_log_likelihood:.6f}",
                    f"{summary.current.recomputed_mcellstate_log_likelihood:.6f}",
                    f"{summary.current.recomputed_cellstates_log_likelihood:.6f}",
                    f"{summary.current.parity_abs_diff:.6g}",
                    str(summary.original.n_clusters),
                    f"{summary.original.prior_mass:.0f}",
                    f"{summary.original.recomputed_mcellstate_log_likelihood:.6f}",
                    f"{summary.original.recomputed_cellstates_log_likelihood:.6f}",
                    f"{summary.original.parity_abs_diff:.6g}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Current tau` is the total prior mass used by the current `mcellstate` run. `Original lambda sum` is the total prior mass loaded from upstream `dirichlet_pseudocounts.txt`.",
            "- `Current saved LL` comes from the saved `currentMCellstate/*.labels.json`. The recomputed columns are fresh evaluations from the raw counts and labels.",
            "- Parity diffs compare `mcellstate` against the upstream `cellstates.Cluster.total_likelihood` on the same partition and the same pseudocount vector.",
            "",
        ]
    )
    return "\n".join(lines)


def summary_to_json_payload(
    root: Path,
    summaries: list[SampleSummary],
    *,
    ll_atol: float,
) -> dict[str, Any]:
    return {
        "root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ll_atol": float(ll_atol),
        "samples": [asdict(summary) for summary in summaries],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    samples = select_samples(root, args.samples)
    summaries = [
        sample_summary(
            sample,
            root,
            cellstates_threads=int(args.cellstates_threads),
            cellstates_n_cache=int(args.cellstates_n_cache),
            ll_atol=float(args.ll_atol),
        )
        for sample in samples
    ]
    markdown = format_markdown_report(root, summaries, ll_atol=float(args.ll_atol))
    payload = summary_to_json_payload(root, summaries, ll_atol=float(args.ll_atol))

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(markdown)
    print()
    print(f"wrote markdown report to {args.output_markdown}")
    print(f"wrote json summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
