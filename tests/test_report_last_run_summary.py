from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "report_last_run_summary.py"
SPEC = importlib.util.spec_from_file_location("report_last_run_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_align_labels_to_cells_reorders_source_labels() -> None:
    target = ["cell_a", "cell_b", "cell_c"]
    source = ["cell_c", "cell_a", "cell_b"]
    labels = np.asarray([30, 10, 20], dtype=np.int64)

    aligned = MODULE.align_labels_to_cells(target, source, labels)

    np.testing.assert_array_equal(aligned, np.asarray([10, 20, 30], dtype=np.int64))


def test_align_labels_to_cells_rejects_missing_cell_id() -> None:
    target = ["cell_a", "cell_b"]
    source = ["cell_a", "cell_c"]
    labels = np.asarray([1, 2], dtype=np.int64)

    try:
        MODULE.align_labels_to_cells(target, source, labels)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected ValueError for a missing cell id")


def test_format_markdown_report_contains_expected_columns() -> None:
    current = MODULE.MethodSummary(
        method="currentMCellstate",
        labels_path="/tmp/current.npy",
        n_clusters=12,
        prior_source="make_prior(tau=4096)",
        prior_mass=4096.0,
        recorded_log_likelihood=-123.0,
        recomputed_mcellstate_log_likelihood=-123.0,
        recomputed_cellstates_log_likelihood=-123.0,
        parity_abs_diff=0.0,
        parity_matches=True,
        recorded_abs_diff=0.0,
        recorded_matches=True,
    )
    original = MODULE.MethodSummary(
        method="originalCellstate",
        labels_path="/tmp/original.txt",
        n_clusters=7,
        prior_source="dirichlet_pseudocounts.txt",
        prior_mass=8192.0,
        recorded_log_likelihood=None,
        recomputed_mcellstate_log_likelihood=-456.0,
        recomputed_cellstates_log_likelihood=-456.0,
        parity_abs_diff=0.0,
        parity_matches=True,
        recorded_abs_diff=None,
        recorded_matches=None,
    )
    summary = MODULE.SampleSummary(
        sample="RNAmatrix_Hajk_S01",
        input_path="/tmp/RNAmatrix_Hajk_S01.tsv",
        n_cells=100,
        n_genes=200,
        current=current,
        original=original,
    )

    report = MODULE.format_markdown_report(Path("/tmp/root"), [summary], ll_atol=1e-6)

    assert "| Sample | Cells | Genes | Current clusters |" in report
    assert "RNAmatrix_Hajk_S01" in report
    assert "Current saved LL" in report
    assert "Original lambda sum" in report
