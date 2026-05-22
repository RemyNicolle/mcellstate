from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from mcellstate.cli import main
from mcellstate.validation import generate_synthetic_dataset


def test_cli_fit_runs_end_to_end(tmp_path):
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=30.0,
        seed=61,
    )
    input_path = tmp_path / "counts.npz"
    output_path = tmp_path / "labels.npy"
    summary_path = tmp_path / "summary.json"
    sparse.save_npz(input_path, synthetic.X)

    exit_code = main(
        [
            "fit",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-json",
            str(summary_path),
            "--preset",
            "gpu",
            "--backend",
            "cpu",
            "--init",
            "singletons",
            "--n-proposals",
            "64",
            "--restarts",
            "1",
            "--max-rounds",
            "4",
            "--seed",
            "61",
        ],
    )

    assert exit_code == 0
    labels = np.load(output_path)
    assert labels.shape == synthetic.z_true.shape

    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    assert summary["n_clusters"] >= 1
    assert summary["labels_path"] == str(output_path)
    assert summary["preset"] == "gpu"
    assert summary["search_policy"] == "gpu_structured"


def test_cli_fit_progress_mode_prints_timing(tmp_path, capsys):
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=30.0,
        seed=62,
    )
    input_path = tmp_path / "counts.npz"
    output_path = tmp_path / "labels.npy"
    sparse.save_npz(input_path, synthetic.X)

    exit_code = main(
        [
            "fit",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--preset",
            "gpu",
            "--backend",
            "cpu",
            "--init",
            "singletons",
            "--n-proposals",
            "64",
            "--restarts",
            "1",
            "--max-rounds",
            "2",
            "--seed",
            "62",
            "--progress",
        ],
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[fit]" in captured.out
    assert "timing_s=" in captured.out


def test_cli_fit_verbose_mode_prints_steps(tmp_path, capsys):
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=30.0,
        seed=63,
    )
    input_path = tmp_path / "counts.npz"
    output_path = tmp_path / "labels.npy"
    sparse.save_npz(input_path, synthetic.X)

    exit_code = main(
        [
            "fit",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--preset",
            "gpu",
            "--backend",
            "cpu",
            "--init",
            "singletons",
            "--n-proposals",
            "32",
            "--restarts",
            "1",
            "--max-rounds",
            "1",
            "--seed",
            "63",
            "--verbose",
        ],
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[trace]" in captured.out
    assert "sample" in captured.out or "score" in captured.out


def test_cli_cpu_preset_forces_cpu_backend_and_records_proposal_workers(tmp_path):
    synthetic = generate_synthetic_dataset(
        n_clusters=2,
        cells_per_cluster=4,
        n_genes=12,
        marker_strength=30.0,
        seed=64,
    )
    input_path = tmp_path / "counts.npz"
    output_path = tmp_path / "labels.npy"
    summary_path = tmp_path / "summary.json"
    sparse.save_npz(input_path, synthetic.X)

    exit_code = main(
        [
            "fit",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-json",
            str(summary_path),
            "--preset",
            "cpu",
            "--backend",
            "auto",
            "--init",
            "singletons",
            "--n-proposals",
            "32",
            "--restarts",
            "1",
            "--max-rounds",
            "1",
            "--seed",
            "64",
            "--proposal-workers",
            "2",
        ],
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text())
    assert summary["preset"] == "cpu"
    assert summary["optimizer_mode"] == "cpu-only"
    assert summary["backend"] == "torch-cpu"
    assert summary["proposal_workers"] == 2


def test_cli_convert_and_audit_doublets(tmp_path):
    tsv_path = tmp_path / "RNAmatrix_test.tsv"
    tsv_path.write_text("GeneID\tcell_a\tcell_b\nG1\t1\t0\nG2\t0\t2\n")
    npz_path = tmp_path / "counts.npz"

    exit_code = main(["convert", "--input", str(tsv_path), "--output", str(npz_path)])
    assert exit_code == 0
    converted = sparse.load_npz(npz_path)
    assert converted.shape == (2, 2)

    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, np.asarray([0, 1], dtype=np.int64))
    audit_path = tmp_path / "doublets.tsv"
    summary_path = tmp_path / "doublets.json"

    exit_code = main(
        [
            "audit-doublets",
            "--input",
            str(npz_path),
            "--labels",
            str(labels_path),
            "--output",
            str(audit_path),
            "--summary-json",
            str(summary_path),
        ],
    )
    assert exit_code == 0
    assert audit_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
