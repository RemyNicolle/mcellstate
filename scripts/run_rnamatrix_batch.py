from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_rna_matrices(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("RNAmatrix*.tsv") if path.is_file())


def run_command(args: list[str]) -> None:
    print("run:", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def _with_progress(args: argparse.Namespace, cmd: list[str]) -> list[str]:
    if bool(args.verbose):
        return [*cmd, "--verbose"]
    if bool(args.progress):
        return [*cmd, "--progress"]
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert RNAmatrix TSV files and fit mcellstate in batch.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing RNAmatrix*.tsv files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for labels and summary JSON outputs.")
    parser.add_argument("--cache-root", type=Path, default=Path("cache/rnamatrix_npz"), help="Directory for converted NPZ matrices.")
    parser.add_argument("--preset", choices=("balanced", "gpu", "quality", "benchmark"), default="gpu")
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
        summary_path = labels_path.with_suffix(".json")

        print(f"prepare: {tsv_path}", flush=True)
        if not npz_path.exists():
            run_command(
                _with_progress(
                    args,
                    [
                        sys.executable,
                        "-m",
                        "mcellstate",
                        "convert",
                        "--input",
                        str(tsv_path),
                        "--output",
                        str(npz_path),
                    ],
                )
            )
        elif args.overwrite:
            run_command(
                _with_progress(
                    args,
                    [
                        sys.executable,
                        "-m",
                        "mcellstate",
                        "convert",
                        "--input",
                        str(tsv_path),
                        "--output",
                        str(npz_path),
                    ],
                )
            )

        if labels_path.exists() and summary_path.exists() and not args.overwrite:
            print(f"skip existing result: {labels_path}", flush=True)
            continue

        run_command(
            _with_progress(
                args,
                [
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
                    "--preset",
                    str(args.preset),
                    "--backend",
                    str(args.backend),
                    "--seed",
                    str(args.seed),
                    "--restarts",
                    str(args.restarts),
                    "--n-proposals",
                    str(args.n_proposals),
                    "--max-rounds",
                    str(args.max_rounds),
                    "--stall-rounds",
                    str(args.stall_rounds),
                    "--improvement-window",
                    str(args.improvement_window),
                    "--eta",
                    str(args.eta),
                ],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
