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
    parser = argparse.ArgumentParser(
        description="Convert RNAmatrix TSV files and fit mcellstate in batch."
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
    )
    parser.add_argument(
        "--backend", default="cuda", help="mcellstate backend. Use cuda for GPU."
    )
    parser.add_argument(
        "--proposal-workers",
        type=int,
        default=None,
        help="Parallel proposal-family worker count.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--n-proposals", type=int, default=100_000)
    parser.add_argument("--max-scored-proposals", type=int, default=None)
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
    parser.add_argument("--stall-rounds", type=int, default=1)
    parser.add_argument("--improvement-window", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.0)
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
                    *(
                        []
                        if args.proposal_workers is None
                        else ["--proposal-workers", str(args.proposal_workers)]
                    ),
                    "--seed",
                    str(args.seed),
                    "--restarts",
                    str(args.restarts),
                    "--n-proposals",
                    str(args.n_proposals),
                    *(
                        []
                        if args.max_scored_proposals is None
                        else ["--max-scored-proposals", str(args.max_scored_proposals)]
                    ),
                    *(
                        []
                        if args.random_proposals is None
                        else (
                            ["--random-proposals"]
                            if args.random_proposals
                            else ["--guided-proposals"]
                        )
                    ),
                    *(
                        []
                        if args.random_accept_prob is None
                        else ["--random-accept-prob", str(args.random_accept_prob)]
                    ),
                    *(
                        []
                        if args.random_accept_max_fraction is None
                        else [
                            "--random-accept-max-fraction",
                            str(args.random_accept_max_fraction),
                        ]
                    ),
                    *(
                        []
                        if args.proposal_batch_size is None
                        else ["--proposal-batch-size", str(args.proposal_batch_size)]
                    ),
                    *(
                        []
                        if args.cuda_chunk_size is None
                        else ["--cuda-chunk-size", str(args.cuda_chunk_size)]
                    ),
                    *(
                        []
                        if args.recompute_ll_each_round is None
                        else (
                            ["--recompute-ll-each-round"]
                            if args.recompute_ll_each_round
                            else ["--fast-ll-tracking"]
                        )
                    ),
                    *(["--cuda-empty-cache"] if args.cuda_empty_cache else []),
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
