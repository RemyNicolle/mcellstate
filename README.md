# mcellstate

`mcellstate` is a clean replacement-oriented workspace for hard partitioning of scRNA-seq UMI count matrices, with a separate post-partition doublet audit tool and a clearly isolated upstream `cellstates` reference copy.

## Repo layout

- `src/mcellstate/`: maintained product code for conversion and fitting
- `src/mcellstate_doublet/`: maintained post-partition doublet audit code
- `scripts/`: thin user-facing wrappers
- `benchmarks/`: comparison scripts, benchmark data, benchmark outputs, benchmark reports
- `reference/cellstates/`: read-only upstream reference code

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U setuptools wheel
pip install --no-build-isolation -e .
```

For CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`mcellstate` requires Python 3.11 or newer.

## Quick start

Convert an RNAmatrix TSV file:

```bash
mcellstate convert --input RNAmatrix_sample.tsv --output sample.npz
```

Fit a partition:

```bash
mcellstate fit --input sample.npz --output labels.npy --preset gpu --backend cuda
```

Audit the fitted labels for likely doublets:

```bash
mcellstate audit-doublets --input sample.npz --labels labels.npy --output doublets.tsv
```

Batch-run a directory of `RNAmatrix*.tsv` files:

```bash
python scripts/run_rnamatrix_batch.py \
  --input-root /path/to/Cellstates \
  --output-root /path/to/results \
  --cache-root /path/to/cache \
  --preset gpu \
  --backend cuda
```

## Presets

- `balanced`: default mixed search strategy
- `gpu`: minimizes CPU-heavy refinement and favors CUDA scoring
- `quality`: more exhaustive refinement
- `benchmark`: stable settings for comparison runs

## Docs

- [Getting started](docs/getting-started.md)
- [CLI](docs/cli.md)
- [Input formats](docs/input-formats.md)
- [GPU notes](docs/gpu.md)
- [Benchmarking](docs/benchmarking.md)
