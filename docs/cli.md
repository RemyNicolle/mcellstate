# CLI

## `mcellstate convert`

Converts `RNAmatrix*.tsv` style input into sparse `.npz`.

```bash
mcellstate convert --input RNAmatrix_sample.tsv --output sample.npz
```

## `mcellstate fit`

Fits a hard partition from `.npz`, `.mtx(.gz)`, or `.npy`.

```bash
mcellstate fit --input sample.npz --output labels.npy --preset balanced --backend auto
```

Useful options:

- `--preset balanced|gpu|quality|benchmark`
- `--backend auto|cpu|torch-cpu|cuda`
- `--proposal-batch-size`
- `--cuda-chunk-size`
- `--restarts`
- `--n-proposals`

## `mcellstate audit-doublets`

Audits an existing label vector without changing clusters.

```bash
mcellstate audit-doublets --input sample.npz --labels labels.npy --output doublets.tsv
```
