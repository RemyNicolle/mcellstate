# Input Formats

## Supported `fit` inputs

- `.npz`
- `.mtx`
- `.mtx.gz`
- `.npy`

`mcellstate fit` does not read raw `RNAmatrix*.tsv` directly. Convert first:

```bash
mcellstate convert --input RNAmatrix_sample.tsv --output sample.npz
```

## RNAmatrix TSV orientation

Expected layout:

- first column: gene identifier
- remaining columns: cells
- one row per gene

The converter outputs a sparse matrix with:

- rows = cells
- columns = genes

and removes genes with zero total counts.
