# Last Run Compact Report

- Root: `/Users/remy.nicolle/Workspace/DEV/mcellstate/benchmarks/results/lastRun`
- Generated: `2026-05-21T10:19:23+00:00`
- Likelihood parity absolute tolerance: `1e-05`

| Sample | Cells | Genes | Current clusters | Current tau | Current saved LL | Current recomputed LL (`mcellstate`) | Current recomputed LL (`cellstates`) | Current parity diff | Original clusters | Original lambda sum | Original recomputed LL (`mcellstate`) | Original recomputed LL (`cellstates`) | Original parity diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RNAmatrix_Hajk_S01 | 8276 | 16025 | 2097 | 4096 | -274996113.831341 | -274996113.831341 | -274996113.831341 | 5.36442e-07 | 827 | 8192 | -272822155.398499 | -272822155.398498 | 1.78814e-07 |
| RNAmatrix_Hajk_S02 | 2134 | 16009 | 494 | 4096 | -59284429.801521 | -59284429.801521 | -59284429.801521 | 1.2666e-07 | 521 | 8192 | -58929146.349137 | -58929146.349138 | 7.45058e-08 |
| RNAmatrix_Hajk_S03 | 6438 | 16022 | 1834 | 4096 | -280660676.548092 | -280660676.548093 | -280660676.548094 | 1.01328e-06 | 402 | 16384 | -278717748.231570 | -278717748.231570 | 1.19209e-07 |
| RNAmatrix_Hajk_S04 | 3480 | 15981 | 643 | 4096 | -154312410.383908 | -154312410.383908 | -154312410.383908 | 2.98023e-08 | 1893 | 16384 | -153608933.988818 | -153608933.988820 | 1.3113e-06 |

## Notes

- `Current tau` is the total prior mass used by the current `mcellstate` run. `Original lambda sum` is the total prior mass loaded from upstream `dirichlet_pseudocounts.txt`.
- `Current saved LL` comes from the saved `currentMCellstate/*.labels.json`. The recomputed columns are fresh evaluations from the raw counts and labels.
- Parity diffs compare `mcellstate` against the upstream `cellstates.Cluster.total_likelihood` on the same partition and the same pseudocount vector.

