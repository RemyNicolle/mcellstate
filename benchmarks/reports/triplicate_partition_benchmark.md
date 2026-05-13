# Triplicate Partition Benchmark

## Setup

- Dataset source: `/Users/remy.nicolle/Workspace/DEV/cellstates/test/data/simulated_data.tsv`
- Cells: 2298
- Expressed genes: 18871
- Truth clusters: 10
- Shared fixed prior magnitude: `4096`
- Requested repeats per method: `3`
- `cellstates` timeout per run: `1800s`

## Per-run results

| Method | Seed | Status | Time (s) | ARI vs truth | Clusters | Log-likelihood |
|---|---:|---|---:|---:|---:|---:|
| `cellstates_1core` | 1 | timeout | 1800.01 | - | - | - |
| `cellstates_6core` | 1 | timeout | 1800.00 | - | - | - |
| `mcellstate` | 1 | completed | 189.43 | 0.981142 | 9 | -81457510.71 |
| `mcellstate` | 2 | completed | 190.28 | 0.981142 | 9 | -81457510.71 |
| `mcellstate` | 3 | completed | 187.19 | 0.981142 | 9 | -81457510.71 |

## Mean by method

| Method | Runs | Mean time (s) | Mean ARI | Mean clusters | Mean log-likelihood |
|---|---:|---:|---:|---:|---:|
| `cellstates_1core` | 0 | - | - | - | - |
| `cellstates_6core` | 0 | - | - | - | - |
| `mcellstate` | 3 | 188.97 | 0.981142 | 9.00 | -81457510.71 |
