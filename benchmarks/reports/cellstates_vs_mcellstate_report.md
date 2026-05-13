# cellstates vs mcellstate on the bundled simulated dataset

## Setup

- Dataset source: `/Users/remy.nicolle/Workspace/DEV/cellstates/test/data/simulated_data.tsv`
- Cells: 2298
- Expressed genes: 18871
- Nonzero entries: 5063404
- Total UMIs: 10413999
- Truth clusters: 10
- Shared fixed prior: `alpha = 2**round(log2(total_umis / n_cells)) = 4096`
- Original `cellstates`: 6 threads, singleton initialization, fixed prior, no prior re-optimization
- `mcellstate`: 6 torch CPU threads, very overclustered Leiden-style warm start on a raw-count overlap graph, fixed prior, exact full merge sweeps, exact best-cell reassignment sweeps, stochastic split/move proposals, and optional perturbation cleanup
- `mcellstate` batch config: until convergence, 4000 proposals per round
  proposal mix `merge/peel/move/block_peel/block_move = 0.45/0.08/0.17/0.15/0.15`
  warm start target clusters `383`
  search target clusters `95`

## Results

| Method | Status | Time (s) | Clusters | ARI vs truth | FMS vs truth | Log-likelihood |
|---|---:|---:|---:|---:|---:|---:|
| `cellstates` fresh 6-core run | completed | 4702.89 | 9 | 0.981142 | 0.985137 | -81457510.71 |
| `cellstates` repo-shipped reference partition | precomputed | n/a | 10 | 1.000000 | 1.000000 | -81458341.63 |
| `mcellstate` fresh 6-core run | completed | 187.40 | 9 | 0.981142 | 0.985137 | -81457510.71 |

## `mcellstate` timing breakdown

| Phase | Time (s) |
|---|---:|
| `cluster_reassign_s` | 0.000 |
| `commit_s` | 0.000 |
| `configure_s` | 0.000 |
| `conflict_s` | 0.000 |
| `exact_cell_reassign_s` | 39.369 |
| `full_merge_s` | 76.627 |
| `greedy_merge_s` | 0.107 |
| `perturbation_s` | 0.000 |
| `proposal_s` | 58.164 |
| `scoring_s` | 12.829 |
| `serial_refine_s` | 0.000 |
| `tau_update_s` | 0.000 |

## Interpretation

- The completed upstream `cellstates` 6-core run reached the same 9-cluster partition as `mcellstate`, with the same ARI and FMS against truth and the same fixed-prior collapsed likelihood. It took 4702.89 seconds to do so.
- The current `mcellstate` implementation completed in 187.40 seconds on 6 CPU threads using an overclustered Leiden-style warm start plus exact full merge sweeps, exact best-cell reassignment sweeps, and stochastic split/move proposals. It ended at 9 clusters running until convergence, with ARI 0.981142 and FMS 0.985137.
- Under the same fixed prior, the `mcellstate` partition log-likelihood is -81457510.71, which is 830.91 above the upstream reference partition.
- On this benchmark, the remaining issue is runtime rather than search quality. `mcellstate` reaches the same partition quality as upstream MCMC about 25x faster, with the dominant costs still being the exact full merge sweep and the exact best-cell reassignment sweep.

## Files

- Benchmark script: `/Users/remy.nicolle/Workspace/DEV/mcellstate/scripts/compare_cellstates_simulation.py`
- `mcellstate` labels: `/Users/remy.nicolle/Workspace/DEV/mcellstate/benchmark_results/cellstates_comparison/mcellstate_labels.npy`
- Upstream 6-core result: `/Users/remy.nicolle/Workspace/DEV/mcellstate/benchmark_results/cellstates_comparison/cellstates_6core_result.json`
- Upstream reference labels: `/Users/remy.nicolle/Workspace/DEV/cellstates/test/results/optimized_clusters.txt`
