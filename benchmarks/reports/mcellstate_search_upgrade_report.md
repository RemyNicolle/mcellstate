# mcellstate search upgrade rerun

## Scope

- Dataset: `/Users/remy.nicolle/Workspace/DEV/cellstates/test/data/simulated_data.tsv`
- Reference method status from the previous comparison: fresh 1-core `cellstates` still timed out at 120 s, while the repo-shipped reference partition remained perfect on truth with 10 clusters.
- This rerun measures the upgraded `mcellstate` search after adding:
  deterministic top-k merge candidates,
  cached approximate merge neighborhoods,
  block peel and block move proposals,
  staged search weights,
  and a serial exact cleanup pass.

## Fresh bounded rerun

- Command logic: same simulated dataset, 6 torch CPU threads, fixed prior, 2 rounds, 5000 proposals per round, `serial_refine_passes=1`, `merge_gene_cluster_cap=24`.
- Wall time: `61.93 s`
- Clusters: `2107`
- ARI vs truth: `0.000896`
- FMS vs truth: `0.022993`
- Log-likelihood: `-83379705.35`

## Comparison to previous mcellstate baseline

| Run | Time (s) | Rounds | Proposals/round | Clusters | ARI | FMS | Log-likelihood |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous simpler search | 36.32 | 5 | 10000 | 1818 | 0.003869 | 0.045643 | -83288215.38 |
| Upgraded search bounded rerun | 61.93 | 2 | 5000 | 2107 | 0.000896 | 0.022993 | -83379705.35 |

## Interpretation

- The requested search features are implemented in code, but on this real simulation they are not yet a win.
- Even with a bounded configuration, the richer search is slower than the earlier simpler search and currently gives a worse partition.
- The main practical issue is that stronger neighborhood construction and exact block/refinement proposals add substantial overhead before they generate enough useful positive operations to offset that cost.
- The code path is now broader and more faithful to the intended search space, but it still needs another round of engineering focused on scalable neighborhood construction and cheaper refinement scheduling before it should replace the earlier benchmark configuration.

## Files

- Fresh bounded rerun JSON: `/Users/remy.nicolle/Workspace/DEV/mcellstate/benchmark_results/cellstates_comparison/search_upgrade_bounded_run.json`
- Previous comparison report: `/Users/remy.nicolle/Workspace/DEV/mcellstate/reports/cellstates_vs_mcellstate_report.md`
- Benchmark script: `/Users/remy.nicolle/Workspace/DEV/mcellstate/scripts/compare_cellstates_simulation.py`
