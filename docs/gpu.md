# GPU Notes

Use:

```bash
mcellstate fit --input sample.npz --output labels.npy --preset gpu --backend cuda --progress
```

Important limitation: GPU acceleration in `mcellstate` currently speeds up proposal scoring, not the entire optimizer. CPU-side proposal generation, acceptance, and commit phases still exist.

This means:

- low average GPU utilization can be normal
- `nvidia-smi` can miss short CUDA bursts
- `--preset gpu` reduces CPU-heavy refinement phases to keep more time in CUDA scoring
- random proposal generation for `--preset gpu` now runs through the GPU backend for merge/peel/move batches
- proposal batches auto-tune from the first rounds; use `--proposal-batch-size` if you need a fixed batch size
- CUDA scoring auto-tunes chunk size from the first rounds and current free memory; use `--cuda-chunk-size` if you need a fixed value

If you want the most stable behavior rather than the most GPU-centric behavior, prefer:

```bash
mcellstate fit --input sample.npz --output labels.npy --preset balanced --backend cuda
```
