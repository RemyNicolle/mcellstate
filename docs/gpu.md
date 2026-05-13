# GPU Notes

Use:

```bash
mcellstate fit --input sample.npz --output labels.npy --preset gpu --backend cuda
```

Important limitation: GPU acceleration in `mcellstate` currently speeds up proposal scoring, not the entire optimizer. CPU-side proposal generation, acceptance, and commit phases still exist.

This means:

- low average GPU utilization can be normal
- `nvidia-smi` can miss short CUDA bursts
- `--preset gpu` reduces CPU-heavy refinement phases to keep more time in CUDA scoring

If you want the most stable behavior rather than the most GPU-centric behavior, prefer:

```bash
mcellstate fit --input sample.npz --output labels.npy --preset balanced --backend cuda
```
