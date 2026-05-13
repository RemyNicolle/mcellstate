# Getting Started

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you have an NVIDIA GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Minimal workflow:

```bash
mcellstate convert --input RNAmatrix_sample.tsv --output sample.npz
mcellstate fit --input sample.npz --output labels.npy --preset gpu --backend cuda
mcellstate audit-doublets --input sample.npz --labels labels.npy --output doublets.tsv
```
