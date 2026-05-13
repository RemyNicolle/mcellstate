# Getting Started

Install:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U setuptools wheel
pip install --no-build-isolation -e .
```

If you have an NVIDIA GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Python 3.11 or newer is required.

Minimal workflow:

```bash
mcellstate convert --input RNAmatrix_sample.tsv --output sample.npz
mcellstate fit --input sample.npz --output labels.npy --preset gpu --backend cuda
mcellstate audit-doublets --input sample.npz --labels labels.npy --output doublets.tsv
```
