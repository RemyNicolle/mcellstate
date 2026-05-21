from __future__ import annotations

from typing import Any

from .optimizer import Optimizer


FIT_PRESETS = ("balanced", "cpu", "gpu", "gpu-full", "quality", "benchmark")


def resolve_fit_preset(name: str) -> dict[str, Any]:
    preset = str(name).lower()
    if preset == "balanced":
        return {
            "optimizer_mode": Optimizer.EFFECTIVE_MODE,
            "optimizer_kwargs": {},
        }
    if preset == "cpu":
        return {
            "optimizer_mode": Optimizer.CPU_ONLY_MODE,
            "optimizer_kwargs": {},
        }
    if preset in {"gpu", "gpu-full"}:
        import os
        cpu_count = os.cpu_count() or 4
        default_workers = max(1, cpu_count // 2)
        return {
            "optimizer_mode": (
                Optimizer.GPU_HEAVY_MODE if preset == "gpu" else Optimizer.GPU_FULL_MODE
            ),
            "optimizer_kwargs": {
                "recompute_ll_each_round": False,
                "max_scored_proposals": 25000,
                "proposal_workers": default_workers,
            },
        }
    if preset == "quality":
        return {
            "optimizer_mode": Optimizer.EFFECTIVE_MODE,
            "optimizer_kwargs": {
                "greedy_merge_sweeps": 2,
                "exact_cell_reassign_passes": 3,
                "serial_refine_passes": 2,
                "cluster_reassign_sweeps": 1,
            },
        }
    if preset == "benchmark":
        return {
            "optimizer_mode": Optimizer.EFFECTIVE_MODE,
            "optimizer_kwargs": {
                "validate_batches": False,
                "perturb_every": 3,
                "perturb_steps": 16,
                "serial_refine_passes": 0,
            },
        }
    raise ValueError(f"unknown preset {name!r}; expected one of {FIT_PRESETS}")
