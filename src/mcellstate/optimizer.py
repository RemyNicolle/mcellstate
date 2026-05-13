from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TextIO

import numpy as np

from .backends import make_backend
from .likelihood import full_partition_log_likelihood
from .prior import optimize_tau
from .proposals import (
    BlockMoveProposal,
    BlockPeelProposal,
    MergeProposal,
    MoveProposal,
    PeelProposal,
    Proposal,
    ProposalSampler,
)
from .state import PartitionState


@dataclass
class FitResult:
    state: PartitionState
    z: np.ndarray
    log_likelihood: float
    history: list[dict]
    psi: np.ndarray
    restart_summaries: list[dict] | None = None


class Optimizer:
    EFFECTIVE_MODE = "effective"
    GPU_HEAVY_MODE = "gpu-heavy"
    OPTIMIZER_MODES = (EFFECTIVE_MODE, GPU_HEAVY_MODE)

    def __init__(
        self,
        *,
        state: PartitionState,
        psi: np.ndarray,
        optimizer_mode: str = EFFECTIVE_MODE,
        backend: str = "cpu",
        n_proposals: int = 100_000,
        seed: int | None = None,
        pi_merge: float = 0.45,
        pi_peel: float = 0.08,
        pi_move: float = 0.17,
        pi_block_peel: float = 0.15,
        pi_block_move: float = 0.15,
        epsilon_uniform: float = 0.05,
        merge_uniform_prob: float | None = None,
        peel_uniform_prob: float | None = None,
        move_uniform_prob: float | None = None,
        block_peel_uniform_prob: float | None = None,
        block_move_uniform_prob: float | None = None,
        top_merge_neighbors: int = 8,
        deterministic_merge_ratio: float = 0.25,
        merge_gene_cluster_cap: int = 24,
        signature_top_genes: int = 16,
        signature_pool_size: int = 48,
        block_size_min: int = 2,
        block_size_max: int = 12,
        validate_batches: bool = True,
        backend_threads: int | None = None,
        staged_search: bool = True,
        target_clusters: int | None = None,
        leiden_restart_targets: list[int] | tuple[int, ...] | None = None,
        full_merge_stage: bool = True,
        full_merge_active_limit: int = 1200,
        full_merge_score_chunk: int = 8192,
        full_merge_max_sweeps: int = 32,
        greedy_merge_sweeps: int = 1,
        greedy_merge_candidates: int = 4096,
        exact_cell_reassign_passes: int = 2,
        exact_cell_reassign_active_limit: int = 1200,
        exact_cell_reassign_top_fraction: float = 0.2,
        exact_cell_score_chunk: int = 512,
        cluster_reassign_sweeps: int = 1,
        cluster_reassign_max_sources: int = 256,
        tau_update_interval: int = 5,
        perturb_every: int = 0,
        perturb_steps: int = 0,
        perturb_temperature: float = 1.0,
        perturb_positive_cleanup: bool = True,
        serial_refine_passes: int = 1,
        serial_refine_merge_candidates: int = 3,
        serial_refine_move_candidates: int = 2,
        serial_refine_cells_per_cluster: int = 6,
    ) -> None:
        optimizer_mode = str(optimizer_mode).lower()
        if optimizer_mode not in self.OPTIMIZER_MODES:
            raise ValueError(
                f"unknown optimizer_mode {optimizer_mode!r}; expected one of {self.OPTIMIZER_MODES}",
            )
        self.initial_state = state.copy()
        self.psi = np.asarray(psi, dtype=np.float64)
        self.optimizer_mode = optimizer_mode
        self.backend_name = backend
        self.backend_threads = backend_threads
        self.n_proposals = int(n_proposals)
        self.seed = seed
        self.validate_batches = bool(validate_batches)
        self.staged_search = bool(staged_search)
        self.target_clusters = None if target_clusters is None else int(target_clusters)
        self.leiden_restart_targets = (
            None if leiden_restart_targets is None else tuple(int(value) for value in leiden_restart_targets)
        )
        self.full_merge_stage = bool(full_merge_stage)
        self.full_merge_active_limit = int(full_merge_active_limit)
        self.full_merge_score_chunk = int(full_merge_score_chunk)
        self.full_merge_max_sweeps = int(full_merge_max_sweeps)
        self.greedy_merge_sweeps = int(greedy_merge_sweeps)
        self.greedy_merge_candidates = int(greedy_merge_candidates)
        self.exact_cell_reassign_passes = int(exact_cell_reassign_passes)
        self.exact_cell_reassign_active_limit = int(exact_cell_reassign_active_limit)
        self.exact_cell_reassign_top_fraction = float(exact_cell_reassign_top_fraction)
        self.exact_cell_score_chunk = int(exact_cell_score_chunk)
        self.cluster_reassign_sweeps = int(cluster_reassign_sweeps)
        self.cluster_reassign_max_sources = int(cluster_reassign_max_sources)
        self.tau_update_interval = int(tau_update_interval)
        self.perturb_every = int(perturb_every)
        self.perturb_steps = int(perturb_steps)
        self.perturb_temperature = float(perturb_temperature)
        self.perturb_positive_cleanup = bool(perturb_positive_cleanup)
        self.serial_refine_passes = int(serial_refine_passes)
        self.serial_refine_merge_candidates = int(serial_refine_merge_candidates)
        self.serial_refine_move_candidates = int(serial_refine_move_candidates)
        self.serial_refine_cells_per_cluster = int(serial_refine_cells_per_cluster)
        self._apply_optimizer_mode_defaults()
        self.rng = np.random.default_rng(seed)
        self.base_family_weights = {
            "merge": float(pi_merge),
            "peel": float(pi_peel),
            "move": float(pi_move),
            "block_peel": float(pi_block_peel),
            "block_move": float(pi_block_move),
        }
        self.sampler = ProposalSampler(
            pi_merge=pi_merge,
            pi_peel=pi_peel,
            pi_move=pi_move,
            pi_block_peel=pi_block_peel,
            pi_block_move=pi_block_move,
            epsilon_uniform=epsilon_uniform,
            merge_uniform_prob=merge_uniform_prob,
            peel_uniform_prob=peel_uniform_prob,
            move_uniform_prob=move_uniform_prob,
            block_peel_uniform_prob=block_peel_uniform_prob,
            block_move_uniform_prob=block_move_uniform_prob,
            top_merge_neighbors=top_merge_neighbors,
            deterministic_merge_ratio=deterministic_merge_ratio,
            merge_gene_cluster_cap=merge_gene_cluster_cap,
            signature_top_genes=signature_top_genes,
            signature_pool_size=signature_pool_size,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
            seed=seed,
        )
        self.sampler.set_scoring_context(self.psi)

    def _apply_optimizer_mode_defaults(self) -> None:
        if self.optimizer_mode == self.EFFECTIVE_MODE:
            return
        self.full_merge_stage = False
        self.greedy_merge_sweeps = 0
        self.exact_cell_reassign_passes = 0
        self.cluster_reassign_sweeps = 0
        self.perturb_every = 0
        self.perturb_steps = 0
        self.serial_refine_passes = 0

    def fit(
        self,
        *,
        max_rounds: int | None = 1000,
        update_psi: bool = False,
        restarts: int = 1,
        stall_rounds: int = 10,
        improvement_window: int = 5,
        eta: float = 1e-8,
        restart_inits: list[str] | None = None,
        progress: bool = False,
        verbose: bool = False,
        progress_stream: TextIO | None = None,
        verbose_stream: TextIO | None = None,
    ) -> FitResult:
        best_result: FitResult | None = None
        histories: list[dict] = []
        progress_stream = progress_stream or None
        verbose_stream = verbose_stream or None

        for restart in range(int(restarts)):
            state = self._make_restart_state(restart, restart_inits=restart_inits)
            psi = self.psi.copy()
            state.initialize_likelihood_cache(psi)
            self.sampler.notify_state_changed(state, set(state.active_cluster_ids))
            backend = make_backend(self.backend_name, psi, state, num_threads=self.backend_threads)
            history: list[dict] = []
            current_ll = state.total_log_likelihood_cached(psi)
            stall = 0
            round_idx = 0
            round_limit = None if max_rounds is None else int(max_rounds)

            while round_limit is None or round_idx < round_limit:
                round_before_ll = current_ll
                stage = self._stage_name(state)
                self.sampler.set_scoring_context(psi)
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="round start",
                    )

                timing: dict[str, float] = {}
                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="configure stage",
                    )
                self._configure_stage(state, stage)
                timing["configure_s"] = time.perf_counter() - timer

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(stream=verbose_stream, restart=restart, round_idx=round_idx, stage=stage, message="full merge phase")
                full_merge = self._full_merge_phase(state, backend, stage=stage)
                timing["full_merge_s"] = time.perf_counter() - timer
                if full_merge["delta"] != 0.0:
                    current_ll = full_partition_log_likelihood(state, psi)

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="exact cell reassignment phase",
                    )
                exact_cell_reassign = self._exact_cell_reassignment_sweep(state, backend, stage=stage)
                timing["exact_cell_reassign_s"] = time.perf_counter() - timer
                if exact_cell_reassign["delta"] != 0.0:
                    current_ll = full_partition_log_likelihood(state, psi)

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(stream=verbose_stream, restart=restart, round_idx=round_idx, stage=stage, message="greedy merge phase")
                greedy_merge = self._greedy_merge_sweep(state, backend, stage=stage)
                timing["greedy_merge_s"] = time.perf_counter() - timer
                if greedy_merge["delta"] != 0.0:
                    current_ll = full_partition_log_likelihood(state, psi)

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message=f"sample {self.n_proposals} proposals",
                    )
                proposals = self.sampler.sample_batch(state, self.n_proposals)
                timing["proposal_s"] = time.perf_counter() - timer
                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message=f"score {len(proposals)} proposals on {self.backend_name}",
                    )
                scores = backend.score_batch(state, proposals)
                timing["scoring_s"] = time.perf_counter() - timer
                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="select non-conflicting positives",
                    )
                accepted = self._select_positive_nonconflicting(proposals, scores)
                timing["conflict_s"] = time.perf_counter() - timer
                accepted_delta = float(sum(item["delta"] for item in accepted))

                round_record = {
                    "restart": restart,
                    "round": round_idx,
                    "stage": stage,
                    "family_weights": dict(
                        zip(self.sampler.family_names, self.sampler.family_weights.tolist(), strict=True),
                    ),
                    "n_proposals": len(proposals),
                    "n_positive": int(np.sum(scores > 0.0)),
                    "n_accepted": len(accepted),
                    "full_merge": full_merge,
                    "exact_cell_reassign": exact_cell_reassign,
                    "greedy_merge": greedy_merge,
                    "accepted_delta": accepted_delta,
                    "log_likelihood_before": round_before_ll,
                    "touch_sets": [sorted(item["touch_set"]) for item in accepted],
                    "operations": [self._proposal_to_record(item["proposal"], item["delta"]) for item in accepted],
                }

                timer = time.perf_counter()
                touched_clusters: set[int] = set()
                if accepted:
                    if verbose:
                        self._emit_verbose(
                            stream=verbose_stream,
                            restart=restart,
                            round_idx=round_idx,
                            stage=stage,
                            message=f"commit {len(accepted)} accepted operations",
                        )
                    touched_clusters = self._commit_batch(state, accepted)
                    self.sampler.notify_state_changed(state, touched_clusters)
                    current_ll = full_partition_log_likelihood(state, psi)
                timing["commit_s"] = time.perf_counter() - timer

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(stream=verbose_stream, restart=restart, round_idx=round_idx, stage=stage, message="serial refine phase")
                refinement = self._serial_refine(state, psi, backend, touched_clusters, stage)
                timing["serial_refine_s"] = time.perf_counter() - timer
                if refinement["delta"] != 0.0:
                    current_ll = full_partition_log_likelihood(state, psi)

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="cluster reassignment phase",
                    )
                cluster_reassign = self._cluster_reassignment_sweep(state, backend, stage=stage)
                timing["cluster_reassign_s"] = time.perf_counter() - timer
                if cluster_reassign["delta"] != 0.0:
                    current_ll = full_partition_log_likelihood(state, psi)

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(stream=verbose_stream, restart=restart, round_idx=round_idx, stage=stage, message="perturbation phase")
                perturbation = self._perturbation_phase(state, backend, round_idx, stage)
                timing["perturbation_s"] = time.perf_counter() - timer
                if perturbation["delta"] != 0.0:
                    current_ll = full_partition_log_likelihood(state, psi)

                after_partition_ll = full_partition_log_likelihood(state, psi)
                total_delta = (
                    float(full_merge["delta"])
                    + float(exact_cell_reassign["delta"])
                    + float(greedy_merge["delta"])
                    + accepted_delta
                    + float(refinement["delta"])
                    + float(cluster_reassign["delta"])
                    + float(perturbation["delta"])
                )

                round_record["active_clusters"] = len(state.active_cluster_ids)
                round_record["refinement"] = refinement
                round_record["cluster_reassign"] = cluster_reassign
                round_record["perturbation"] = perturbation
                round_record["total_delta"] = total_delta
                round_record["partition_log_likelihood_after"] = after_partition_ll

                if self.validate_batches:
                    observed_delta = after_partition_ll - round_before_ll
                    expected_total_delta = total_delta
                    delta_error = abs(observed_delta - expected_total_delta)
                    delta_tolerance = 1e-8 * max(1.0, abs(observed_delta), abs(expected_total_delta))
                    if delta_error > delta_tolerance:
                        raise AssertionError(
                            "batch likelihood gain mismatch: "
                            f"observed {observed_delta}, expected {expected_total_delta}",
                        )
                    self._assert_disjoint_touch_sets(accepted)
                    state.validate()

                current_ll = after_partition_ll
                tau_update = {"updated": False, "delta": 0.0}
                if update_psi and self._should_update_tau(round_idx):
                    timer = time.perf_counter()
                    old_ll = current_ll
                    _, psi, current_ll = optimize_tau(state, psi)
                    state.initialize_likelihood_cache(psi)
                    backend = make_backend(self.backend_name, psi, state, num_threads=self.backend_threads)
                    self.sampler.set_scoring_context(psi)
                    tau_update = {"updated": True, "delta": float(current_ll - old_ll)}
                    timing["tau_update_s"] = time.perf_counter() - timer
                else:
                    timing["tau_update_s"] = 0.0
                round_record["tau_update"] = tau_update
                round_record["log_likelihood_after"] = current_ll
                round_record["timing_s"] = timing
                history.append(round_record)
                if progress:
                    self._emit_progress(
                        stream=progress_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        round_record=round_record,
                    )
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="round end",
                    )

                n_positive_steps = (
                    int(full_merge["n_steps"])
                    + int(exact_cell_reassign["n_steps"])
                    + int(greedy_merge["n_steps"])
                    + len(accepted)
                    + int(refinement["n_steps"])
                    + int(cluster_reassign["n_steps"])
                    + int(perturbation.get("positive_cleanup_steps", 0))
                )
                stall = stall + 1 if n_positive_steps == 0 else 0
                if stall >= stall_rounds:
                    break

                if self._relative_improvement_below_threshold(history, improvement_window, eta):
                    break
                round_idx += 1

            result = FitResult(
                state=state,
                z=state.z.copy(),
                log_likelihood=current_ll,
                history=history,
                psi=psi.copy(),
                restart_summaries=None,
            )
            histories.append({"restart": restart, "history": history, "log_likelihood": current_ll})
            if best_result is None or result.log_likelihood > best_result.log_likelihood:
                best_result = result

        if best_result is None:  # pragma: no cover - defensive branch
            raise RuntimeError("optimizer did not run any restarts")
        return FitResult(
            state=best_result.state,
            z=best_result.z.copy(),
            log_likelihood=best_result.log_likelihood,
            history=list(best_result.history),
            psi=best_result.psi.copy(),
            restart_summaries=histories,
        )

    def _emit_progress(
        self,
        *,
        stream: TextIO | None,
        restart: int,
        round_idx: int,
        stage: str,
        round_record: dict,
    ) -> None:
        import sys

        out = stream or sys.stdout
        timing = round_record.get("timing_s", {})
        total_timing = sum(float(value) for value in timing.values())
        gpu_mem = self._cuda_memory_report()
        parts = [
            f"[fit] restart={restart}",
            f"round={round_idx}",
            f"stage={stage}",
            f"clusters={round_record.get('active_clusters')}",
            f"ll_before={round_record.get('log_likelihood_before'):.3f}",
            f"ll_after={round_record.get('log_likelihood_after'):.3f}",
            f"accepted={round_record.get('n_accepted')}",
            f"positive={round_record.get('n_positive')}",
            f"timing_s={total_timing:.3f}",
        ]
        for key in (
            "configure_s",
            "full_merge_s",
            "exact_cell_reassign_s",
            "greedy_merge_s",
            "proposal_s",
            "scoring_s",
            "conflict_s",
            "commit_s",
            "serial_refine_s",
            "cluster_reassign_s",
            "perturbation_s",
            "tau_update_s",
        ):
            if key in timing:
                parts.append(f"{key}={float(timing[key]):.3f}")
        if gpu_mem is not None:
            allocated_mb, reserved_mb = gpu_mem
            parts.append(f"cuda_mb={allocated_mb:.0f}/{reserved_mb:.0f}")
        print(" ".join(parts), file=out, flush=True)

    def _emit_verbose(
        self,
        *,
        stream: TextIO | None,
        restart: int,
        round_idx: int,
        stage: str,
        message: str,
    ) -> None:
        import sys

        out = stream or sys.stdout
        print(f"[trace] restart={restart} round={round_idx} stage={stage} {message}", file=out, flush=True)

    def _cuda_memory_report(self) -> tuple[float, float] | None:
        try:
            from .backends import torch
        except Exception:  # pragma: no cover - defensive
            return None
        if torch is None or not torch.cuda.is_available():
            return None
        try:
            allocated = float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
            reserved = float(torch.cuda.memory_reserved() / (1024.0 * 1024.0))
        except Exception:  # pragma: no cover - defensive
            return None
        return allocated, reserved

    def _make_restart_state(
        self,
        restart: int,
        *,
        restart_inits: list[str] | None,
    ) -> PartitionState:
        if restart == 0:
            return self.initial_state.copy()
        if restart_inits is None or not restart_inits:
            init = ("leiden_overclustered", "random", "singletons")[(restart - 1) % 3]
        else:
            init = restart_inits[(restart - 1) % len(restart_inits)]
        n_clusters = self._restart_target_clusters(restart)
        return PartitionState.from_csr(
            self.initial_state.X,
            init=init,
            seed=None if self.seed is None else self.seed + restart,
            n_clusters=n_clusters,
        )

    def _restart_target_clusters(self, restart: int) -> int | None:
        if self.leiden_restart_targets:
            return int(self.leiden_restart_targets[(restart - 1) % len(self.leiden_restart_targets)])
        if self.target_clusters is not None:
            base = max(1, int(self.target_clusters))
        else:
            base = max(8, int(8.0 * np.sqrt(max(self.initial_state.n_cells, 1))))
        multipliers = (2.0, 4.0, 1.0, 8.0)
        target = int(round(base * multipliers[(restart - 1) % len(multipliers)]))
        return max(1, min(self.initial_state.n_cells, target))

    def _configure_stage(self, state: PartitionState, stage: str) -> None:
        if not self.staged_search:
            self.sampler.set_family_weights(**self.base_family_weights)
            return
        del state
        self.sampler.set_family_weights(**self._stage_weights(stage))

    def _stage_name(self, state: PartitionState) -> str:
        active = len(state.active_cluster_ids)
        n_cells = state.n_cells
        target = self._target_cluster_count(state)
        if active > max(target * 2, n_cells // 8):
            return "coarsen"
        if active > max(target // 2, int(3.0 * np.sqrt(max(n_cells, 1)))):
            return "balanced"
        return "refine"

    def _target_cluster_count(self, state: PartitionState) -> int:
        if self.target_clusters is not None:
            return max(1, min(state.n_cells, int(self.target_clusters)))
        return max(16, int(8.0 * np.sqrt(max(state.n_cells, 1))))

    def _stage_weights(self, stage: str) -> dict[str, float]:
        if stage == "coarsen":
            return {
                "merge": 0.94,
                "peel": 0.02,
                "move": 0.03,
                "block_peel": 0.005,
                "block_move": 0.005,
            }
        if stage == "balanced":
            return {
                "merge": 0.70,
                "peel": 0.05,
                "move": 0.12,
                "block_peel": 0.07,
                "block_move": 0.06,
            }
        return {
            "merge": 0.26,
            "peel": 0.10,
            "move": 0.22,
            "block_peel": 0.22,
            "block_move": 0.20,
        }

    def _select_positive_nonconflicting(
        self,
        proposals: list[Proposal],
        scores: np.ndarray,
    ) -> list[dict]:
        candidates = self._positive_candidates(proposals, scores)
        return self._select_positive_nonconflicting_candidates(candidates)

    def _positive_candidates(self, proposals: list[Proposal], scores: np.ndarray) -> list[dict]:
        candidates = [
            {"proposal": proposal, "delta": float(delta), "touch_set": proposal.touch_set()}
            for proposal, delta in zip(proposals, scores, strict=True)
            if delta > 0.0 and np.isfinite(delta)
        ]
        return candidates

    def _select_positive_nonconflicting_candidates(self, candidates: list[dict]) -> list[dict]:
        candidates.sort(key=lambda item: item["delta"], reverse=True)

        accepted: list[dict] = []
        touched: set[int] = set()
        for item in candidates:
            touch_set = set(item["touch_set"])
            if touched.isdisjoint(touch_set):
                accepted.append(item)
                touched.update(touch_set)
        return accepted

    def _should_run_full_merge(self, state: PartitionState, stage: str) -> bool:
        return (
            self.full_merge_stage
            and stage == "coarsen"
            and 1 < len(state.active_cluster_ids) <= self.full_merge_active_limit
        )

    def _full_merge_phase(self, state: PartitionState, backend, *, stage: str) -> dict:
        if not self._should_run_full_merge(state, stage):
            return {"n_steps": 0, "delta": 0.0, "operations": [], "stage": stage, "n_sweeps": 0}

        total_delta = 0.0
        operations: list[dict] = []
        sweeps_run = 0
        for _ in range(self.full_merge_max_sweeps):
            candidates = self._score_all_merge_candidates(state, backend)
            if not candidates:
                break
            accepted = self._select_positive_nonconflicting_candidates(candidates)
            if not accepted:
                break
            self._assert_disjoint_touch_sets(accepted)
            touched_clusters = self._commit_batch(state, accepted)
            self.sampler.notify_state_changed(state, touched_clusters)
            delta = float(sum(item["delta"] for item in accepted))
            total_delta += delta
            operations.extend(self._proposal_to_record(item["proposal"], item["delta"]) for item in accepted)
            sweeps_run += 1
        return {
            "n_steps": len(operations),
            "n_sweeps": sweeps_run,
            "delta": total_delta,
            "operations": operations,
            "stage": stage,
        }

    def _score_all_merge_candidates(self, state: PartitionState, backend) -> list[dict]:
        active = state.active_cluster_array()
        if active.size < 2:
            return []

        candidates: list[dict] = []
        chunk: list[Proposal] = []
        chunk_size = max(1, self.full_merge_score_chunk)
        for i in range(active.size):
            cluster_b = int(active[i])
            for j in range(i):
                cluster_a = int(active[j])
                chunk.append(MergeProposal(cluster_a=min(cluster_a, cluster_b), cluster_b=max(cluster_a, cluster_b)))
                if len(chunk) >= chunk_size:
                    scores = backend.score_batch(state, chunk)
                    candidates.extend(self._positive_candidates(chunk, scores))
                    chunk = []
        if chunk:
            scores = backend.score_batch(state, chunk)
            candidates.extend(self._positive_candidates(chunk, scores))
        return candidates

    def _greedy_merge_sweep(self, state: PartitionState, backend, *, stage: str) -> dict:
        if self._should_run_full_merge(state, stage):
            return {"n_steps": 0, "delta": 0.0, "operations": [], "stage": stage}
        if self.greedy_merge_sweeps <= 0 or self.greedy_merge_candidates <= 0:
            return {"n_steps": 0, "delta": 0.0, "operations": [], "stage": stage}
        if len(state.active_cluster_ids) < 2:
            return {"n_steps": 0, "delta": 0.0, "operations": [], "stage": stage}

        total_delta = 0.0
        operations: list[dict] = []
        sweeps_run = 0
        for _ in range(self.greedy_merge_sweeps):
            self.sampler.prepare_round(state)
            proposals = self._build_greedy_merge_candidates(state, self.greedy_merge_candidates)
            if not proposals:
                break
            scores = backend.score_batch(state, proposals)
            accepted = self._select_positive_nonconflicting(proposals, scores)
            if not accepted:
                break
            self._assert_disjoint_touch_sets(accepted)
            touched_clusters = self._commit_batch(state, accepted)
            self.sampler.notify_state_changed(state, touched_clusters)
            delta = float(sum(item["delta"] for item in accepted))
            total_delta += delta
            operations.extend(self._proposal_to_record(item["proposal"], item["delta"]) for item in accepted)
            sweeps_run += 1
        return {
            "n_steps": len(operations),
            "n_sweeps": sweeps_run,
            "delta": total_delta,
            "operations": operations,
            "stage": stage,
        }

    def _build_greedy_merge_candidates(self, state: PartitionState, limit: int) -> list[Proposal]:
        unique: dict[tuple, Proposal] = {}

        for cluster_a, cluster_b in self.sampler._deterministic_merge_pairs[: int(limit)]:
            if cluster_a not in state.active_cluster_ids or cluster_b not in state.active_cluster_ids:
                continue
            proposal = MergeProposal(int(cluster_a), int(cluster_b))
            unique[self._proposal_key(proposal)] = proposal
            if len(unique) >= int(limit):
                return list(unique.values())

        active = state.active_cluster_array()
        for cluster_id in active.tolist():
            for target_cluster in self.sampler.merge_neighbors_for(int(cluster_id), self.sampler.top_merge_neighbors):
                if int(target_cluster) not in state.active_cluster_ids or int(cluster_id) == int(target_cluster):
                    continue
                cluster_a, cluster_b = sorted((int(cluster_id), int(target_cluster)))
                proposal = MergeProposal(cluster_a, cluster_b)
                unique[self._proposal_key(proposal)] = proposal
                if len(unique) >= int(limit):
                    return list(unique.values())

        attempts = 0
        while len(unique) < int(limit) and active.size >= 2 and attempts < int(limit) * 4:
            choice = self.rng.choice(active, size=2, replace=False)
            cluster_a, cluster_b = sorted((int(choice[0]), int(choice[1])))
            proposal = MergeProposal(cluster_a, cluster_b)
            unique[self._proposal_key(proposal)] = proposal
            attempts += 1
        return list(unique.values())

    def _should_run_exact_cell_reassign(self, state: PartitionState) -> bool:
        return (
            self.exact_cell_reassign_passes > 0
            and 1 < len(state.active_cluster_ids) <= self.exact_cell_reassign_active_limit
        )

    def _exact_cell_reassignment_sweep(self, state: PartitionState, backend, *, stage: str) -> dict:
        del stage
        if not self._should_run_exact_cell_reassign(state):
            return {"n_steps": 0, "delta": 0.0, "operations": [], "n_passes": 0}

        total_delta = 0.0
        operations: list[dict] = []
        best_delta_per_cell = np.zeros(state.n_cells, dtype=np.float64)
        cell_iter = self.rng.permutation(state.n_cells).astype(np.int64, copy=False)
        passes_run = 0

        for _ in range(self.exact_cell_reassign_passes):
            move_count = 0
            for cell in cell_iter:
                proposal, delta = self._best_cell_reassignment(state, backend, int(cell))
                best_delta_per_cell[int(cell)] = max(float(delta), 0.0)
                if proposal is None or not np.isfinite(delta) or delta <= 0.0:
                    continue
                touched_clusters = self._commit_proposal(state, proposal)
                self.sampler.notify_state_changed(state, touched_clusters)
                total_delta += float(delta)
                operations.append(self._proposal_to_record(proposal, float(delta)))
                move_count += 1

            passes_run += 1
            if move_count == 0:
                break
            focus_count = max(1, int(np.ceil(state.n_cells * self.exact_cell_reassign_top_fraction)))
            cell_iter = np.argsort(-best_delta_per_cell)[:focus_count].astype(np.int64, copy=False)

        return {
            "n_steps": len(operations),
            "n_passes": passes_run,
            "delta": total_delta,
            "operations": operations,
        }

    def _best_cell_reassignment(
        self,
        state: PartitionState,
        backend,
        cell: int,
    ) -> tuple[Proposal | None, float]:
        cell = int(cell)
        source_cluster = int(state.z[cell])
        best_proposal: Proposal | None = None
        best_delta = 0.0

        if state.cluster_size(source_cluster) > 1:
            peel = PeelProposal(cell=cell, source_cluster=source_cluster)
            peel_delta = float(backend.score_batch(state, [peel])[0])
            if np.isfinite(peel_delta) and peel_delta > best_delta:
                best_proposal = peel
                best_delta = peel_delta

        active = state.active_cluster_array()
        if active.size <= 1:
            return best_proposal, best_delta
        targets = active[active != source_cluster]
        chunk_size = max(1, self.exact_cell_score_chunk)
        for start in range(0, targets.size, chunk_size):
            target_chunk = targets[start : start + chunk_size]
            proposals = [
                MoveProposal(cell=cell, source_cluster=source_cluster, target_cluster=int(target_cluster))
                for target_cluster in target_chunk.tolist()
            ]
            if not proposals:
                continue
            scores = backend.score_batch(state, proposals)
            if scores.size == 0:
                continue
            best_idx = int(np.argmax(scores))
            delta = float(scores[best_idx])
            if np.isfinite(delta) and delta > best_delta:
                best_delta = delta
                best_proposal = proposals[best_idx]
        return best_proposal, best_delta

    def _cluster_reassignment_sweep(self, state: PartitionState, backend, *, stage: str) -> dict:
        if self.cluster_reassign_sweeps <= 0 or len(state.active_cluster_ids) < 2:
            return {"n_steps": 0, "delta": 0.0, "operations": [], "stage": stage}
        if stage == "coarsen":
            max_sources = min(self.cluster_reassign_max_sources, max(8, self._target_cluster_count(state) // 2))
        else:
            max_sources = self.cluster_reassign_max_sources

        total_delta = 0.0
        operations: list[dict] = []
        sweeps_run = 0
        for _ in range(self.cluster_reassign_sweeps):
            self.sampler.prepare_round(state)
            proposals = self._build_cluster_reassignment_candidates(state, max_sources=max_sources)
            if not proposals:
                break
            scores = backend.score_batch(state, proposals)
            accepted = self._select_positive_nonconflicting(proposals, scores)
            if not accepted:
                break
            self._assert_disjoint_touch_sets(accepted)
            touched_clusters = self._commit_batch(state, accepted)
            self.sampler.notify_state_changed(state, touched_clusters)
            delta = float(sum(item["delta"] for item in accepted))
            total_delta += delta
            operations.extend(self._proposal_to_record(item["proposal"], item["delta"]) for item in accepted)
            sweeps_run += 1
        return {
            "n_steps": len(operations),
            "n_sweeps": sweeps_run,
            "delta": total_delta,
            "operations": operations,
            "stage": stage,
        }

    def _build_cluster_reassignment_candidates(
        self,
        state: PartitionState,
        *,
        max_sources: int,
    ) -> list[Proposal]:
        active = [int(cluster_id) for cluster_id in state.active_cluster_ids]
        if len(active) < 2:
            return []
        sources = sorted(active, key=lambda cluster_id: (state.cluster_size(cluster_id), state.cluster_total(cluster_id)))
        sources = sources[: max(0, int(max_sources))]
        unique: dict[tuple, Proposal] = {}
        for source_cluster in sources:
            genes, values = state.clusters[source_cluster].sorted_items()
            if genes.size:
                order = np.argsort(values)[::-1][: self.sampler.signature_top_genes]
                targets = self.sampler.rank_target_clusters(
                    state,
                    source_cluster,
                    genes[order],
                    values[order],
                    limit=self.sampler.top_merge_neighbors,
                )
            else:
                targets = []
            if not targets:
                targets = [
                    int(cluster_id)
                    for cluster_id in self.sampler.merge_neighbors_for(source_cluster, self.sampler.top_merge_neighbors)
                ]
            for target_cluster in targets:
                if source_cluster == int(target_cluster) or int(target_cluster) not in state.active_cluster_ids:
                    continue
                cluster_a, cluster_b = sorted((source_cluster, int(target_cluster)))
                proposal = MergeProposal(cluster_a, cluster_b)
                unique[self._proposal_key(proposal)] = proposal
        return list(unique.values())

    def _perturbation_phase(
        self,
        state: PartitionState,
        backend,
        round_idx: int,
        stage: str,
    ) -> dict:
        if self.perturb_every <= 0 or self.perturb_steps <= 0:
            return {"n_steps": 0, "delta": 0.0, "operations": [], "enabled": False}
        if (int(round_idx) + 1) % self.perturb_every != 0:
            return {"n_steps": 0, "delta": 0.0, "operations": [], "enabled": True}

        temperature = max(self.perturb_temperature, np.finfo(np.float64).tiny)
        total_delta = 0.0
        positive_steps = 0
        negative_steps = 0
        operations: list[dict] = []
        self.sampler.prepare_round(state)
        for _ in range(self.perturb_steps):
            proposal = self._sample_perturbation_proposal(state)
            if proposal is None:
                continue
            delta = float(backend.score_batch(state, [proposal])[0])
            if not np.isfinite(delta):
                continue
            accept = delta > 0.0 or self.rng.random() < float(np.exp(min(0.0, delta / temperature)))
            if not accept:
                continue
            touched_clusters = self._commit_proposal(state, proposal)
            self.sampler.notify_state_changed(state, touched_clusters)
            self.sampler.prepare_round(state)
            total_delta += delta
            if delta > 0.0:
                positive_steps += 1
            else:
                negative_steps += 1
            operations.append(self._proposal_to_record(proposal, delta))

        cleanup = {"n_steps": 0, "delta": 0.0, "operations": [], "stage": stage}
        if self.perturb_positive_cleanup and operations:
            cleanup = self._greedy_merge_sweep(state, backend, stage=stage)
            total_delta += float(cleanup["delta"])
        return {
            "n_steps": len(operations),
            "positive_steps": positive_steps,
            "negative_steps": negative_steps,
            "positive_cleanup_steps": int(cleanup["n_steps"]),
            "delta": total_delta,
            "operations": operations,
            "cleanup": cleanup,
            "enabled": True,
        }

    def _sample_perturbation_proposal(self, state: PartitionState) -> Proposal | None:
        self.sampler.prepare_round(state)
        families = ("merge", "peel", "move", "block_peel", "block_move")
        weights = np.asarray((0.25, 0.20, 0.25, 0.15, 0.15), dtype=np.float64)
        for _ in range(32):
            family = str(self.rng.choice(families, p=weights))
            if family == "merge":
                proposal = self.sampler._sample_merge_uniform(state)
            elif family == "peel":
                proposal = self.sampler._sample_peel_uniform(state)
            elif family == "move":
                proposal = self.sampler._sample_move_uniform(state)
            elif family == "block_peel":
                proposal = self.sampler._sample_block_peel_uniform(state)
            else:
                proposal = self.sampler._sample_block_move_uniform(state)
            if proposal is not None:
                return proposal
        return None

    def _should_update_tau(self, round_idx: int) -> bool:
        if self.tau_update_interval <= 1:
            return True
        return (int(round_idx) + 1) % self.tau_update_interval == 0

    def _commit_batch(self, state: PartitionState, accepted: list[dict]) -> set[int]:
        touched_clusters: set[int] = set()
        for item in accepted:
            touched_clusters.update(self._commit_proposal(state, item["proposal"]))
        return touched_clusters

    def _commit_proposal(self, state: PartitionState, proposal: Proposal) -> set[int]:
        if isinstance(proposal, MergeProposal):
            state.merge_clusters(proposal.cluster_a, proposal.cluster_b)
        elif isinstance(proposal, PeelProposal):
            state.peel_cell_to_new_cluster(proposal.cell, proposal.source_cluster)
        elif isinstance(proposal, MoveProposal):
            state.move_cell(proposal.cell, proposal.source_cluster, proposal.target_cluster)
        elif isinstance(proposal, BlockPeelProposal):
            state.peel_block_to_new_cluster(
                proposal.block.cells,
                proposal.source_cluster,
                block_indices=proposal.block.indices,
                block_values=proposal.block.values,
            )
        elif isinstance(proposal, BlockMoveProposal):
            state.move_block(
                proposal.block.cells,
                proposal.source_cluster,
                proposal.target_cluster,
                block_indices=proposal.block.indices,
                block_values=proposal.block.values,
            )
        else:  # pragma: no cover - defensive branch
            raise TypeError(f"unsupported proposal type: {type(proposal)!r}")
        return set(state.last_touched_clusters)

    def _serial_refine(
        self,
        state: PartitionState,
        psi: np.ndarray,
        backend,
        touched_clusters: set[int],
        stage: str,
    ) -> dict:
        del psi
        if self.serial_refine_passes <= 0 or stage == "coarsen":
            return {"n_steps": 0, "delta": 0.0, "operations": []}
        total_delta = 0.0
        operations: list[dict] = []
        focus_clusters = [cluster_id for cluster_id in touched_clusters if cluster_id in state.active_cluster_ids]

        for _ in range(self.serial_refine_passes):
            self.sampler.prepare_round(state)
            candidates = self._build_refinement_candidates(state, focus_clusters)
            if not candidates:
                break
            scores = backend.score_batch(state, candidates)
            best_idx = int(np.argmax(scores))
            best_delta = float(scores[best_idx])
            if not np.isfinite(best_delta) or best_delta <= 0.0:
                break
            proposal = candidates[best_idx]
            touched_clusters = self._commit_proposal(state, proposal)
            self.sampler.notify_state_changed(state, touched_clusters)
            focus_clusters = [cluster_id for cluster_id in touched_clusters if cluster_id in state.active_cluster_ids]
            total_delta += best_delta
            operations.append(self._proposal_to_record(proposal, best_delta))
        return {"n_steps": len(operations), "delta": total_delta, "operations": operations}

    def _build_refinement_candidates(
        self,
        state: PartitionState,
        focus_clusters: list[int],
    ) -> list[Proposal]:
        if not focus_clusters:
            focus_clusters = [int(cluster_id) for cluster_id in self.sampler._active_clusters[: min(12, len(self.sampler._active_clusters))]]
        proposals: list[Proposal] = []

        for cluster_id in focus_clusters:
            cluster_id = int(cluster_id)
            if cluster_id not in state.active_cluster_ids:
                continue

            for target_cluster in self.sampler.merge_neighbors_for(cluster_id, self.serial_refine_merge_candidates):
                target_cluster = int(target_cluster)
                if cluster_id == target_cluster or target_cluster not in state.active_cluster_ids:
                    continue
                cluster_a, cluster_b = sorted((cluster_id, target_cluster))
                proposals.append(MergeProposal(cluster_a, cluster_b))

            membership = state.cells_by_cluster[cluster_id]
            if len(membership) > 1:
                cells = np.asarray(membership.cells, dtype=np.int64)
                sample_size = min(self.serial_refine_cells_per_cluster, len(cells))
                sampled = self.rng.choice(cells, size=sample_size, replace=False)
                for cell in sampled:
                    cell = int(cell)
                    proposals.append(PeelProposal(cell=cell, source_cluster=cluster_id))
                    genes, values = state.cell_counts(cell)
                    for target_cluster in self.sampler.rank_target_clusters(
                        state,
                        cluster_id,
                        genes,
                        values,
                        limit=self.serial_refine_move_candidates,
                    ):
                        proposals.append(
                            MoveProposal(
                                cell=cell,
                                source_cluster=cluster_id,
                                target_cluster=int(target_cluster),
                            ),
                        )
                block_peel = self.sampler._sample_block_peel_biased(state)
                if block_peel is not None and block_peel.source_cluster == cluster_id:
                    proposals.append(block_peel)
                block_move = self.sampler._sample_block_move_biased(state)
                if block_move is not None and block_move.source_cluster == cluster_id:
                    proposals.append(block_move)

        unique: dict[tuple, Proposal] = {}
        for proposal in proposals:
            unique[self._proposal_key(proposal)] = proposal
        return list(unique.values())

    def _proposal_key(self, proposal: Proposal) -> tuple:
        if isinstance(proposal, MergeProposal):
            return ("merge", proposal.cluster_a, proposal.cluster_b)
        if isinstance(proposal, PeelProposal):
            return ("peel", proposal.cell, proposal.source_cluster)
        if isinstance(proposal, MoveProposal):
            return ("move", proposal.cell, proposal.source_cluster, proposal.target_cluster)
        if isinstance(proposal, BlockPeelProposal):
            return ("block_peel", proposal.source_cluster, proposal.block.cells)
        if isinstance(proposal, BlockMoveProposal):
            return ("block_move", proposal.source_cluster, proposal.target_cluster, proposal.block.cells)
        raise TypeError(f"unsupported proposal type: {type(proposal)!r}")

    def _assert_disjoint_touch_sets(self, accepted: list[dict]) -> None:
        touched: set[int] = set()
        for item in accepted:
            touch_set = set(item["touch_set"])
            if not touched.isdisjoint(touch_set):
                raise AssertionError("accepted operations do not have disjoint touch sets")
            touched.update(touch_set)

    def _relative_improvement_below_threshold(
        self,
        history: list[dict],
        improvement_window: int,
        eta: float,
    ) -> bool:
        if len(history) < improvement_window:
            return False
        tail = history[-improvement_window:]
        start = float(tail[0]["log_likelihood_before"])
        end = float(tail[-1]["log_likelihood_after"])
        relative = (end - start) / max(1.0, abs(start))
        return relative < eta

    def _proposal_to_record(self, proposal: Proposal, delta: float) -> dict:
        if isinstance(proposal, MergeProposal):
            return {
                "kind": "merge",
                "cluster_a": proposal.cluster_a,
                "cluster_b": proposal.cluster_b,
                "delta": delta,
            }
        if isinstance(proposal, PeelProposal):
            return {
                "kind": "peel",
                "cell": proposal.cell,
                "source_cluster": proposal.source_cluster,
                "delta": delta,
            }
        if isinstance(proposal, MoveProposal):
            return {
                "kind": "move",
                "cell": proposal.cell,
                "source_cluster": proposal.source_cluster,
                "target_cluster": proposal.target_cluster,
                "delta": delta,
            }
        if isinstance(proposal, BlockPeelProposal):
            return {
                "kind": "block_peel",
                "cells": list(proposal.block.cells),
                "source_cluster": proposal.source_cluster,
                "delta": delta,
            }
        if isinstance(proposal, BlockMoveProposal):
            return {
                "kind": "block_move",
                "cells": list(proposal.block.cells),
                "source_cluster": proposal.source_cluster,
                "target_cluster": proposal.target_cluster,
                "delta": delta,
            }
        raise TypeError(f"unsupported proposal type: {type(proposal)!r}")
