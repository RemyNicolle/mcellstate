from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
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
    GPU_MODE = "gpu"
    CPU_ONLY_MODE = "cpu-only"
    OPTIMIZER_MODES = (EFFECTIVE_MODE, GPU_MODE, CPU_ONLY_MODE)
    CELLSTATES_LIKE_POLICY = "cellstates_like"
    GPU_STRUCTURED_POLICY = "gpu_structured"
    GPU_FAST_WEAK_POLICY = "gpu_fast_weak"
    SEARCH_POLICIES = (
        CELLSTATES_LIKE_POLICY,
        GPU_STRUCTURED_POLICY,
        GPU_FAST_WEAK_POLICY,
    )

    def __init__(
        self,
        *,
        state: PartitionState,
        psi: np.ndarray,
        optimizer_mode: str = EFFECTIVE_MODE,
        search_policy: str | None = None,
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
        proposal_workers: int | None = None,
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
        random_proposals: bool | None = None,
        max_scored_proposals: int | None = None,
        random_accept_prob: float | None = None,
        random_accept_max_fraction: float | None = None,
        proposal_batch_size: int | None = None,
        cuda_chunk_size: int | None = None,
        recompute_ll_each_round: bool | None = None,
        cuda_empty_cache: bool = False,
        merge_closure_enabled: bool | None = None,
        merge_closure_max_batches: int = 8,
        merge_closure_stall_batches: int = 2,
        merge_closure_pairs_per_batch: int = 8192,
        merge_closure_epsilon_uniform: float = 0.05,
        merge_closure_candidate_mode: str = "idf_shared_gene",
        merge_closure_min_delta: float = 0.0,
        multi_greedy_trials: int = 4,
        batched_reassign_passes: int = 1,
        batched_reassign_cells: int = 512,
        batched_reassign_guided_targets: int = 8,
        batched_reassign_uniform_targets: int = 2,
        exhaustive_reassign_active_limit: int = 64,
        block_grouping_enabled: bool = True,
        block_grouping_min_cells: int = 2,
        oracle_reference_labels: np.ndarray | None = None,
    ) -> None:
        optimizer_mode = str(optimizer_mode).lower()
        if optimizer_mode not in self.OPTIMIZER_MODES:
            raise ValueError(
                f"unknown optimizer_mode {optimizer_mode!r}; expected one of {self.OPTIMIZER_MODES}",
            )
        self.initial_state = state.copy()
        self.psi = np.asarray(psi, dtype=np.float64)
        self.optimizer_mode = optimizer_mode
        self.backend_name = str(backend)
        self.search_policy = self._resolve_search_policy(
            search_policy=search_policy,
            optimizer_mode=optimizer_mode,
            backend=self.backend_name,
        )
        self.backend_threads = backend_threads
        self.n_proposals = int(n_proposals)
        self.proposal_workers = (
            None if proposal_workers is None else max(1, int(proposal_workers))
        )
        self.seed = seed
        self.validate_batches = bool(validate_batches)
        self.staged_search = bool(staged_search)
        self.target_clusters = None if target_clusters is None else int(target_clusters)
        self.leiden_restart_targets = (
            None
            if leiden_restart_targets is None
            else tuple(int(value) for value in leiden_restart_targets)
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
        self.random_proposals = (
            None if random_proposals is None else bool(random_proposals)
        )
        self.max_scored_proposals = (
            None
            if max_scored_proposals is None or int(max_scored_proposals) <= 0
            else int(max_scored_proposals)
        )
        self.random_accept_prob = (
            None if random_accept_prob is None else max(0.0, float(random_accept_prob))
        )
        self.random_accept_max_fraction = (
            None
            if random_accept_max_fraction is None
            else max(0.0, float(random_accept_max_fraction))
        )
        self.proposal_batch_size = (
            None
            if proposal_batch_size is None or int(proposal_batch_size) <= 0
            else int(proposal_batch_size)
        )
        self._proposal_batch_size_locked = self.proposal_batch_size is not None
        self.cuda_chunk_size = (
            None
            if cuda_chunk_size is None or int(cuda_chunk_size) <= 0
            else int(cuda_chunk_size)
        )
        self.recompute_ll_each_round = (
            None if recompute_ll_each_round is None else bool(recompute_ll_each_round)
        )
        self.cuda_empty_cache = bool(cuda_empty_cache)
        self.merge_closure_enabled = (
            None if merge_closure_enabled is None else bool(merge_closure_enabled)
        )
        self.merge_closure_max_batches = max(0, int(merge_closure_max_batches))
        self.merge_closure_stall_batches = max(1, int(merge_closure_stall_batches))
        self.merge_closure_pairs_per_batch = max(1, int(merge_closure_pairs_per_batch))
        self.merge_closure_epsilon_uniform = max(
            0.0, min(1.0, float(merge_closure_epsilon_uniform))
        )
        self.merge_closure_candidate_mode = str(merge_closure_candidate_mode)
        self.merge_closure_min_delta = float(merge_closure_min_delta)
        self.multi_greedy_trials = max(1, int(multi_greedy_trials))
        self.batched_reassign_passes = max(0, int(batched_reassign_passes))
        self.batched_reassign_cells = max(1, int(batched_reassign_cells))
        self.batched_reassign_guided_targets = max(
            0, int(batched_reassign_guided_targets)
        )
        self.batched_reassign_uniform_targets = max(
            0, int(batched_reassign_uniform_targets)
        )
        self.exhaustive_reassign_active_limit = max(
            2, int(exhaustive_reassign_active_limit)
        )
        self.block_grouping_enabled = bool(block_grouping_enabled)
        self.block_grouping_min_cells = max(2, int(block_grouping_min_cells))
        self.oracle_reference_labels = (
            None
            if oracle_reference_labels is None
            else np.asarray(oracle_reference_labels, dtype=np.int64)
        )
        self._apply_search_policy_defaults()
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
            proposal_workers=self.proposal_workers or 1,
            random_proposals=bool(self.random_proposals),
            max_unique_proposals=self.max_scored_proposals,
            seed=seed,
        )
        self.sampler.set_scoring_context(self.psi)

    def _resolve_search_policy(
        self,
        *,
        search_policy: str | None,
        optimizer_mode: str,
        backend: str,
    ) -> str:
        if search_policy is not None:
            resolved = str(search_policy).lower()
            if resolved not in self.SEARCH_POLICIES:
                raise ValueError(
                    f"unknown search_policy {resolved!r}; expected one of {self.SEARCH_POLICIES}",
                )
            return resolved
        if optimizer_mode == self.GPU_MODE:
            return self.GPU_FAST_WEAK_POLICY
        backend_name = str(backend).lower()
        if backend_name == "cuda":
            return self.GPU_STRUCTURED_POLICY
        return self.CELLSTATES_LIKE_POLICY

    def _apply_shared_defaults(self) -> None:
        if self.proposal_workers is None:
            self.proposal_workers = 1
        if self.random_proposals is None:
            self.random_proposals = False
        if self.random_accept_prob is None:
            self.random_accept_prob = 0.0
        if self.random_accept_max_fraction is None:
            self.random_accept_max_fraction = 0.0
        if self.proposal_batch_size is None:
            self.proposal_batch_size = min(
                int(self.n_proposals),
                max(4096, min(16_384, int(max(self.n_proposals // 8, 4096)))),
            )
        if self.recompute_ll_each_round is None:
            self.recompute_ll_each_round = True
        if self.cuda_chunk_size is None:
            self.cuda_chunk_size = 8192
        if self.merge_closure_enabled is None:
            self.merge_closure_enabled = False

    def _apply_search_policy_defaults(self) -> None:
        self._apply_shared_defaults()
        if self.search_policy == self.GPU_FAST_WEAK_POLICY:
            self.full_merge_stage = False
            self.greedy_merge_sweeps = 0
            self.exact_cell_reassign_passes = 0
            self.cluster_reassign_sweeps = 0
            self.perturb_every = 0
            self.perturb_steps = 0
            self.serial_refine_passes = 0
            if self.proposal_workers is None or self.proposal_workers <= 1:
                self.proposal_workers = max(2, min(32, os.cpu_count() or 2))
            if self.backend_threads is None:
                self.backend_threads = max(4, min(32, os.cpu_count() or 4))
            self.random_proposals = True
            if self.max_scored_proposals is None:
                self.max_scored_proposals = int(self.n_proposals)
            if self.random_accept_prob is None or self.random_accept_prob == 0.0:
                self.random_accept_prob = 0.002
            if (
                self.random_accept_max_fraction is None
                or self.random_accept_max_fraction == 0.0
            ):
                self.random_accept_max_fraction = 0.002
            if self.proposal_batch_size is None or self.proposal_batch_size < 8192:
                self.proposal_batch_size = min(
                    int(self.n_proposals),
                    max(8192, min(32_768, int(max(self.n_proposals // 4, 8192)))),
                )
            self.recompute_ll_each_round = False
            self.merge_closure_enabled = False
            return

        if self.search_policy == self.GPU_STRUCTURED_POLICY:
            if self.proposal_workers is None or self.proposal_workers <= 1:
                self.proposal_workers = max(2, min(32, os.cpu_count() or 2))
            if self.backend_threads is None:
                self.backend_threads = max(4, min(32, os.cpu_count() or 4))
            self.random_proposals = False
            if self.max_scored_proposals is None:
                self.max_scored_proposals = min(int(self.n_proposals), 20_000)
            if self.proposal_batch_size is None or self.proposal_batch_size < 8192:
                self.proposal_batch_size = min(
                    int(self.n_proposals),
                    max(8192, min(32_768, int(max(self.n_proposals // 4, 8192)))),
                )
            self.recompute_ll_each_round = False
            if (
                self.merge_closure_enabled is None
                or self.merge_closure_enabled is False
            ):
                self.merge_closure_enabled = True
            self.full_merge_stage = False
            self.greedy_merge_sweeps = max(1, int(self.greedy_merge_sweeps))
            self.exact_cell_reassign_passes = max(
                1, int(self.exact_cell_reassign_passes)
            )
            self.cluster_reassign_sweeps = max(1, int(self.cluster_reassign_sweeps))
            self.serial_refine_passes = max(1, int(self.serial_refine_passes))
            return

        if self.optimizer_mode == self.CPU_ONLY_MODE and self.backend_threads is None:
            self.backend_threads = max(2, min(8, os.cpu_count() or 2))
        if self.optimizer_mode == self.CPU_ONLY_MODE and self.proposal_workers == 1:
            self.proposal_workers = max(2, min(8, os.cpu_count() or 2))

    def fit(
        self,
        *,
        max_rounds: int | None = 1000,
        update_psi: bool = False,
        restarts: int | None = None,
        stall_rounds: int | None = None,
        improvement_window: int | None = None,
        eta: float | None = None,
        restart_inits: list[str] | None = None,
        progress: bool = False,
        verbose: bool = False,
        progress_stream: TextIO | None = None,
        verbose_stream: TextIO | None = None,
    ) -> FitResult:
        best_result: FitResult | None = None
        best_restart_state: PartitionState | None = None
        histories: list[dict] = []
        progress_stream = progress_stream or None
        verbose_stream = verbose_stream or None
        fit_restarts = (
            (
                8
                if restarts is None and self.search_policy == self.GPU_STRUCTURED_POLICY
                else 1
            )
            if restarts is None
            else max(1, int(restarts))
        )
        fit_stall_rounds = (
            25
            if stall_rounds is None and self.search_policy == self.GPU_STRUCTURED_POLICY
            else 10
            if stall_rounds is None
            else max(1, int(stall_rounds))
        )
        fit_improvement_window = (
            20
            if improvement_window is None
            and self.search_policy == self.GPU_STRUCTURED_POLICY
            else 5
            if improvement_window is None
            else max(1, int(improvement_window))
        )
        fit_eta = (
            1e-10
            if eta is None and self.search_policy == self.GPU_STRUCTURED_POLICY
            else 1e-8
            if eta is None
            else float(eta)
        )

        for restart in range(int(fit_restarts)):
            state = self._make_restart_state(
                restart,
                restart_inits=restart_inits,
                best_restart_state=best_restart_state,
            )
            psi = self.psi.copy()
            state.initialize_likelihood_cache(psi)
            self.sampler.notify_state_changed(state, set(state.active_cluster_ids))
            self.sampler.set_trace(
                (
                    lambda message, restart=restart, stream=verbose_stream: self._emit_sampler_trace(
                        stream=stream, restart=restart, message=message
                    )
                )
                if verbose
                else None
            )
            backend = make_backend(
                self.backend_name,
                psi,
                state,
                num_threads=self.backend_threads,
                chunk_size=self.cuda_chunk_size,
            )
            history: list[dict] = []
            current_ll = state.total_log_likelihood_cached(psi)
            stall = 0
            merge_closure_stall = 0
            reassign_stall = 0
            stop_reason = "completed"
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
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="merge closure phase",
                    )
                merge_closure = self._merge_closure_phase(
                    state, backend, stage=stage, reason="round"
                )
                timing["merge_closure_s"] = time.perf_counter() - timer
                if merge_closure["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(merge_closure["delta"]),
                        state,
                        psi,
                    )

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="full merge phase",
                    )
                full_merge = self._full_merge_phase(state, backend, stage=stage)
                timing["full_merge_s"] = time.perf_counter() - timer
                if full_merge["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(full_merge["delta"]),
                        state,
                        psi,
                    )

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="exact reassignment phase",
                    )
                if self.search_policy == self.GPU_STRUCTURED_POLICY:
                    exact_cell_reassign = self._batched_reassignment_cleanup(
                        state,
                        backend,
                        stage=stage,
                    )
                else:
                    exact_cell_reassign = self._exact_cell_reassignment_sweep(
                        state, backend, stage=stage
                    )
                timing["exact_cell_reassign_s"] = time.perf_counter() - timer
                if exact_cell_reassign["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(exact_cell_reassign["delta"]),
                        state,
                        psi,
                    )

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="greedy merge phase",
                    )
                greedy_merge = self._greedy_merge_sweep(state, backend, stage=stage)
                timing["greedy_merge_s"] = time.perf_counter() - timer
                if greedy_merge["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(greedy_merge["delta"]),
                        state,
                        psi,
                    )

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message=(
                            f"sample {self._effective_round_proposal_count()} proposals"
                        ),
                    )
                (
                    proposals,
                    scores,
                    proposal_s,
                    scoring_s,
                    proposal_chunks,
                ) = self._sample_and_score_proposals(state, backend)
                (
                    proposals,
                    scores,
                    move_grouping_stats,
                ) = self._augment_with_grouped_block_moves(
                    state,
                    backend,
                    proposals,
                    scores,
                )
                timing["proposal_s"] = proposal_s
                timing["scoring_s"] = scoring_s
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message=(
                            f"score {len(proposals)} proposals on {self.backend_name} "
                            f"in {proposal_chunks} chunks"
                        ),
                    )
                self._adapt_proposal_batch_size(
                    chunk_count=proposal_chunks,
                    proposal_s=proposal_s,
                    scoring_s=scoring_s,
                    backend=backend,
                )
                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="select non-conflicting proposals",
                    )
                accepted = self._select_positive_nonconflicting(
                    proposals, scores, backend=backend
                )
                family_diagnostics = self._family_diagnostics(
                    proposals,
                    scores,
                    accepted,
                )
                timing["conflict_s"] = time.perf_counter() - timer
                accepted_delta = float(sum(item["delta"] for item in accepted))
                random_walk_accepted = int(
                    sum(1 for item in accepted if item.get("random_walk", False))
                )
                accepted_positive_steps = int(
                    sum(1 for item in accepted if float(item["delta"]) > 0.0)
                )

                round_record = {
                    "restart": restart,
                    "round": round_idx,
                    "stage": stage,
                    "proposal_chunks": int(proposal_chunks),
                    "proposal_batch_size": None
                    if self.proposal_batch_size is None
                    else int(self.proposal_batch_size),
                    "family_weights": dict(
                        zip(
                            self.sampler.family_names,
                            self.sampler.family_weights.tolist(),
                            strict=True,
                        ),
                    ),
                    "n_proposals": len(proposals),
                    "n_positive": int(np.sum(scores > 0.0)),
                    "n_accepted": len(accepted),
                    "n_random_walk": random_walk_accepted,
                    "merge_closure": merge_closure,
                    "full_merge": full_merge,
                    "exact_cell_reassign": exact_cell_reassign,
                    "greedy_merge": greedy_merge,
                    "family_diagnostics": family_diagnostics,
                    "move_grouping": move_grouping_stats,
                    "accepted_delta": accepted_delta,
                    "log_likelihood_before": round_before_ll,
                    "touch_sets": [sorted(item["touch_set"]) for item in accepted],
                    "operations": [
                        self._proposal_to_record(item["proposal"], item["delta"])
                        for item in accepted
                    ],
                }
                if self.oracle_reference_labels is not None:
                    round_record["oracle_reference"] = self._oracle_reference_report(
                        state,
                        backend,
                    )

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
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        accepted_delta,
                        state,
                        psi,
                    )
                timing["commit_s"] = time.perf_counter() - timer

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="serial refine phase",
                    )
                refinement = self._serial_refine(
                    state, psi, backend, touched_clusters, stage
                )
                timing["serial_refine_s"] = time.perf_counter() - timer
                if refinement["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(refinement["delta"]),
                        state,
                        psi,
                    )

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="cluster reassignment phase",
                    )
                cluster_reassign = self._cluster_reassignment_sweep(
                    state, backend, stage=stage
                )
                timing["cluster_reassign_s"] = time.perf_counter() - timer
                if cluster_reassign["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(cluster_reassign["delta"]),
                        state,
                        psi,
                    )

                timer = time.perf_counter()
                if verbose:
                    self._emit_verbose(
                        stream=verbose_stream,
                        restart=restart,
                        round_idx=round_idx,
                        stage=stage,
                        message="perturbation phase",
                    )
                perturbation = self._perturbation_phase(
                    state, backend, round_idx, stage
                )
                timing["perturbation_s"] = time.perf_counter() - timer
                if perturbation["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(perturbation["delta"]),
                        state,
                        psi,
                    )

                total_delta = (
                    float(merge_closure["delta"])
                    + float(full_merge["delta"])
                    + float(exact_cell_reassign["delta"])
                    + float(greedy_merge["delta"])
                    + accepted_delta
                    + float(refinement["delta"])
                    + float(cluster_reassign["delta"])
                    + float(perturbation["delta"])
                )
                after_partition_ll = self._round_log_likelihood_after(
                    round_before_ll,
                    total_delta,
                    state,
                    psi,
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
                    delta_tolerance = 1e-8 * max(
                        1.0, abs(observed_delta), abs(expected_total_delta)
                    )
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
                    backend = make_backend(
                        self.backend_name,
                        psi,
                        state,
                        num_threads=self.backend_threads,
                        chunk_size=self.cuda_chunk_size,
                    )
                    self.sampler.set_scoring_context(psi)
                    tau_update = {"updated": True, "delta": float(current_ll - old_ll)}
                    timing["tau_update_s"] = time.perf_counter() - timer
                else:
                    timing["tau_update_s"] = 0.0
                round_record["tau_update"] = tau_update
                round_record["log_likelihood_after"] = current_ll
                round_record["timing_s"] = timing
                round_record["search_policy"] = self.search_policy
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
                    int(merge_closure["n_steps"])
                    + int(full_merge["n_steps"])
                    + int(exact_cell_reassign["n_steps"])
                    + int(greedy_merge["n_steps"])
                    + accepted_positive_steps
                    + random_walk_accepted
                    + int(refinement["n_steps"])
                    + int(cluster_reassign["n_steps"])
                    + int(perturbation.get("positive_cleanup_steps", 0))
                )
                merge_closure_stall = (
                    merge_closure_stall + 1 if int(merge_closure["n_steps"]) == 0 else 0
                )
                reassign_stall = (
                    reassign_stall + 1
                    if int(exact_cell_reassign["n_steps"]) == 0
                    else 0
                )
                stall = stall + 1 if n_positive_steps == 0 else 0
                if (
                    stall >= fit_stall_rounds
                    and merge_closure_stall >= self.merge_closure_stall_batches
                    and reassign_stall >= 1
                ):
                    stop_reason = "stall"
                    round_record["stop_reason"] = stop_reason
                    break

                if (
                    self._relative_improvement_below_threshold(
                        history, fit_improvement_window, fit_eta
                    )
                    and merge_closure_stall >= self.merge_closure_stall_batches
                    and reassign_stall >= 1
                ):
                    stop_reason = "improvement_window"
                    round_record["stop_reason"] = stop_reason
                    break
                round_idx += 1

            if history and "stop_reason" not in history[-1]:
                history[-1]["stop_reason"] = stop_reason
            if self.search_policy == self.GPU_STRUCTURED_POLICY:
                final_merge_closure = self._merge_closure_phase(
                    state,
                    backend,
                    stage=self._stage_name(state),
                    reason="final_cleanup",
                )
                if final_merge_closure["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(final_merge_closure["delta"]),
                        state,
                        psi,
                    )
                final_reassign = self._batched_reassignment_cleanup(
                    state,
                    backend,
                    stage=self._stage_name(state),
                )
                if final_reassign["delta"] != 0.0:
                    current_ll = self._advance_log_likelihood(
                        current_ll,
                        float(final_reassign["delta"]),
                        state,
                        psi,
                    )
                if history:
                    history[-1]["final_cleanup"] = {
                        "merge_closure": final_merge_closure,
                        "reassignment": final_reassign,
                    }
            exact_final_ll = full_partition_log_likelihood(state, psi)
            if history:
                history[-1]["log_likelihood_after"] = exact_final_ll
                history[-1]["partition_log_likelihood_after"] = exact_final_ll
            result = FitResult(
                state=state,
                z=state.z.copy(),
                log_likelihood=exact_final_ll,
                history=history,
                psi=psi.copy(),
                restart_summaries=None,
            )
            histories.append(
                {
                    "restart": restart,
                    "history": history,
                    "log_likelihood": exact_final_ll,
                }
            )
            if (
                best_result is None
                or result.log_likelihood > best_result.log_likelihood
            ):
                best_result = result
                best_restart_state = result.state.copy()

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
            f"random_walk={round_record.get('n_random_walk', 0)}",
            f"timing_s={total_timing:.3f}",
        ]
        for key in (
            "configure_s",
            "merge_closure_s",
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
        print(
            f"[trace] restart={restart} round={round_idx} stage={stage} {message}",
            file=out,
            flush=True,
        )

    def _emit_sampler_trace(
        self,
        *,
        stream: TextIO | None,
        restart: int,
        message: str,
    ) -> None:
        import sys

        out = stream or sys.stdout
        print(f"[trace] restart={restart} sampler {message}", file=out, flush=True)

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

    def _empty_cuda_cache(self) -> None:
        try:
            from .backends import torch
        except Exception:  # pragma: no cover - defensive
            return
        if torch is None or not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()

    def _advance_log_likelihood(
        self,
        current_ll: float,
        delta: float,
        state: PartitionState,
        psi: np.ndarray,
    ) -> float:
        if self.recompute_ll_each_round or self.validate_batches:
            return full_partition_log_likelihood(state, psi)
        return float(current_ll + float(delta))

    def _round_log_likelihood_after(
        self,
        round_before_ll: float,
        total_delta: float,
        state: PartitionState,
        psi: np.ndarray,
    ) -> float:
        if self.recompute_ll_each_round or self.validate_batches:
            return full_partition_log_likelihood(state, psi)
        return float(round_before_ll + float(total_delta))

    def _proposal_chunk_sizes(self, total: int) -> list[int]:
        total = max(0, int(total))
        if total <= 0:
            return []
        batch_size = (
            total
            if self.proposal_batch_size is None
            else max(1, int(self.proposal_batch_size))
        )
        if batch_size >= total:
            return [total]
        counts: list[int] = []
        remaining = total
        while remaining > 0:
            count = min(batch_size, remaining)
            counts.append(count)
            remaining -= count
        return counts

    def _effective_round_proposal_count(self) -> int:
        total = max(0, int(self.n_proposals))
        if self.max_scored_proposals is None:
            return total
        return min(total, max(0, int(self.max_scored_proposals)))

    def _sample_proposal_chunk(
        self,
        state: PartitionState,
        backend,
        count: int,
        seed: int,
    ) -> list[Proposal]:
        count = int(count)
        if count <= 0:
            return []
        if self.search_policy == self.GPU_FAST_WEAK_POLICY and self.random_proposals:
            backend_sampler = getattr(backend, "sample_random_proposals", None)
            if backend_sampler is not None:
                return backend_sampler(
                    state,
                    count,
                    family_weights=self.sampler.family_weights,
                    max_unique_proposals=self.max_scored_proposals,
                    seed=seed,
                )
        self.sampler.prepare_round(state)
        return self.sampler.sample_batch(state, count)

    def _sample_and_score_proposals(
        self, state: PartitionState, backend
    ) -> tuple[list[Proposal], np.ndarray, float, float, int]:
        chunk_sizes = self._proposal_chunk_sizes(self._effective_round_proposal_count())
        if not chunk_sizes:
            return [], np.empty(0, dtype=np.float64), 0.0, 0.0, 0

        proposal_batches: list[list[Proposal]] = []
        score_batches: list[np.ndarray] = []
        proposal_s = 0.0
        scoring_s = 0.0
        seeds = self.rng.integers(
            np.iinfo(np.int64).max, size=len(chunk_sizes), dtype=np.int64
        )

        if len(chunk_sizes) == 1:
            start = time.perf_counter()
            batch = self._sample_proposal_chunk(
                state, backend, chunk_sizes[0], int(seeds[0])
            )
            proposal_s += time.perf_counter() - start
            start = time.perf_counter()
            score_batch = backend.score_batch(state, batch)
            if self.cuda_empty_cache:
                self._empty_cuda_cache()
            scoring_s += time.perf_counter() - start
            return batch, score_batch, proposal_s, scoring_s, 1

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._sample_proposal_chunk,
                state,
                backend,
                chunk_sizes[0],
                int(seeds[0]),
            )
            for idx, count in enumerate(chunk_sizes):
                start = time.perf_counter()
                batch = future.result()
                proposal_s += time.perf_counter() - start
                if idx + 1 < len(chunk_sizes):
                    future = executor.submit(
                        self._sample_proposal_chunk,
                        state,
                        backend,
                        chunk_sizes[idx + 1],
                        int(seeds[idx + 1]),
                    )
                start = time.perf_counter()
                score_batch = backend.score_batch(state, batch)
                if self.cuda_empty_cache:
                    self._empty_cuda_cache()
                scoring_s += time.perf_counter() - start
                proposal_batches.append(batch)
                score_batches.append(score_batch)

        proposals = [proposal for batch in proposal_batches for proposal in batch]
        scores = (
            np.concatenate(score_batches)
            if score_batches
            else np.empty(0, dtype=np.float64)
        )
        return proposals, scores, proposal_s, scoring_s, len(chunk_sizes)

    def _adapt_proposal_batch_size(
        self, *, chunk_count: int, proposal_s: float, scoring_s: float, backend
    ) -> None:
        if self._proposal_batch_size_locked:
            return
        if self.proposal_batch_size is None:
            return
        if chunk_count <= 1:
            return
        round_total = max(1, self._effective_round_proposal_count())
        current = max(1, int(self.proposal_batch_size))
        should_grow = chunk_count > 4 or proposal_s > 1.5 * scoring_s
        if should_grow:
            current = min(round_total, max(current + 1, current * 2))
        elif (
            getattr(backend, "device", None) is not None
            and getattr(backend.device, "type", None) == "cuda"
            and current < round_total
        ):
            current = min(round_total, max(current + 1, current * 2))
        self.proposal_batch_size = current

    def _make_restart_state(
        self,
        restart: int,
        *,
        restart_inits: list[str] | None,
        best_restart_state: PartitionState | None,
    ) -> PartitionState:
        if restart == 0:
            return self.initial_state.copy()
        if restart_inits is None or not restart_inits:
            init = ("leiden_overclustered", "random_online", "singletons")[
                (restart - 1) % 3
            ]
        else:
            init = restart_inits[(restart - 1) % len(restart_inits)]
        if isinstance(init, str) and init == "perturbed_best":
            if best_restart_state is None:
                init = "leiden_overclustered"
            else:
                return self._perturb_restart_state(
                    best_restart_state,
                    seed=None if self.seed is None else self.seed + restart,
                )
        n_clusters = self._restart_target_clusters(restart)
        return PartitionState.from_csr(
            self.initial_state.X,
            init=init,
            seed=None if self.seed is None else self.seed + restart,
            n_clusters=n_clusters,
        )

    def _perturb_restart_state(
        self,
        state: PartitionState,
        *,
        seed: int | None,
    ) -> PartitionState:
        rng = np.random.default_rng(seed)
        z = state.z.copy()
        n_cells = int(z.size)
        if n_cells <= 1:
            return PartitionState.from_assignment(state.X, z)
        active_labels = np.unique(z)
        if active_labels.size <= 1:
            return PartitionState.from_assignment(state.X, z)
        n_perturb = min(
            n_cells - 1,
            max(1, min(64, int(np.ceil(0.05 * float(n_cells))))),
        )
        selected = rng.choice(n_cells, size=n_perturb, replace=False)
        next_label = int(active_labels.max()) + 1
        for cell in np.asarray(selected, dtype=np.int64).tolist():
            current = int(z[int(cell)])
            if rng.random() < 0.75:
                candidates = active_labels[active_labels != current]
                if candidates.size:
                    z[int(cell)] = int(rng.choice(candidates))
                    continue
            z[int(cell)] = next_label
            next_label += 1
        return PartitionState.from_assignment(state.X, z)

    def _restart_target_clusters(self, restart: int) -> int | None:
        if self.leiden_restart_targets:
            return int(
                self.leiden_restart_targets[
                    (restart - 1) % len(self.leiden_restart_targets)
                ]
            )
        if self.target_clusters is not None:
            base = max(1, int(self.target_clusters))
        else:
            base = max(8, int(8.0 * np.sqrt(max(self.initial_state.n_cells, 1))))
        multipliers = (2.0, 4.0, 1.0, 8.0)
        target = int(round(base * multipliers[(restart - 1) % len(multipliers)]))
        return max(1, min(self.initial_state.n_cells, target))

    def _configure_stage(self, state: PartitionState, stage: str) -> None:
        if self.search_policy == self.GPU_FAST_WEAK_POLICY:
            self.sampler.set_family_weights(**self._gpu_fast_weak_stage_weights(stage))
            return
        if self.search_policy == self.GPU_STRUCTURED_POLICY:
            self.sampler.set_family_weights(**self._gpu_structured_stage_weights(stage))
            return
        if self.optimizer_mode == self.CPU_ONLY_MODE:
            self.sampler.set_family_weights(
                merge=0.42,
                peel=0.08,
                move=0.26,
                block_peel=0.10,
                block_move=0.14,
            )
            return
        if not self.staged_search:
            self.sampler.set_family_weights(**self.base_family_weights)
            return
        del state
        self.sampler.set_family_weights(**self._stage_weights(stage))

    def _gpu_fast_weak_stage_weights(self, stage: str) -> dict[str, float]:
        if stage == "coarsen":
            return {
                "merge": 0.75,
                "peel": 0.05,
                "move": 0.20,
                "block_peel": 0.0,
                "block_move": 0.0,
            }
        if stage == "balanced":
            return {
                "merge": 0.55,
                "peel": 0.10,
                "move": 0.35,
                "block_peel": 0.0,
                "block_move": 0.0,
            }
        return {
            "merge": 0.35,
            "peel": 0.10,
            "move": 0.55,
            "block_peel": 0.0,
            "block_move": 0.0,
        }

    def _gpu_structured_stage_weights(self, stage: str) -> dict[str, float]:
        if stage == "coarsen":
            return {
                "merge": 0.85,
                "move": 0.10,
                "peel": 0.0,
                "block_move": 0.05,
                "block_peel": 0.0,
            }
        if stage == "balanced":
            return {
                "merge": 0.45,
                "move": 0.30,
                "block_move": 0.20,
                "peel": 0.05,
                "block_peel": 0.0,
            }
        return {
            "merge": 0.20,
            "move": 0.40,
            "block_move": 0.25,
            "peel": 0.10,
            "block_peel": 0.05,
        }

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
        *,
        backend=None,
    ) -> list[dict]:
        candidates = self._positive_candidates(proposals, scores)
        candidates.extend(self._random_walk_candidates(proposals, scores))
        return self._select_positive_nonconflicting_candidates(
            candidates, backend=backend
        )

    def _positive_candidates(
        self, proposals: list[Proposal], scores: np.ndarray
    ) -> list[dict]:
        candidates = [
            {
                "proposal": proposal,
                "delta": float(delta),
                "touch_set": proposal.touch_set(),
                "touch_ids": proposal.touch_ids(),
            }
            for proposal, delta in zip(proposals, scores, strict=True)
            if delta > 0.0 and np.isfinite(delta)
        ]
        return candidates

    def _random_walk_candidates(
        self, proposals: list[Proposal], scores: np.ndarray
    ) -> list[dict]:
        if self.random_accept_prob is None or self.random_accept_prob <= 0.0:
            return []
        if (
            self.random_accept_max_fraction is None
            or self.random_accept_max_fraction <= 0.0
        ):
            return []
        max_candidates = int(
            np.floor(len(proposals) * float(self.random_accept_max_fraction))
        )
        if max_candidates <= 0:
            return []

        selected: list[dict] = []
        for proposal, delta in zip(proposals, scores, strict=True):
            delta = float(delta)
            if delta > 0.0 or not np.isfinite(delta):
                continue
            if isinstance(proposal, MergeProposal):
                continue
            if self.rng.random() >= float(self.random_accept_prob):
                continue
            selected.append(
                {
                    "proposal": proposal,
                    "delta": delta,
                    "touch_set": proposal.touch_set(),
                    "touch_ids": proposal.touch_ids(),
                    "random_walk": True,
                },
            )

        if len(selected) <= max_candidates:
            return selected
        keep = self.rng.choice(len(selected), size=max_candidates, replace=False)
        return [selected[int(idx)] for idx in keep.tolist()]

    def _select_positive_nonconflicting_candidates(
        self, candidates: list[dict], *, backend=None
    ) -> list[dict]:
        if backend is not None and self.multi_greedy_trials <= 1:
            backend_selector = getattr(
                backend, "select_nonconflicting_candidates", None
            )
            if backend_selector is not None:
                return backend_selector(candidates)
        if not candidates:
            return []
        ordered = sorted(
            candidates, key=lambda item: float(item["delta"]), reverse=True
        )
        best = self._greedy_accept_candidates(ordered)
        best_delta = float(sum(float(item["delta"]) for item in best))
        if self.multi_greedy_trials <= 1 or len(ordered) <= 1:
            return best
        delta_scale = max(
            1e-12,
            max(abs(float(item["delta"])) for item in ordered) * 1e-9,
        )
        for _ in range(self.multi_greedy_trials - 1):
            trial_order = self._shuffle_near_equal_candidates(
                ordered, delta_tolerance=delta_scale
            )
            accepted = self._greedy_accept_candidates(trial_order)
            accepted_delta = float(sum(float(item["delta"]) for item in accepted))
            if accepted_delta > best_delta:
                best = accepted
                best_delta = accepted_delta
        return best

    def _shuffle_near_equal_candidates(
        self,
        ordered: list[dict],
        *,
        delta_tolerance: float,
    ) -> list[dict]:
        shuffled: list[dict] = []
        start = 0
        while start < len(ordered):
            end = start + 1
            base_delta = float(ordered[start]["delta"])
            while end < len(ordered):
                if abs(float(ordered[end]["delta"]) - base_delta) > delta_tolerance:
                    break
                end += 1
            block = ordered[start:end]
            if len(block) > 1:
                perm = self.rng.permutation(len(block))
                block = [block[int(idx)] for idx in perm.tolist()]
            shuffled.extend(block)
            start = end
        return shuffled

    def _greedy_accept_candidates(self, ordered: list[dict]) -> list[dict]:
        accepted: list[dict] = []
        touched: set[int] = set()
        for item in ordered:
            touch_set = set(item["touch_set"])
            if touched.isdisjoint(touch_set):
                accepted.append(item)
                touched.update(touch_set)
        return accepted

    def _candidate_family(self, proposal: Proposal) -> str:
        return str(proposal.kind)

    def _family_diagnostics(
        self,
        proposals: list[Proposal],
        scores: np.ndarray,
        accepted: list[dict],
    ) -> dict[str, dict[str, float | int]]:
        diagnostics: dict[str, dict[str, float | int]] = {
            family: {
                "n_proposed": 0,
                "n_scored": 0,
                "n_finite": 0,
                "n_positive": 0,
                "n_conflict_rejected": 0,
                "n_committed": 0,
                "sum_delta_committed": 0.0,
                "mean_delta_positive": 0.0,
                "max_delta_positive": 0.0,
            }
            for family in self.sampler.family_names
        }
        positive_values: dict[str, list[float]] = {
            family: [] for family in self.sampler.family_names
        }
        for proposal, delta in zip(proposals, scores, strict=True):
            family = self._candidate_family(proposal)
            record = diagnostics.setdefault(family, {})
            record["n_proposed"] = int(record.get("n_proposed", 0)) + 1
            record["n_scored"] = int(record.get("n_scored", 0)) + 1
            delta = float(delta)
            if np.isfinite(delta):
                record["n_finite"] = int(record.get("n_finite", 0)) + 1
                if delta > 0.0:
                    record["n_positive"] = int(record.get("n_positive", 0)) + 1
                    positive_values[family].append(delta)
        for item in accepted:
            family = self._candidate_family(item["proposal"])
            record = diagnostics.setdefault(family, {})
            record["n_committed"] = int(record.get("n_committed", 0)) + 1
            record["sum_delta_committed"] = float(
                record.get("sum_delta_committed", 0.0)
            ) + float(item["delta"])
        for family, record in diagnostics.items():
            positives = positive_values.get(family, [])
            n_positive = int(record.get("n_positive", 0))
            n_committed = int(record.get("n_committed", 0))
            record["n_conflict_rejected"] = max(0, n_positive - n_committed)
            if positives:
                record["mean_delta_positive"] = float(np.mean(positives))
                record["max_delta_positive"] = float(np.max(positives))
        return diagnostics

    def _oracle_reference_report(
        self,
        state: PartitionState,
        backend,
        *,
        max_pairs: int = 128,
    ) -> dict | None:
        if self.oracle_reference_labels is None:
            return None
        reference = np.asarray(self.oracle_reference_labels, dtype=np.int64)
        if reference.shape != (state.n_cells,):
            return None
        groups_by_reference: dict[int, list[tuple[int, float, int]]] = {}
        for cluster_id in sorted(state.active_cluster_ids):
            cells = np.asarray(
                state.cells_by_cluster[int(cluster_id)].cells, dtype=np.int64
            )
            labels = reference[cells]
            unique, counts = np.unique(labels, return_counts=True)
            if unique.size == 0:
                continue
            best_idx = int(np.argmax(counts))
            purity = float(counts[best_idx]) / float(len(cells))
            if purity < 0.6:
                continue
            ref_id = int(unique[best_idx])
            groups_by_reference.setdefault(ref_id, []).append(
                (int(cluster_id), purity, int(len(cells)))
            )
        proposals: list[Proposal] = []
        annotated_pairs: list[tuple[int, int, int]] = []
        candidate_groups: list[dict] = []
        for ref_id, groups in groups_by_reference.items():
            groups = sorted(groups, key=lambda item: (-item[1], item[2], item[0]))
            if len(groups) >= 3:
                candidate_groups.append(
                    {
                        "reference_cluster": int(ref_id),
                        "current_clusters": [
                            int(cluster_id) for cluster_id, _, _ in groups
                        ],
                    }
                )
            for idx in range(len(groups)):
                for jdx in range(idx):
                    cluster_a = int(groups[jdx][0])
                    cluster_b = int(groups[idx][0])
                    proposals.append(
                        MergeProposal(
                            cluster_a=min(cluster_a, cluster_b),
                            cluster_b=max(cluster_a, cluster_b),
                        )
                    )
                    annotated_pairs.append((int(ref_id), cluster_a, cluster_b))
                    if len(proposals) >= max_pairs:
                        break
                if len(proposals) >= max_pairs:
                    break
            if len(proposals) >= max_pairs:
                break
        if not proposals:
            return {
                "positive_pairs": 0,
                "total_possible_gain": 0.0,
                "top_positive_merges": [],
                "candidate_groups": candidate_groups,
            }
        scores = backend.score_batch(state, proposals)
        positive_items = []
        total_gain = 0.0
        for (ref_id, cluster_a, cluster_b), delta in zip(
            annotated_pairs, scores, strict=True
        ):
            delta = float(delta)
            if not np.isfinite(delta) or delta <= 0.0:
                continue
            total_gain += delta
            positive_items.append(
                {
                    "reference_cluster": int(ref_id),
                    "cluster_a": int(cluster_a),
                    "cluster_b": int(cluster_b),
                    "delta": delta,
                }
            )
        positive_items.sort(key=lambda item: float(item["delta"]), reverse=True)
        return {
            "positive_pairs": int(len(positive_items)),
            "total_possible_gain": float(total_gain),
            "top_positive_merges": positive_items[:10],
            "candidate_groups": candidate_groups[:10],
        }

    def _augment_with_grouped_block_moves(
        self,
        state: PartitionState,
        backend,
        proposals: list[Proposal],
        scores: np.ndarray,
    ) -> tuple[list[Proposal], np.ndarray, dict[str, int]]:
        stats = {
            "positive_single_moves": 0,
            "source_target_groups": 0,
            "positive_blocks": 0,
        }
        if not self.block_grouping_enabled or len(proposals) == 0:
            return proposals, scores, stats
        positive_moves: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for proposal, delta in zip(proposals, scores, strict=True):
            if not isinstance(proposal, MoveProposal):
                continue
            delta = float(delta)
            if not np.isfinite(delta) or delta <= 0.0:
                continue
            stats["positive_single_moves"] += 1
            key = (int(proposal.source_cluster), int(proposal.target_cluster))
            positive_moves.setdefault(key, []).append((int(proposal.cell), delta))
        if not positive_moves:
            return proposals, scores, stats
        stats["source_target_groups"] = len(positive_moves)
        grouped_proposals: list[Proposal] = []
        grouped_scores: list[float] = []
        existing_keys = {self._proposal_key(proposal) for proposal in proposals}
        for (
            source_cluster,
            target_cluster,
        ), cells_and_scores in positive_moves.items():
            cells = tuple(sorted({int(cell) for cell, _ in cells_and_scores}))
            if len(cells) < self.block_grouping_min_cells:
                continue
            if len(cells) >= state.cluster_size(int(source_cluster)):
                continue
            from .proposals import BlockPayload  # noqa: PLC0415

            block = BlockPayload.from_cells(state, cells)
            proposal = BlockMoveProposal(
                block=block,
                source_cluster=int(source_cluster),
                target_cluster=int(target_cluster),
            )
            key = self._proposal_key(proposal)
            if key in existing_keys:
                continue
            delta = float(backend.score_batch(state, [proposal])[0])
            if not np.isfinite(delta) or delta <= 0.0:
                continue
            grouped_proposals.append(proposal)
            grouped_scores.append(delta)
            existing_keys.add(key)
            stats["positive_blocks"] += 1
        if not grouped_proposals:
            return proposals, scores, stats
        return (
            [*proposals, *grouped_proposals],
            np.concatenate(
                [
                    np.asarray(scores, dtype=np.float64),
                    np.asarray(grouped_scores, dtype=np.float64),
                ]
            ),
            stats,
        )

    def _should_run_merge_closure(self, state: PartitionState, stage: str) -> bool:
        if not self.merge_closure_enabled:
            return False
        if len(state.active_cluster_ids) < 2:
            return False
        if stage == "coarsen":
            return True
        if stage == "balanced":
            return True
        return self.search_policy == self.GPU_STRUCTURED_POLICY

    def _build_merge_closure_candidates(
        self,
        state: PartitionState,
        *,
        n_pairs: int,
    ) -> list[Proposal]:
        if self.merge_closure_candidate_mode == "idf_shared_gene":
            return self.sampler.sample_guided_merge_pairs(
                state,
                n_pairs,
                epsilon_uniform=self.merge_closure_epsilon_uniform,
                include_cached_pairs=True,
            )
        return self._build_greedy_merge_candidates(state, n_pairs)

    def _merge_closure_phase(
        self,
        state: PartitionState,
        backend,
        *,
        stage: str,
        reason: str,
    ) -> dict:
        if not self._should_run_merge_closure(state, stage):
            return {
                "n_steps": 0,
                "delta": 0.0,
                "operations": [],
                "stage": stage,
                "reason": reason,
                "n_batches": 0,
                "stall_batches": 0,
                "batches": [],
            }
        total_delta = 0.0
        operations: list[dict] = []
        batch_records: list[dict] = []
        stall_batches = 0
        for batch_idx in range(self.merge_closure_max_batches):
            self.sampler.prepare_round(state)
            proposals = self._build_merge_closure_candidates(
                state,
                n_pairs=self.merge_closure_pairs_per_batch,
            )
            if not proposals:
                stall_batches += 1
                batch_records.append(
                    {
                        "batch": int(batch_idx),
                        "n_proposed": 0,
                        "n_scored": 0,
                        "n_positive": 0,
                        "n_committed": 0,
                        "delta": 0.0,
                    }
                )
                if stall_batches >= self.merge_closure_stall_batches:
                    break
                continue
            scores = backend.score_batch(state, proposals)
            positive = [
                item
                for item in self._positive_candidates(proposals, scores)
                if float(item["delta"]) > float(self.merge_closure_min_delta)
            ]
            accepted = self._select_positive_nonconflicting_candidates(
                positive, backend=backend
            )
            batch_delta = float(sum(float(item["delta"]) for item in accepted))
            batch_records.append(
                {
                    "batch": int(batch_idx),
                    "n_proposed": int(len(proposals)),
                    "n_scored": int(len(proposals)),
                    "n_positive": int(len(positive)),
                    "n_committed": int(len(accepted)),
                    "delta": batch_delta,
                }
            )
            if not accepted:
                stall_batches += 1
                if stall_batches >= self.merge_closure_stall_batches:
                    break
                continue
            stall_batches = 0
            self._assert_disjoint_touch_sets(accepted)
            touched_clusters = self._commit_batch(state, accepted)
            self.sampler.notify_state_changed(state, touched_clusters)
            total_delta += batch_delta
            operations.extend(
                self._proposal_to_record(item["proposal"], item["delta"])
                for item in accepted
            )
        return {
            "n_steps": len(operations),
            "delta": total_delta,
            "operations": operations,
            "stage": stage,
            "reason": reason,
            "n_batches": len(batch_records),
            "stall_batches": stall_batches,
            "batches": batch_records,
        }

    def _should_run_full_merge(self, state: PartitionState, stage: str) -> bool:
        return (
            self.full_merge_stage
            and stage == "coarsen"
            and 1 < len(state.active_cluster_ids) <= self.full_merge_active_limit
        )

    def _full_merge_phase(self, state: PartitionState, backend, *, stage: str) -> dict:
        if not self._should_run_full_merge(state, stage):
            return {
                "n_steps": 0,
                "delta": 0.0,
                "operations": [],
                "stage": stage,
                "n_sweeps": 0,
            }

        total_delta = 0.0
        operations: list[dict] = []
        sweeps_run = 0
        for _ in range(self.full_merge_max_sweeps):
            candidates = self._score_all_merge_candidates(state, backend)
            if not candidates:
                break
            accepted = self._select_positive_nonconflicting_candidates(
                candidates, backend=backend
            )
            if not accepted:
                break
            self._assert_disjoint_touch_sets(accepted)
            touched_clusters = self._commit_batch(state, accepted)
            self.sampler.notify_state_changed(state, touched_clusters)
            delta = float(sum(item["delta"] for item in accepted))
            total_delta += delta
            operations.extend(
                self._proposal_to_record(item["proposal"], item["delta"])
                for item in accepted
            )
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
                chunk.append(
                    MergeProposal(
                        cluster_a=min(cluster_a, cluster_b),
                        cluster_b=max(cluster_a, cluster_b),
                    )
                )
                if len(chunk) >= chunk_size:
                    scores = backend.score_batch(state, chunk)
                    candidates.extend(self._positive_candidates(chunk, scores))
                    chunk = []
        if chunk:
            scores = backend.score_batch(state, chunk)
            candidates.extend(self._positive_candidates(chunk, scores))
        return candidates

    def _greedy_merge_sweep(
        self, state: PartitionState, backend, *, stage: str
    ) -> dict:
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
            proposals = self._build_greedy_merge_candidates(
                state, self.greedy_merge_candidates
            )
            if not proposals:
                break
            scores = backend.score_batch(state, proposals)
            accepted = self._select_positive_nonconflicting(
                proposals, scores, backend=backend
            )
            if not accepted:
                break
            self._assert_disjoint_touch_sets(accepted)
            touched_clusters = self._commit_batch(state, accepted)
            self.sampler.notify_state_changed(state, touched_clusters)
            delta = float(sum(item["delta"] for item in accepted))
            total_delta += delta
            operations.extend(
                self._proposal_to_record(item["proposal"], item["delta"])
                for item in accepted
            )
            sweeps_run += 1
        return {
            "n_steps": len(operations),
            "n_sweeps": sweeps_run,
            "delta": total_delta,
            "operations": operations,
            "stage": stage,
        }

    def _build_greedy_merge_candidates(
        self, state: PartitionState, limit: int
    ) -> list[Proposal]:
        unique: dict[tuple, Proposal] = {}

        for cluster_a, cluster_b in self.sampler._deterministic_merge_pairs[
            : int(limit)
        ]:
            if (
                cluster_a not in state.active_cluster_ids
                or cluster_b not in state.active_cluster_ids
            ):
                continue
            proposal = MergeProposal(int(cluster_a), int(cluster_b))
            unique[self._proposal_key(proposal)] = proposal
            if len(unique) >= int(limit):
                return list(unique.values())

        active = state.active_cluster_array()
        for cluster_id in active.tolist():
            for target_cluster in self.sampler.merge_neighbors_for(
                int(cluster_id), self.sampler.top_merge_neighbors
            ):
                if int(target_cluster) not in state.active_cluster_ids or int(
                    cluster_id
                ) == int(target_cluster):
                    continue
                cluster_a, cluster_b = sorted((int(cluster_id), int(target_cluster)))
                proposal = MergeProposal(cluster_a, cluster_b)
                unique[self._proposal_key(proposal)] = proposal
                if len(unique) >= int(limit):
                    return list(unique.values())

        attempts = 0
        while (
            len(unique) < int(limit) and active.size >= 2 and attempts < int(limit) * 4
        ):
            choice = self.rng.choice(active, size=2, replace=False)
            cluster_a, cluster_b = sorted((int(choice[0]), int(choice[1])))
            proposal = MergeProposal(cluster_a, cluster_b)
            unique[self._proposal_key(proposal)] = proposal
            attempts += 1
        return list(unique.values())

    def _allow_peel_in_stage(self, state: PartitionState, stage: str) -> bool:
        if stage != "coarsen":
            return True
        return len(state.active_cluster_ids) <= self._target_cluster_count(state)

    def _batched_reassignment_cells(self, state: PartitionState) -> np.ndarray:
        if state.n_cells <= self.batched_reassign_cells:
            return self.rng.permutation(state.n_cells).astype(np.int64, copy=False)
        sampled = self.rng.choice(
            state.n_cells,
            size=self.batched_reassign_cells,
            replace=False,
        )
        return np.asarray(sampled, dtype=np.int64)

    def _batched_reassignment_cleanup(
        self,
        state: PartitionState,
        backend,
        *,
        stage: str,
    ) -> dict:
        if self.batched_reassign_passes <= 0 or len(state.active_cluster_ids) < 2:
            return {
                "n_steps": 0,
                "delta": 0.0,
                "operations": [],
                "n_passes": 0,
                "move_grouping": {
                    "positive_single_moves": 0,
                    "source_target_groups": 0,
                    "positive_blocks": 0,
                },
            }
        total_delta = 0.0
        operations: list[dict] = []
        passes_run = 0
        grouping_totals = {
            "positive_single_moves": 0,
            "source_target_groups": 0,
            "positive_blocks": 0,
        }
        allow_peel = self._allow_peel_in_stage(state, stage)

        for _ in range(self.batched_reassign_passes):
            self.sampler.prepare_round(state)
            cells = self._batched_reassignment_cells(state)
            proposals: dict[tuple, Proposal] = {}
            exhaustive = (
                len(state.active_cluster_ids) <= self.exhaustive_reassign_active_limit
            )
            for cell in cells.tolist():
                cell = int(cell)
                source_cluster = int(state.z[cell])
                if source_cluster not in state.active_cluster_ids:
                    continue
                genes, values = state.cell_counts(cell)
                if exhaustive:
                    targets = [
                        int(cluster_id)
                        for cluster_id in state.active_cluster_array().tolist()
                        if int(cluster_id) != source_cluster
                    ]
                else:
                    candidate_targets = self.sampler._candidate_targets_for_payload(
                        state,
                        source_cluster,
                        np.asarray(genes, dtype=np.int64),
                        np.asarray(values, dtype=np.int64),
                    )
                    ranked_targets = self.sampler.rank_target_clusters(
                        state,
                        source_cluster,
                        np.asarray(genes, dtype=np.int64),
                        np.asarray(values, dtype=np.int64),
                        limit=self.batched_reassign_guided_targets,
                        candidate_targets=candidate_targets,
                    )
                    targets = [int(target) for target in ranked_targets]
                    if self.batched_reassign_uniform_targets > 0:
                        active = state.active_cluster_array()
                        uniform_pool = active[active != source_cluster]
                        if uniform_pool.size:
                            take = min(
                                int(self.batched_reassign_uniform_targets),
                                int(uniform_pool.size),
                            )
                            extra = self.rng.choice(
                                uniform_pool,
                                size=take,
                                replace=False,
                            )
                            targets.extend(int(target) for target in extra.tolist())
                    targets = [
                        int(cluster_id)
                        for cluster_id in dict.fromkeys(targets)
                        if int(cluster_id) != source_cluster
                    ]
                for target_cluster in targets:
                    proposal = MoveProposal(
                        cell=cell,
                        source_cluster=source_cluster,
                        target_cluster=int(target_cluster),
                    )
                    proposals[self._proposal_key(proposal)] = proposal
                if allow_peel and state.cluster_size(source_cluster) > 1:
                    peel = PeelProposal(cell=cell, source_cluster=source_cluster)
                    proposals[self._proposal_key(peel)] = peel
            if not proposals:
                break

            proposal_list = list(proposals.values())
            scores = backend.score_batch(state, proposal_list)
            best_by_cell: dict[int, tuple[Proposal, float]] = {}
            for proposal, delta in zip(proposal_list, scores, strict=True):
                delta = float(delta)
                if not np.isfinite(delta) or delta <= 0.0:
                    continue
                if isinstance(proposal, (MoveProposal, PeelProposal)):
                    cell = int(proposal.cell)
                    best = best_by_cell.get(cell)
                    if best is None or delta > best[1]:
                        best_by_cell[cell] = (proposal, delta)
            if not best_by_cell:
                passes_run += 1
                break
            selected_proposals = [proposal for proposal, _ in best_by_cell.values()]
            selected_scores = np.asarray(
                [delta for _, delta in best_by_cell.values()],
                dtype=np.float64,
            )
            (
                selected_proposals,
                selected_scores,
                grouping_stats,
            ) = self._augment_with_grouped_block_moves(
                state,
                backend,
                selected_proposals,
                selected_scores,
            )
            for key, value in grouping_stats.items():
                grouping_totals[key] += int(value)
            accepted = self._select_positive_nonconflicting(
                selected_proposals,
                selected_scores,
                backend=backend,
            )
            if not accepted:
                passes_run += 1
                break
            self._assert_disjoint_touch_sets(accepted)
            touched_clusters = self._commit_batch(state, accepted)
            self.sampler.notify_state_changed(state, touched_clusters)
            batch_delta = float(sum(float(item["delta"]) for item in accepted))
            total_delta += batch_delta
            operations.extend(
                self._proposal_to_record(item["proposal"], item["delta"])
                for item in accepted
            )
            passes_run += 1
        return {
            "n_steps": len(operations),
            "delta": total_delta,
            "operations": operations,
            "n_passes": passes_run,
            "move_grouping": grouping_totals,
        }

    def _should_run_exact_cell_reassign(self, state: PartitionState) -> bool:
        return (
            self.exact_cell_reassign_passes > 0
            and 1
            < len(state.active_cluster_ids)
            <= self.exact_cell_reassign_active_limit
        )

    def _exact_cell_reassignment_sweep(
        self, state: PartitionState, backend, *, stage: str
    ) -> dict:
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
                proposal, delta = self._best_cell_reassignment(
                    state, backend, int(cell)
                )
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
            focus_count = max(
                1, int(np.ceil(state.n_cells * self.exact_cell_reassign_top_fraction))
            )
            cell_iter = np.argsort(-best_delta_per_cell)[:focus_count].astype(
                np.int64, copy=False
            )

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
                MoveProposal(
                    cell=cell,
                    source_cluster=source_cluster,
                    target_cluster=int(target_cluster),
                )
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

    def _cluster_reassignment_sweep(
        self, state: PartitionState, backend, *, stage: str
    ) -> dict:
        if self.cluster_reassign_sweeps <= 0 or len(state.active_cluster_ids) < 2:
            return {"n_steps": 0, "delta": 0.0, "operations": [], "stage": stage}
        if stage == "coarsen":
            max_sources = min(
                self.cluster_reassign_max_sources,
                max(8, self._target_cluster_count(state) // 2),
            )
        else:
            max_sources = self.cluster_reassign_max_sources

        total_delta = 0.0
        operations: list[dict] = []
        sweeps_run = 0
        for _ in range(self.cluster_reassign_sweeps):
            self.sampler.prepare_round(state)
            proposals = self._build_cluster_reassignment_candidates(
                state, max_sources=max_sources
            )
            if not proposals:
                break
            scores = backend.score_batch(state, proposals)
            accepted = self._select_positive_nonconflicting(
                proposals, scores, backend=backend
            )
            if not accepted:
                break
            self._assert_disjoint_touch_sets(accepted)
            touched_clusters = self._commit_batch(state, accepted)
            self.sampler.notify_state_changed(state, touched_clusters)
            delta = float(sum(item["delta"] for item in accepted))
            total_delta += delta
            operations.extend(
                self._proposal_to_record(item["proposal"], item["delta"])
                for item in accepted
            )
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
        sources = sorted(
            active,
            key=lambda cluster_id: (
                state.cluster_size(cluster_id),
                state.cluster_total(cluster_id),
            ),
        )
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
                    for cluster_id in self.sampler.merge_neighbors_for(
                        source_cluster, self.sampler.top_merge_neighbors
                    )
                ]
            for target_cluster in targets:
                if (
                    source_cluster == int(target_cluster)
                    or int(target_cluster) not in state.active_cluster_ids
                ):
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
            accept = delta > 0.0 or self.rng.random() < float(
                np.exp(min(0.0, delta / temperature))
            )
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
            state.move_cell(
                proposal.cell, proposal.source_cluster, proposal.target_cluster
            )
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
        focus_clusters = [
            cluster_id
            for cluster_id in touched_clusters
            if cluster_id in state.active_cluster_ids
        ]

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
            focus_clusters = [
                cluster_id
                for cluster_id in touched_clusters
                if cluster_id in state.active_cluster_ids
            ]
            total_delta += best_delta
            operations.append(self._proposal_to_record(proposal, best_delta))
        return {
            "n_steps": len(operations),
            "delta": total_delta,
            "operations": operations,
        }

    def _build_refinement_candidates(
        self,
        state: PartitionState,
        focus_clusters: list[int],
    ) -> list[Proposal]:
        if not focus_clusters:
            focus_clusters = [
                int(cluster_id)
                for cluster_id in self.sampler._active_clusters[
                    : min(12, len(self.sampler._active_clusters))
                ]
            ]
        proposals: list[Proposal] = []

        for cluster_id in focus_clusters:
            cluster_id = int(cluster_id)
            if cluster_id not in state.active_cluster_ids:
                continue

            for target_cluster in self.sampler.merge_neighbors_for(
                cluster_id, self.serial_refine_merge_candidates
            ):
                target_cluster = int(target_cluster)
                if (
                    cluster_id == target_cluster
                    or target_cluster not in state.active_cluster_ids
                ):
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
            return (
                "move",
                proposal.cell,
                proposal.source_cluster,
                proposal.target_cluster,
            )
        if isinstance(proposal, BlockPeelProposal):
            return ("block_peel", proposal.source_cluster, proposal.block.cells)
        if isinstance(proposal, BlockMoveProposal):
            return (
                "block_move",
                proposal.source_cluster,
                proposal.target_cluster,
                proposal.block.cells,
            )
        raise TypeError(f"unsupported proposal type: {type(proposal)!r}")

    def _assert_disjoint_touch_sets(self, accepted: list[dict]) -> None:
        touched: set[int] = set()
        for item in accepted:
            touch_set = set(item["touch_set"])
            if not touched.isdisjoint(touch_set):
                raise AssertionError(
                    "accepted operations do not have disjoint touch sets"
                )
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
