from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .abundance import abundance_logbf, fit_simplet_size_prior
from .likelihood import (
    cell_lambda_maps,
    make_lambda_grid,
    mixture_score_for_pair,
    parent_log_probs,
    parent_scores_for_target,
    self_loo_score,
    validate_psi,
)
from .search import generate_parent_pairs, pair_prior_logprob, top_parent_candidates
from .state import ClusterState, build_cluster_state


@dataclass(frozen=True)
class AuditThresholds:
    """Conservative default call thresholds.

    The defaults intentionally avoid overcalling. A likely doublet must survive
    the pair-search penalty, improve over the best single parent, and have a
    non-extreme mixture weight.
    """

    threshold_total: float = 10.0
    threshold_possible_expr: float = 0.0
    threshold_parent_gain: float = 5.0
    threshold_parent_gain_per_umi: float = 1e-4
    lambda_edge_margin: float = 0.05
    min_cells_confident: int = 3
    min_umis_confident: int = 10


@dataclass(frozen=True)
class AuditResult:
    cluster_table: pd.DataFrame
    state: ClusterState
    lambda_grid: np.ndarray
    lambda_posteriors: dict[Any, np.ndarray] | None = None


def _confidence_flag(n_cells: int, n_umi: float, thresholds: AuditThresholds) -> str:
    if n_cells <= 1:
        return "singleton_low_confidence"
    if n_cells < thresholds.min_cells_confident or n_umi < thresholds.min_umis_confident:
        return "small_cluster_low_confidence"
    return "ok"


def _call(row: dict[str, Any], thresholds: AuditThresholds) -> str:
    lam = row["lambda_map"]
    non_extreme = thresholds.lambda_edge_margin <= lam <= 1.0 - thresholds.lambda_edge_margin
    parent_gain_ok = (
        row["expr_gain_vs_best_parent"] > thresholds.threshold_parent_gain
        and row["parent_gain_per_UMI"] > thresholds.threshold_parent_gain_per_umi
    )
    if (
        row["total_log_odds_minus_search_penalty"] > thresholds.threshold_total
        and row["expr_logBF_doublet_vs_self"] > thresholds.threshold_possible_expr
        and parent_gain_ok
        and non_extreme
        and row["confidence_flag"] == "ok"
    ):
        return "likely_doublet"
    if (
        row["expr_gain_vs_best_parent"] > thresholds.threshold_parent_gain
        and row["expr_logBF_doublet_vs_self"] > thresholds.threshold_possible_expr
        and non_extreme
    ):
        return "possible_doublet"
    return "likely_simplet"


def audit_doublets(
    X: Any,
    z: Any,
    psi: np.ndarray,
    doublet_rate: float = 0.05,
    top_m_parents: int = 50,
    lambda_grid_size: int = 101,
    beta_lambda: tuple[float, float] = (2.0, 2.0),
    abundance_weight: float = 1.0,
    log_prior_odds_doublet_vs_simplet: float = 0.0,
    phi_doublet: float = 2.0,
    pair_prior_mode: str = "collision_normalized",
    candidate_parents: Any = None,
    include_homotypic: bool = False,
    return_cell_lambda: bool = False,
    return_lambda_posterior: bool = False,
    thresholds: AuditThresholds | None = None,
    backend: str = "cpu",
) -> AuditResult:
    """Annotate existing clusters as likely simplets or doublet-like mixtures.

    This is a post-partition audit. It never changes ``z`` and it uses only raw
    count likelihoods plus an optional weak abundance model.
    """

    if backend != "cpu":
        raise NotImplementedError("Only backend='cpu' is currently implemented")
    if not (0.0 < doublet_rate < 1.0):
        raise ValueError("doublet_rate must be between 0 and 1")
    thresholds = thresholds or AuditThresholds()
    state = build_cluster_state(X, z)
    psi = validate_psi(psi, state.n_genes)
    lambda_grid = make_lambda_grid(lambda_grid_size)
    log_q = parent_log_probs(state, psi)

    rows: list[dict[str, Any]] = []
    lambda_posteriors: dict[Any, np.ndarray] = {}

    for k in range(state.n_clusters):
        cluster_id = state.cluster_ids[k]
        l_self = self_loo_score(state, k, psi)
        parent_scores = parent_scores_for_target(state, k, log_q)
        parent_scores[k] = -np.inf
        if np.all(~np.isfinite(parent_scores)):
            best_single = -1
            best_single_score = -math.inf
        else:
            best_single = int(np.nanargmax(parent_scores))
            best_single_score = float(parent_scores[best_single])

        candidates = top_parent_candidates(parent_scores, k, top_m_parents, candidate_parents)
        pairs = generate_parent_pairs(candidates, include_homotypic=include_homotypic)
        target = state.C.getrow(k)
        target_indices = target.indices
        target_counts = target.data.astype(np.float64, copy=False)
        simplet_prior = fit_simplet_size_prior(state.n_cells, exclude_index=k)
        best: dict[str, Any] | None = None

        for a, b in pairs:
            mix = mixture_score_for_pair(
                target_indices,
                target_counts,
                log_q,
                a,
                b,
                lambda_grid,
                beta_lambda=beta_lambda,
                return_posterior=return_lambda_posterior,
            )
            abund = abundance_logbf(
                int(state.n_cells[k]),
                state.total_cells,
                state.n_cells,
                a,
                b,
                doublet_rate,
                phi_doublet,
                simplet_prior,
            )
            pair_prior = pair_prior_logprob(a, b, state.n_cells, state.total_cells, pairs, pair_prior_mode)
            expr_bf = mix.log_likelihood - l_self
            total = expr_bf + abundance_weight * abund + log_prior_odds_doublet_vs_simplet + pair_prior
            if best is None or total > best["total_log_odds"]:
                best = {
                    "a": a,
                    "b": b,
                    "mix": mix,
                    "abundance_logBF": float(abund),
                    "pair_prior_logprob": float(pair_prior),
                    "total_log_odds": float(total),
                }

        n_pairs = len(pairs)
        if best is None:
            best = {
                "a": -1,
                "b": -1,
                "mix": None,
                "abundance_logBF": np.nan,
                "pair_prior_logprob": np.nan,
                "total_log_odds": -math.inf,
            }
            l_mix = -math.inf
            lambda_map = np.nan
            lambda_mean = np.nan
        else:
            l_mix = float(best["mix"].log_likelihood)
            lambda_map = float(best["mix"].lambda_map)
            lambda_mean = float(best["mix"].lambda_mean)
            if return_lambda_posterior:
                lambda_posteriors[cluster_id] = best["mix"].lambda_log_posterior

        n_umi = float(state.N[k])
        expr_bf = l_mix - l_self
        parent_gain = l_mix - best_single_score
        search_penalty = math.log(max(1, n_pairs))
        row = {
            "cluster_id": cluster_id,
            "n_cells": int(state.n_cells[k]),
            "n_umi": n_umi,
            "best_parent_a": state.cluster_ids[int(best["a"])] if best["a"] >= 0 else None,
            "best_parent_b": state.cluster_ids[int(best["b"])] if best["b"] >= 0 else None,
            "lambda_map": lambda_map,
            "lambda_mean": lambda_mean,
            "L_self_LOO": float(l_self),
            "L_mix_best": float(l_mix),
            "best_single_parent": state.cluster_ids[best_single] if best_single >= 0 else None,
            "L_best_single_parent": float(best_single_score),
            "expr_logBF_doublet_vs_self": float(expr_bf),
            "expr_gain_vs_best_parent": float(parent_gain),
            "expr_logBF_per_UMI": float(expr_bf / max(1.0, n_umi)),
            "parent_gain_per_UMI": float(parent_gain / max(1.0, n_umi)),
            "abundance_logBF": best["abundance_logBF"],
            "total_log_odds": float(best["total_log_odds"]),
            "total_log_odds_minus_search_penalty": float(best["total_log_odds"] - search_penalty),
            "n_pairs_tested": int(n_pairs),
            "confidence_flag": _confidence_flag(int(state.n_cells[k]), n_umi, thresholds),
        }

        if return_cell_lambda and best["a"] >= 0:
            maps = cell_lambda_maps(state, k, int(best["a"]), int(best["b"]), log_q, lambda_grid)
            finite = maps[np.isfinite(maps)]
            if finite.size:
                q25, q75 = np.quantile(finite, [0.25, 0.75])
                row["lambda_cell_median"] = float(np.median(finite))
                row["lambda_cell_IQR"] = float(q75 - q25)
                edge = thresholds.lambda_edge_margin
                row["frac_cells_lambda_extreme"] = float(np.mean((finite <= edge) | (finite >= 1.0 - edge)))
            else:
                row["lambda_cell_median"] = np.nan
                row["lambda_cell_IQR"] = np.nan
                row["frac_cells_lambda_extreme"] = np.nan

        row["call"] = _call(row, thresholds)
        rows.append(row)

    df = pd.DataFrame(rows)
    return AuditResult(
        cluster_table=df,
        state=state,
        lambda_grid=lambda_grid,
        lambda_posteriors=lambda_posteriors if return_lambda_posterior else None,
    )
