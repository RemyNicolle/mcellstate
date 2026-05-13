import numpy as np
from scipy import sparse
from scipy.special import gammaln

from mcellstate_doublet import audit_doublets, build_cluster_state, make_prior
from mcellstate_doublet.abundance import fit_simplet_size_prior, nb_logpmf
from mcellstate_doublet.likelihood import (
    dm_posterior_predictive_logpmf_sparse,
    parent_log_probs,
    parent_scores_for_target,
    self_loo_score,
)


def test_dm_posterior_predictive_formula_matches_direct_difference():
    c = np.array([4.0, 1.0, 0.0])
    x = np.array([2.0, 0.0, 3.0])
    psi = np.array([0.5, 0.25, 0.25])
    idx = np.flatnonzero(x)
    got = dm_posterior_predictive_logpmf_sparse(
        idx, x[idx], x.sum(), c[idx], c.sum(), psi[idx], psi.sum()
    )
    direct = gammaln(c.sum() + psi.sum()) - gammaln(c.sum() + x.sum() + psi.sum())
    direct += np.sum(gammaln(c + x + psi) - gammaln(c + psi))
    assert np.isclose(got, direct)


def test_self_loo_hand_built_cluster():
    X = sparse.csr_matrix([[2, 0, 1], [1, 1, 0], [0, 3, 0]], dtype=float)
    z = np.array([0, 0, 1])
    state = build_cluster_state(X, z)
    psi = np.ones(3) / 3
    got = self_loo_score(state, 0, psi)
    row0 = X.getrow(0)
    row1 = X.getrow(1)
    expected = dm_posterior_predictive_logpmf_sparse(
        row0.indices, row0.data, row0.data.sum(), row1.toarray().ravel()[row0.indices], row1.data.sum(), psi[row0.indices], psi.sum()
    )
    expected += dm_posterior_predictive_logpmf_sparse(
        row1.indices, row1.data, row1.data.sum(), row0.toarray().ravel()[row1.indices], row0.data.sum(), psi[row1.indices], psi.sum()
    )
    assert np.isclose(got, expected)


def _synthetic_matrix(seed=0):
    rng = np.random.default_rng(seed)
    profiles = np.array(
        [
            [0.82, 0.12, 0.03, 0.03],
            [0.03, 0.03, 0.12, 0.82],
            [0.42, 0.08, 0.08, 0.42],
            [0.05, 0.80, 0.10, 0.05],
        ]
    )
    labels = np.repeat([0, 1, 2, 3], [30, 30, 12, 30])
    rows = [rng.multinomial(120, profiles[k]) for k in labels]
    return sparse.csr_matrix(np.asarray(rows)), labels


def test_mixture_likelihood_recovers_synthetic_parent_pair():
    X, z = _synthetic_matrix()
    psi = make_prior(X, tau=1.0)
    res = audit_doublets(
        X,
        z,
        psi=psi,
        top_m_parents=3,
        lambda_grid_size=51,
        beta_lambda=(1, 1),
        pair_prior_mode="none",
    )
    row = res.cluster_table.set_index("cluster_id").loc[2]
    assert {row.best_parent_a, row.best_parent_b} == {0, 1}
    assert 0.25 <= row.lambda_map <= 0.75
    assert row.expr_gain_vs_best_parent > 0


def test_obvious_simplet_is_not_overcalled():
    X, z = _synthetic_matrix(2)
    psi = make_prior(X, tau=1.0)
    res = audit_doublets(X, z, psi=psi, top_m_parents=3, lambda_grid_size=31)
    table = res.cluster_table.set_index("cluster_id")
    assert table.loc[0, "call"] == "likely_simplet"
    assert table.loc[1, "call"] == "likely_simplet"


def test_rare_distinct_cluster_not_called_doublet_just_for_size():
    rng = np.random.default_rng(3)
    profiles = np.array([[0.9, 0.1, 0.0], [0.0, 0.1, 0.9], [0.0, 0.95, 0.05]])
    labels = np.repeat([0, 1, 2], [25, 25, 3])
    X = sparse.csr_matrix(np.asarray([rng.multinomial(80, profiles[k]) for k in labels]))
    psi = make_prior(X, tau=1.0)
    res = audit_doublets(X, labels, psi=psi, top_m_parents=2, lambda_grid_size=31)
    assert res.cluster_table.set_index("cluster_id").loc[2, "call"] != "likely_doublet"


def test_sparse_and_dense_paths_agree():
    X_sparse, z = _synthetic_matrix(4)
    X_dense = X_sparse.toarray()
    psi = make_prior(X_sparse, tau=1.0)
    a = audit_doublets(X_sparse, z, psi=psi, top_m_parents=3, lambda_grid_size=21)
    b = audit_doublets(X_dense, z, psi=psi, top_m_parents=3, lambda_grid_size=21)
    cols = ["L_self_LOO", "L_mix_best", "expr_gain_vs_best_parent", "total_log_odds"]
    assert np.allclose(a.cluster_table[cols], b.cluster_table[cols])


def test_output_columns_and_parent_scores_are_deterministic():
    X, z = _synthetic_matrix(5)
    psi = make_prior(X, tau=1.0)
    res1 = audit_doublets(X, z, psi=psi, top_m_parents=3, lambda_grid_size=21)
    res2 = audit_doublets(X, z, psi=psi, top_m_parents=3, lambda_grid_size=21)
    expected = {
        "cluster_id",
        "n_cells",
        "n_umi",
        "best_parent_a",
        "best_parent_b",
        "lambda_map",
        "lambda_mean",
        "L_self_LOO",
        "L_mix_best",
        "best_single_parent",
        "L_best_single_parent",
        "expr_logBF_doublet_vs_self",
        "expr_gain_vs_best_parent",
        "expr_logBF_per_UMI",
        "parent_gain_per_UMI",
        "abundance_logBF",
        "total_log_odds",
        "total_log_odds_minus_search_penalty",
        "n_pairs_tested",
        "confidence_flag",
        "call",
    }
    assert expected.issubset(set(res1.cluster_table.columns))
    assert res1.cluster_table.equals(res2.cluster_table)

    state = build_cluster_state(X, z)
    model = parent_log_probs(state, psi)
    scores = parent_scores_for_target(state, 0, model)
    assert scores.shape == (state.n_clusters,)


def test_abundance_nb_logpmf_and_prior_are_finite():
    assert np.isfinite(nb_logpmf(4, mean=3.0, phi=2.0))
    prior = fit_simplet_size_prior(np.array([1, 5, 8, 12]), exclude_index=0)
    assert np.isfinite(prior.logpmf(4))
