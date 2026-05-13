from __future__ import annotations

import pandas as pd


def plot_score_distribution(cluster_table: pd.DataFrame, ax=None):
    """Plot total log-odds by cluster; returns the matplotlib axis."""

    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.hist(cluster_table["total_log_odds_minus_search_penalty"].dropna(), bins=30)
    ax.set_xlabel("total log-odds minus search penalty")
    ax.set_ylabel("clusters")
    return ax


def plot_lambda_posterior(lambda_grid, posterior, ax=None):
    """Plot a saved lambda posterior for one cluster."""

    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.plot(lambda_grid, posterior)
    ax.set_xlabel("lambda")
    ax.set_ylabel("posterior mass")
    return ax
