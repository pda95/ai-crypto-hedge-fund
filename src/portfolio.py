import numpy as np
import pandas as pd
from scipy.optimize import minimize

PERIODS_PER_YEAR = 365  # крипта торгуется 24/7


def compute_returns(prices):
    """Дневные доходности из цен закрытия. dropna() убирает первый NaN."""
    return prices.pct_change().dropna()


def portfolio_performance(weights, mean_returns, cov_matrix,
                          periods_per_year=PERIODS_PER_YEAR):
    """Годовые доходность и волатильность портфеля по весам."""
    ret = np.sum(mean_returns * weights) * periods_per_year
    vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(periods_per_year)
    return ret, vol


def _neg_sharpe(weights, mean_returns, cov_matrix, rf=0.0):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - rf) / vol


def optimize_max_sharpe(mean_returns, cov_matrix):
    """Касательный портфель — максимальный Sharpe (long-only, веса 0..1)."""
    n = len(mean_returns)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    res = minimize(_neg_sharpe, n * [1.0 / n],
                   args=(mean_returns, cov_matrix),
                   method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def optimize_min_variance(mean_returns, cov_matrix):
    """Портфель минимальной волатильности (самый консервативный)."""
    n = len(mean_returns)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    res = minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), n * [1.0 / n],
                   method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def efficient_frontier(mean_returns, cov_matrix, n_points=50):
    """Эффективная граница: мин. волатильность для набора целевых доходностей."""
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    targets = np.linspace(mean_returns.min() * PERIODS_PER_YEAR,
                          mean_returns.max() * PERIODS_PER_YEAR, n_points)
    frontier = []
    for t in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=t: np.sum(mean_returns * w) * PERIODS_PER_YEAR - t},
        )
        res = minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), n * [1.0 / n],
                       method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            ret, vol = portfolio_performance(res.x, mean_returns, cov_matrix)
            frontier.append((vol, ret))
    return np.array(frontier)


def weights_table(weights, asset_names):
    """Веса в проценты -> Series для красивого вывода."""
    return pd.Series(np.round(weights * 100, 2), index=asset_names, name="weight %")
