import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio import PERIODS_PER_YEAR

def optimize_max_sharpe_capped(mean_returns, cov_matrix, max_weight=0.40):
    """
    Max Sharpe с ОГРАНИЧЕНИЕМ макс. веса на актив.
    Лечит проблему Уровня 3 (100% в одну монету): max_weight=0.40
    заставляет диверсифицировать минимум по 3 активам.
    """
    n = len(mean_returns)

    def neg_sharpe(w):
        ret = np.sum(mean_returns * w) * PERIODS_PER_YEAR
        vol = np.sqrt(w.T @ cov_matrix @ w) * np.sqrt(PERIODS_PER_YEAR)
        return -ret / vol if vol > 0 else 0.0

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, max_weight) for _ in range(n))
    res = minimize(neg_sharpe, n * [1.0 / n], method="SLSQP",
                   bounds=bounds, constraints=constraints)
    return res.x if res.success else np.array(n * [1.0 / n])

def backtest_rebalanced(returns, window=90, rebal_freq=30,
                        max_weight=0.40, fee=0.001, strategy="max_sharpe"):
    """
    Бэктест динамической ребалансировки на скользящем окне.

    window      : сколько прошлых дней используем для оптимизации
    rebal_freq  : как часто пересчитываем веса (в днях)
    strategy    : 'max_sharpe' | 'equal_weight'

    Look-ahead отсутствует: веса на день t считаются по [t-window : t]
    (строго прошлое) и применяются к доходности дня t.
    Издержки начисляются на оборот (turnover) при каждой ребалансировке.
    """
    n = returns.shape[1]
    weights = np.array([1.0 / n] * n)
    last_weights = weights.copy()

    equity = [1.0]
    strat_returns = []
    weight_history = []

    for t in range(window, len(returns)):
        # ребалансировка по расписанию
        if (t - window) % rebal_freq == 0:
            past = returns.iloc[t - window:t]
            if strategy == "max_sharpe":
                new_w = optimize_max_sharpe_capped(past.mean(), past.cov(), max_weight)
            else:  # equal_weight
                new_w = np.array([1.0 / n] * n)

            turnover = np.abs(new_w - last_weights).sum()
            cost = turnover * fee
            weights = new_w
            last_weights = new_w
        else:
            cost = 0.0

        day_ret = (returns.iloc[t] * weights).sum() - cost
        strat_returns.append(day_ret)
        equity.append(equity[-1] * (1 + day_ret))
        weight_history.append(weights.copy())

    idx = returns.index[window:]
    equity_series = pd.Series(equity[1:], index=idx)
    returns_series = pd.Series(strat_returns, index=idx)
    weights_df = pd.DataFrame(weight_history, index=idx, columns=returns.columns)
    return equity_series, returns_series, weights_df

def buy_and_hold_weights(returns, max_weight=0.40, window=90):
    """
    Бенчмарк: оптимизировать веса ОДИН раз (на первом окне) и держать.
    Это статический портфель из Уровня 3 — для сравнения с динамикой.
    """
    n = returns.shape[1]
    past = returns.iloc[:window]
    w = optimize_max_sharpe_capped(past.mean(), past.cov(), max_weight)

    sub = returns.iloc[window:]
    daily = (sub * w).sum(axis=1)
    equity = (1 + daily).cumprod()
    return equity, daily
