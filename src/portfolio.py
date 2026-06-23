import numpy as np
import pandas as pd
from scipy.optimize import minimize

PERIODS_PER_YEAR = 365


def compute_returns(prices):
    """Дневные доходности из цен закрытия."""
    return prices.pct_change().dropna()


def portfolio_performance(weights, mean_returns, cov_matrix,
                          periods_per_year=PERIODS_PER_YEAR):
    """Годовые доходность и волатильность портфеля по весам."""
    ret = np.sum(mean_returns * weights) * periods_per_year
    vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(periods_per_year)
    return ret, vol


def optimize_max_sharpe(mean_returns, cov_matrix,
                        risk_free=0.0, max_weight=0.30):
    """
    Max Sharpe с ограничением на максимальный вес одного актива.
    max_weight=0.30 -> не более 30% в одну монету.
    """
    n = len(mean_returns)
    init_w = np.array([1 / n] * n)
    bounds = tuple((0, max_weight) for _ in range(n))  # <-- ограничение
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_sharpe(w):
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(w @ cov_matrix.values @ w)
        return -(ret - risk_free) / vol

    result = minimize(neg_sharpe, init_w,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints)
    return result.x


def optimize_min_variance(mean_returns, cov_matrix, max_weight=0.30):
    n = len(mean_returns)
    init_w = np.array([1 / n] * n)
    bounds = tuple((0, max_weight) for _ in range(n))  # <-- ограничение
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def port_var(w):
        return w @ cov_matrix.values @ w

    result = minimize(port_var, init_w,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints)
    return result.x


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


def backtest_with_risk_off(returns, rebal_freq=30,
                           strategy="max_sharpe",
                           drawdown_threshold=-0.15,
                           vol_threshold=0.04,
                           max_weight=0.30):
    """
    Ребалансировка с Risk-Off режимом:
    - Если drawdown портфеля > threshold -> уходим в кэш
    - Если rolling vol > vol_threshold -> уходим в кэш
    - Возвращаемся когда условия нормализовались

    Это прямой ответ на вопрос:
    'How can AI help mitigate losses during high-volatility periods?'
    """
    n_assets = len(returns.columns)
    equity = [1.0]
    daily_returns = []
    in_risk_off = False

    for i in range(1, len(returns)):
        # --- Текущий drawdown ---
        peak = max(equity)
        current_dd = (equity[-1] - peak) / peak

        # --- Текущая волатильность (rolling 20d) ---
        if i >= 20:
            current_vol = returns.iloc[i - 20:i].mean(axis=1).std()
        else:
            current_vol = 0

        # --- Risk-Off условие ---
        if current_dd < drawdown_threshold or current_vol > vol_threshold:
            in_risk_off = True
        elif current_dd > drawdown_threshold * 0.5:
            # восстанавливаемся когда drawdown уменьшился вдвое
            in_risk_off = False

        if in_risk_off:
            # в кэше — доходность 0
            daily_r = 0.0
        else:
            # ребалансировка по расписанию
            if i % rebal_freq == 0 and i >= 60:
                train = returns.iloc[max(0, i - 180):i]
                if strategy == "max_sharpe":
                    w = optimize_max_sharpe(train.mean(), train.cov(),
                                            max_weight=max_weight)
                else:
                    w = np.array([1 / n_assets] * n_assets)
            else:
                w = np.array([1 / n_assets] * n_assets)

            daily_r = (returns.iloc[i] * w).sum()

        daily_returns.append(daily_r)
        equity.append(equity[-1] * (1 + daily_r))

    equity_series = pd.Series(equity[1:], index=returns.index[1:])
    returns_series = pd.Series(daily_returns, index=returns.index[1:])
    return equity_series, returns_series


def clean_universe(prices, max_zero_pct=10, min_vol=0.005, max_vol=0.50):
    """
    Финальный отбор торгуемого универса (реализация select_pairs на данных).
    Убирает: мёртвые (низкая vol), forward-fill артефакты (много нулей),
    скам-кандидаты (экстремальная vol).
    """
    returns = prices.pct_change().dropna()

    daily_vol = returns.std()
    zero_pct = (returns == 0).sum() / len(returns) * 100
    max_move = returns.abs().max()

    keep = (
            (daily_vol >= min_vol) &
            (daily_vol <= max_vol) &
            (zero_pct <= max_zero_pct) &
            (max_move <= 0.60)
    )
    selected = keep[keep].index.tolist()
    return prices[selected]
