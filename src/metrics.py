import numpy as np


def roi(equity):
    """Total return in %, from an equity curve."""
    return (equity.iloc[-1] / equity.iloc[0] - 1) * 100


def sharpe_ratio(returns, periods_per_year=365):
    """
    Annualized Sharpe ratio.
    Crypto trades 24/7, so daily bars -> 365 periods per year.
    For hourly bars use periods_per_year=24*365.
    """
    if returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()


def sortino_ratio(returns, periods_per_year=365):
    """
    Like Sharpe, but penalizes only downside volatility
    (std of negative returns). Better for asymmetric strategies.
    """
    downside = returns[returns < 0]
    if downside.std() == 0 or len(downside) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / downside.std()


def max_drawdown(equity):
    """
    Maximum drawdown in % (negative number).
    Largest peak-to-trough drop of the equity curve.
    """
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min() * 100


def calmar_ratio(equity, returns, periods_per_year=365):
    """
    Annualized return divided by max drawdown.
    Shows return per unit of worst-case loss.
    """
    n_years = len(returns) / periods_per_year
    if n_years == 0:
        return 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    mdd = abs(max_drawdown(equity) / 100)
    if mdd == 0:
        return 0.0
    return (cagr * 100) / (mdd * 100)


def win_rate(returns):
    """Share of bars with positive return, in %."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean() * 100


def summary(equity, returns, periods_per_year=365):
    """All metrics at once -> dict. Convenient for a comparison table."""
    return {
        "ROI %": round(roi(equity), 2),
        "Sharpe": round(sharpe_ratio(returns, periods_per_year), 2),
        "Sortino": round(sortino_ratio(returns, periods_per_year), 2),
        "Calmar": round(calmar_ratio(equity, returns, periods_per_year), 2),
        "Max Drawdown %": round(max_drawdown(equity), 2),
        "Win Rate %": round(win_rate(returns), 2),
    }
