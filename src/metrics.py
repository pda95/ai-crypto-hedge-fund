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

def win_rate_by_trades(df):
    """Calculate win rate per trade, not per day"""
    trades = df[df['position'].diff() != 0]  # моменты входа
    # или считать от сделки до сделки
    trade_returns = []
    for i in range(len(trades)-1):
        entry = trades.index[i]
        exit = trades.index[i+1]
        trade_return = df.loc[exit, 'strategy_equity'] / df.loc[entry, 'strategy_equity'] - 1
        trade_returns.append(trade_return)
    return sum(1 for r in trade_returns if r > 0) / len(trade_returns)

def add_performance_metrics(df):
    """Additional metrics for better analysis"""

    # Monthly returns
    df['month'] = df.index.to_period('M')
    monthly_returns = df.groupby('month')['strategy_return'].apply(
        lambda x: (1 + x).prod() - 1
    )

    # Maximum consecutive losses
    returns_binary = (df['strategy_return'] > 0).astype(int)
    consecutive_losses = (returns_binary == 0).groupby(
        (returns_binary != returns_binary.shift()).cumsum()
    ).sum().max()

    # Profit Factor (Total Profit / Total Loss)
    total_profit = df[df['strategy_return'] > 0]['strategy_return'].sum()
    total_loss = abs(df[df['strategy_return'] < 0]['strategy_return'].sum())
    profit_factor = total_profit / total_loss if total_loss != 0 else np.inf

    return {
        'Monthly Returns': monthly_returns,
        'Max Consecutive Losses': consecutive_losses,
        'Profit Factor': profit_factor
    }