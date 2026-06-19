import numpy as np


def roi(equity_curve):
    return (equity_curve.iloc[-1] - 1) * 100


def sharpe_ratio(strategy_returns):
    return (
            np.sqrt(365)
            * strategy_returns.mean()
            / strategy_returns.std()
    )


def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()

    drawdown = (
                       equity_curve - running_max
               ) / running_max

    return drawdown.min() * 100
