import numpy as np


def run_ma_strategy(df, fee=0.001):
    """
    MA50/MA200 strategy backtest.

    Parameters
    ----------
    df : DataFrame with a 'close' column and 'ma50', 'ma200' columns.
    fee : commission per trade (0.001 = 0.1%, typical Binance taker fee).
          Charged every time the position changes.
    """
    df = df.copy()

    # 1 = long when fast MA above slow MA, else flat (0)
    df["signal"] = np.where(
        df["ma50"] > df["ma200"],
        1,
        0
    )

    # shift(1): trade on the NEXT bar -> no look-ahead bias
    df["position"] = df["signal"].shift(1)

    df["market_return"] = df["close"].pct_change()

    # 1 when position changes (enter/exit) -> pay commission
    df["trades"] = df["position"].diff().abs()

    # strategy return minus commission on each position change
    df["strategy_return"] = (
            df["position"]
            * df["market_return"]
            - df["trades"] * fee
    )

    df = df.dropna()

    # equity curves (start from 1.0)
    df["market_equity"] = (1 + df["market_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    return df


def run_ml_strategy(test, fee=0.001):
    """
    Превращает предсказания модели в стратегию.
    pred предсказывает ЗНАК доходности СЛЕДУЮЩЕГО бара (target = shift(-1)).
    Значит позицию, основанную на pred текущего бара,
    нужно применять к доходности СЛЕДУЮЩЕГО бара.
    """
    df = test.copy()

    # pred[t] -> прогноз для бара t+1, поэтому сдвигаем доходность назад:
    # позиция, открытая по сигналу на баре t, зарабатывает на баре t+1
    df["position"] = df["pred"]
    df["next_return"] = df["market_return"].shift(-1)

    df["trades"] = df["position"].diff().abs()

    df["strategy_return"] = (
        df["position"] * df["next_return"]
        - df["trades"] * fee
    )

    df = df.dropna()
    df["market_equity"] = (1 + df["next_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    return df
