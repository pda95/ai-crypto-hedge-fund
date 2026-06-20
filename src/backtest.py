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
    Превращает предсказания модели (колонка 'pred') в стратегию.
    1 = в позиции (лонг), 0 = в кэше.
    Логика комиссий и сдвига та же, что в run_ma_strategy.
    """
    df = test.copy()

    # pred уже относится к следующему бару (target = знак след. доходности),
    # поэтому позиция = pred, но сделку считаем по факту смены позиции
    df["position"] = df["pred"]
    df["trades"] = df["position"].diff().abs()

    df["strategy_return"] = (
            df["position"] * df["market_return"]
            - df["trades"] * fee
    )

    df = df.dropna()
    df["market_equity"] = (1 + df["market_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    return df
