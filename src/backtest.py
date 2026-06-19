import numpy as np


def run_ma_strategy(df):
    df = df.copy()

    df["signal"] = np.where(
        df["ma50"] > df["ma200"],
        1,
        0
    )

    df["position"] = df["signal"].shift(1)

    df["market_return"] = df["close"].pct_change()

    df["strategy_return"] = (
            df["position"]
            * df["market_return"]
    )

    return df.dropna()
