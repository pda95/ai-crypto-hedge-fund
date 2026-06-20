def add_moving_averages(df, fast=50, slow=200):
    df = df.copy()

    df["ma50"] = df["close"].rolling(fast).mean()
    df["ma200"] = df["close"].rolling(slow).mean()

    return df


def add_returns(df):
    df = df.copy()

    df["market_return"] = df["close"].pct_change()

    return df
