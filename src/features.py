def add_moving_averages(df):
    df = df.copy()

    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    return df