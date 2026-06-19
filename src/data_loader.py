import ccxt
import pandas as pd

def load_binance_data(
        symbol="BTC/USDT",
        timeframe="1d",
        limit=1000
):
    exchange = ccxt.binance()

    ohlcv = exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit
    )

    df = pd.DataFrame(
        ohlcv,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    return df

if __name__ == "__main__":
    df = load_binance_data()

    print(df.head())
    print(df.shape)