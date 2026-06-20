import ccxt
import pandas as pd


def load_binance_data_csv(
        filename="data/btc_ccxt.csv"
):
    """
    Загружает данные из CSV-файла.
    """
    df = pd.read_csv(filename)

    # Преобразуем timestamp в datetime, если это необходимо
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df
