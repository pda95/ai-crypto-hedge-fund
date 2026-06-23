import pandas as pd


def add_moving_averages(df, fast=50, slow=200):
    df = df.copy()

    df["ma50"] = df["close"].rolling(fast).mean()
    df["ma200"] = df["close"].rolling(slow).mean()

    return df


def add_returns(df):
    df = df.copy()

    df["market_return"] = df["close"].pct_change()

    return df


def add_rsi(df, period=14):
    """
    RSI — индекс относительной силы (0..100).
    >70 обычно перекупленность, <30 перепроданность.
    Считается только по прошлым ценам (rolling).
    """
    df = df.copy()
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def add_bollinger_width(df, period=20, n_std=2):
    """
    Ширина полос Боллинджера — мера волатильности.
    Широкие полосы -> высокая волатильность, узкие -> сжатие/штиль.
    """
    df = df.copy()

    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()

    upper = ma + n_std * std
    lower = ma - n_std * std

    df["bb_width"] = (upper - lower) / ma  # нормируем на цену

    return df


def add_volatility(df, period=14):
    """
    Скользящая волатильность дневной доходности.
    Нужна и как признак, и пригодится для риск-агента позже.
    """
    df = df.copy()
    if "market_return" not in df.columns:
        df["market_return"] = df["close"].pct_change()

    df["volatility"] = df["market_return"].rolling(period).std()

    return df


def add_return_lags(df, lags=(1, 2, 3, 5)):
    """
    Лаги доходности: доходность 1, 2, 3, 5 дней назад.
    Дают модели «память» о недавней динамике.
    shift(n) -> строго прошлые значения, без утечки.
    """
    df = df.copy()
    if "market_return" not in df.columns:
        df["market_return"] = df["close"].pct_change()

    for lag in lags:
        df[f"return_lag_{lag}"] = df["market_return"].shift(lag)

    return df


def add_target(df, horizon=1):
    """
    Целевая переменная для классификации:
    вырастет ли цена на следующем баре.
    1 -> доходность следующего дня > 0, иначе 0.

    shift(-horizon) смотрит в будущее — это НОРМАЛЬНО для target,
    но такие колонки НИКОГДА не должны попадать в признаки.
    """
    df = df.copy()
    if "market_return" not in df.columns:
        df["market_return"] = df["close"].pct_change()

    future_return = df["market_return"].shift(-horizon)
    df["target"] = (future_return > 0).astype(int)

    return df


def add_temporal_features(df):
    """
    Временные признаки на основе колонки timestamp.
    """
    df = df.copy()

    # Убедимся, что timestamp в правильном формате
    if 'timestamp' in df.columns:
        # Если timestamp в миллисекундах (Unix)
        if df['timestamp'].dtype == 'int64':
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])

        # Извлекаем временные признаки
        df["day_of_week"] = df['timestamp_dt'].dt.dayofweek
        df["month"] = df['timestamp_dt'].dt.month

        # Удаляем временную колонку
        df = df.drop('timestamp_dt', axis=1)

    return df


def add_price_features(df):
    """Ценовые признаки."""
    df = df.copy()
    # High-Low диапазон
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    # Price position in range
    df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-6)
    return df


def add_volume_features(df):
    """Признаки объема."""
    df = df.copy()
    df["volume_change"] = df["volume"].pct_change()
    # Volume-price correlation
    df["volume_price_corr"] = df["volume"].rolling(20).corr(df["close"])
    return df


def build_features(df):
    """
    Собирает все признаки + целевую переменную в один вызов.
    Возвращает готовый к ML датафрейм (с очищенными NaN).
    """
    df = df.copy()
    df["market_return"] = df["close"].pct_change()

    df = add_moving_averages(df, fast=50, slow=200)
    df = add_rsi(df)
    df = add_bollinger_width(df)
    df = add_volatility(df)
    df = add_return_lags(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_temporal_features(df)

    df = add_target(df, horizon=1)

    return df.dropna().reset_index(drop=True)
