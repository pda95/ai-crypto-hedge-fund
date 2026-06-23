import numpy as np

def run_ma_strategy(df, fee=0.001):
    """
    MA50/MA200 crossover strategy.


    Parameters
    ----------
    df : DataFrame с колонками 'close', 'ma50', 'ma200'.
    fee : комиссия за смену позиции (0.001 = 0.1%).
    """
    df = df.copy()

    # состояние "ma50 выше ma200" -> 1, иначе 0
    above = (df["ma50"] > df["ma200"]).astype(int)

    # пересечение = изменение состояния относительно прошлого бара
    cross = above.diff()           # +1 = Golden Cross, -1 = Death Cross, 0 = нет
    cross.iloc[0] = 0              # на первом баре пересечения нет по определению

    # строим позицию ТОЛЬКО из наблюдаемых пересечений; старт из КЭША
    signal = np.zeros(len(df))
    position = 0                   # <-- ключевое: стартуем в кэше, а не по above
    for i in range(len(df)):
        if cross.iloc[i] == 1:        # Golden Cross -> входим в long
            position = 1
        elif cross.iloc[i] == -1:     # Death Cross -> выходим в кэш
            position = 0
        signal[i] = position

    df["signal"] = signal
    df["position"] = df["signal"].shift(1)   # сдвиг на 1 бар -> без look-ahead

    df["market_return"] = df["close"].pct_change()
    df["trades"] = df["position"].diff().abs()
    df["strategy_return"] = df["position"] * df["market_return"] - df["trades"] * fee

    df = df.dropna()
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


def run_agent_strategy(df, fee=0.001):
    """Бэктест агента: position уже задан агентом."""
    df = df.copy()
    df["next_return"] = df["market_return"].shift(-1)
    df["trades"] = df["position"].diff().abs()
    df["strategy_return"] = df["position"] * df["next_return"] - df["trades"] * fee
    df = df.dropna()
    df["market_equity"] = (1 + df["next_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    return df



def run_ma_rsi_strategy(df, fee=0.001, rsi_low=45, rsi_high=55):
    """
    MA50/200 с RSI-фильтром флэта.
    Идея: в боковике RSI колеблется около 50 -> много ложных сигналов.
    Берём только сделки, где есть выраженный импульс (RSI вне нейтральной зоны).
    Требует колонку 'rsi' (из add_rsi) и 'ma50','ma200'.
    """
    df = df.copy()

    ma_signal = np.where(df["ma50"] > df["ma200"], 1, 0)

    # Фильтр флэта: торгуем, только если RSI вышел из нейтральной зоны 45..55
    not_flat = (df["rsi"] < rsi_low) | (df["rsi"] > rsi_high)

    df["signal"] = np.where(not_flat, ma_signal, 0)
    df["position"] = df["signal"].shift(1)        # сдвиг -> без look-ahead

    df["market_return"] = df["close"].pct_change()
    df["trades"] = df["position"].diff().abs()
    df["strategy_return"] = df["position"] * df["market_return"] - df["trades"] * fee

    df = df.dropna()
    df["market_equity"] = (1 + df["market_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    return df



