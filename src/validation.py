from sklearn.model_selection import TimeSeriesSplit


def chronological_split(df, train_ratio=0.7):
    """
    Простой хронологический train/test split.
    Первые train_ratio% времени -> train, остальное -> test.
    НИКАКОГО перемешивания: для временных рядов это обязательно,
    иначе модель «увидит будущее».
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def walk_forward_splits(df, n_splits=5):
    """
    Walk-forward валидация (расширяющееся окно).
    Каждый следующий фолд обучается на всём прошлом
    и тестируется на следующем отрезке времени.

    Возвращает индексы (train_idx, test_idx) для каждого фолда.
    Имитирует реальную торговлю: учимся на истории -> торгуем дальше.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(df))


import numpy as np

def random_strategy_baseline(test, n_trials=1000, fee=0.001, seed=42):
    """
    Permutation-тест: сравниваем стратегию со случайными сигналами.
    Генерируем n_trials случайных стратегий (случайные позиции 0/1)
    и смотрим распределение их ROI.

    Если ROI нашей модели выше, чем у, скажем, 95% случайных —
    результат статистически значим, а не везение.
    """
    rng = np.random.default_rng(seed)
    market_return = test["market_return"].values
    rois = []

    for _ in range(n_trials):
        positions = rng.integers(0, 2, size=len(market_return))
        trades = np.abs(np.diff(positions, prepend=0))
        strat_ret = positions * market_return - trades * fee
        equity = np.cumprod(1 + strat_ret)
        rois.append((equity[-1] - 1) * 100)

    return np.array(rois)

def significance(strategy_roi, random_rois):
    """
    Доля случайных стратегий, которые ХУЖЕ нашей.
    Возвращает «перцентиль» и псевдо-p-value.
    """
    percentile = (random_rois < strategy_roi).mean() * 100
    p_value = (random_rois >= strategy_roi).mean()
    return percentile, p_value
