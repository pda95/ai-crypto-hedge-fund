import numpy as np
import pandas as pd
class TradingAgent:
    """
    Простой rule-based AI-агент, объединяющий несколько источников сигнала
    в одно торговое решение и масштабирующий позицию по риску.

    Логика:
      - направление берём из ML-предсказания (pred) И трендового фильтра (MA);
      - размер позиции снижаем, когда прогноз волатильности высок
        (vol-targeting — управление риском).
    """

    def __init__(self, target_vol=0.02, max_leverage=1.0):
        # целевая дневная волатильность портфеля (2%)
        self.target_vol = target_vol
        self.max_leverage = max_leverage

    def decide(self, df):
        df = df.copy()

        # 1) Направление: лонг только если И ML, И тренд (MA) согласны
        ml_long = (df["pred"] == 1).astype(int)
        trend_long = (df["ma50"] > df["ma200"]).astype(int)
        direction = ml_long & trend_long  # консенсус сигналов

        # 2) Риск-скейлинг по прогнозной волатильности (vol targeting)
        vol = df["volatility"].replace(0, np.nan)
        size = (self.target_vol / vol).clip(upper=self.max_leverage)
        size = size.fillna(0)

        # 3) Итоговая позиция = направление * размер с учётом риска
        df["position"] = direction * size
        return df