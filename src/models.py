import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_logreg():
    """
    Логистическая регрессия — простой и интерпретируемый БЕНЧМАРК.
    StandardScaler обязателен: признаки в разных масштабах.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def make_gradient_boosting():
    """
    Градиентный бустинг — более мощная нелинейная модель.
    Деревья не требуют масштабирования.
    """
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )


def train_and_predict(model, train, test, feature_cols, target_col="target"):
    """
    Обучает модель на train, предсказывает на test.
    Возвращает test с колонками 'pred' (0/1) и 'pred_proba'.
    """
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]

    model.fit(X_train, y_train)

    test = test.copy()
    test["pred"] = model.predict(X_test)
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]

    return test, model


import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arima_signal(train_returns, test_returns, order=(1, 0, 1)):
    """
    ARIMA для прогноза доходности -> торговый сигнал.
    Walk-forward: на каждом шаге переобучаемся на всей доступной истории
    и прогнозируем доходность следующего дня.

    order=(p,d,q): (1,0,1) — стартовая разумная конфигурация для доходностей
    (доходности уже стационарны, поэтому d=0).

    Возвращает массив сигналов: 1 если прогноз доходности > 0, иначе 0.
    """
    history = list(train_returns.dropna().values)
    signals = []

    for r in test_returns.values:
        model = ARIMA(history, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=1)[0]

        signals.append(1 if forecast > 0 else 0)
        history.append(r)  # добавляем реальное значение и идём дальше

    return np.array(signals)


from arch import arch_model

def fit_garch_volatility(returns, p=1, q=1):
    """
    GARCH(1,1) для прогноза волатильности дневной доходности.
    Волатильность кластеризуется (спокойные и бурные периоды),
    поэтому, в отличие от направления, она ПРЕДСКАЗУЕМА.

    returns — в процентах (умножаем на 100 для устойчивости оптимизатора).
    """
    r = returns.dropna() * 100

    model = arch_model(r, vol="GARCH", p=p, q=q, mean="constant")
    fitted = model.fit(disp="off")

    # условная волатильность (in-sample), обратно в доли
    cond_vol = fitted.conditional_volatility / 100
    return fitted, cond_vol

def garch_forecast(fitted, horizon=1):
    """Прогноз волатильности на horizon дней вперёд (в долях)."""
    fc = fitted.forecast(horizon=horizon, reindex=False)
    variance = fc.variance.values[-1]
    return np.sqrt(variance) / 100