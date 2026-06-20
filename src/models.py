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
