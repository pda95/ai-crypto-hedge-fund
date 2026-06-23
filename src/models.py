import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


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

def make_random_forest():
    """
    Random Forest — ансамбль деревьев, устойчив к переобучению.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

def make_xgboost():
    """
    XGBoost — градиентный бустинг с регуляризацией.
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )

def make_logreg_gridsearch():
    """
    Logistic Regression with GridSearch.
    """
    return GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        param_grid={
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__class_weight': ['balanced', None]
        },
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

def make_gradient_boosting_gridsearch():
    """
    Gradient Boosting with GridSearch.
    """
    return GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        },
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

def make_random_forest_gridsearch():
    """
    Random Forest with GridSearch.
    """
    return GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

def make_xgboost_gridsearch():
    """
    XGBoost with GridSearch.
    """
    return GridSearchCV(
        XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )


def train_and_predict(model, train, test, feature_cols, target_col="target"):
    """
    Обучает модель на train, предсказывает на test.
    Поддерживает как обычные модели, так и GridSearchCV.
    """
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]

    model.fit(X_train, y_train)

    # Если это GridSearchCV, берем лучшую модель
    if hasattr(model, 'best_estimator_'):
        best_model = model.best_estimator_
        best_score = model.best_score_
        best_params = model.best_params_
    else:
        best_model = model
        best_score = None
        best_params = None

    test = test.copy()
    test["pred"] = best_model.predict(X_test)
    test["pred_proba"] = best_model.predict_proba(X_test)[:, 1]

    return test, model, best_score, best_params



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



def evaluate_classification_model(test_df, model_name="Model"):
    """
    Оценка качества классификационной модели на тестовой выборке.
    """
    y_true = test_df['target']
    y_pred = test_df['pred']
    y_proba = test_df.get('pred_proba', None)

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_proba)

    return metrics