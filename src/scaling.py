import numpy as np
import pandas as pd

# ============ 1. ОТБОР ПАР (уже продемонстрирован воронкой) ============
def selection_funnel():
    """Воронка отбора пар — для отчёта."""
    return pd.Series({
        "Requested (by liquidity)": 100,
        "Have 1y history": 80,
        "After removing stables/wrapped": 75,
        "After quality filters (tradable)": 63,
    }, name="pairs")

# ============ 2. КОРРЕЛЯЦИОННАЯ ГРУППИРОВКА (для риск-агента) ============
def correlation_groups(returns, threshold=0.7):
    """
    Группирует активы в кластеры по корреляции.
    Внутри кластера активы движутся вместе -> для риска считаются
    как одна "эффективная" позиция. Лечит ложную диверсификацию.
    """
    corr = returns.corr()
    assets = list(corr.columns)
    visited, groups = set(), []

    for a in assets:
        if a in visited:
            continue
        cluster = [a]
        visited.add(a)
        for b in assets:
            if b not in visited and corr.loc[a, b] > threshold:
                cluster.append(b)
                visited.add(b)
        groups.append(cluster)
    return groups

# ============ 3. ПРИОРИТИЗАЦИЯ СИГНАЛОВ ============
def prioritize_signals(signals, top_k=20):
    """
    Ранжирует сигналы по 63 монетам, оставляет топ-K для торговли.
    priority = сила * уверенность * ликвидность / волатильность.
    """
    df = signals.copy()
    df["priority"] = (
        df["signal_strength"].abs()
        * df["confidence"]
        * df["liquidity_score"]
        / (df["volatility"] + 1e-6)
    )
    return df.sort_values("priority", ascending=False).head(top_k)

# ============ 4. РИСК-АГЕНТ ============
class RiskAgent:
    """Отдельный риск-агент с правом вето над сигналами."""
    def __init__(self, target_vol=0.02, max_weight=0.10,
                 max_group_weight=0.30, max_drawdown_limit=0.25):
        self.target_vol = target_vol
        self.max_weight = max_weight
        self.max_group_weight = max_group_weight
        self.max_drawdown_limit = max_drawdown_limit

    def apply(self, weights, groups, asset_names, portfolio_vol, current_drawdown):
        w = pd.Series(weights, index=asset_names, dtype=float)

        # 1. Лимит на актив
        w = w.clip(upper=self.max_weight)

        # 2. Лимит на корреляционную группу (анти-ложная диверсификация)
        for g in groups:
            g_in = [a for a in g if a in w.index]
            if w[g_in].sum() > self.max_group_weight:
                w[g_in] *= self.max_group_weight / w[g_in].sum()

        # 3. Vol-targeting
        if portfolio_vol > 0:
            w *= min(1.0, self.target_vol / portfolio_vol)

        # 4. Drawdown stop
        risk_off = current_drawdown <= -self.max_drawdown_limit
        if risk_off:
            w *= 0.5

        if w.sum() > 1:
            w /= w.sum()
        return w, risk_off

# ============ 5. МОНИТОРИНГ (за пределами KPI) ============
def monitor_health(state):
    alerts = []
    if state["data_staleness_min"] > 15:
        alerts.append(f"STALE DATA: {state['data_staleness_min']} min")
    if state["missing_data_pct"] > 0.05:
        alerts.append(f"DATA GAPS: {state['missing_data_pct']:.0%}")
    if abs(state["feature_drift_score"]) > 3:
        alerts.append(f"MODEL DRIFT: z={state['feature_drift_score']:.1f}")
    if state["daily_turnover"] > 2.0:
        alerts.append(f"OVERTRADING: {state['daily_turnover']:.1f}x")
    return alerts

# ============ 6. FAIL-SAFE / KILL-SWITCH ============
def fail_safe_check(state, limits):
    reasons, halt, derisk = [], False, False
    if state["daily_pnl"] <= -limits["max_daily_loss"]:
        reasons.append("MAX DAILY LOSS -> HALT"); halt = True
    if state["drawdown"] <= -limits["max_drawdown"]:
        reasons.append("MAX DRAWDOWN -> DE-RISK"); derisk = True
    if state["turnover"] > limits["max_turnover"]:
        reasons.append("ABNORMAL TURNOVER -> HALT"); halt = True
    if state["max_position"] > limits["max_position"]:
        reasons.append("POSITION LIMIT -> HALT"); halt = True
    if not state["data_ok"]:
        reasons.append("DATA FEED FAILURE -> HALT"); halt = True

    if halt:
        return "HALT", reasons
    if derisk:
        return "DE_RISK", reasons
    return "NORMAL", reasons
