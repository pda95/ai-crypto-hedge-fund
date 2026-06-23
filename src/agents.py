class TradingAgent:
    """
    Trading Agent with position sizing and confidence level
    """

    def __init__(self, max_leverage=1.0, base_confidence=0.5):
        self.max_leverage = max_leverage
        self.base_confidence = base_confidence

    def decide(self, df):
        df = df.copy()

        confidence = df["pred_proba"]
        position_size = (confidence - 0.5) * 4
        position_size = position_size.clip(0, self.max_leverage)

        df["position"] = df["pred"] * position_size

        return df


