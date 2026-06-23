# 🤖 AI Crypto Hedge Fund

> AI-agent based cryptocurrency hedge fund prototype.  
---

## 📁 Project Structure

```
ai-crypto-hedge-fund/
│
├── notebooks/
│   └── hedge_fund.ipynb        # Main notebook (all levels)
│
├── src/
│   ├── data_loader.py
│   ├── backtest.py
│   ├── features.py
│   ├── models.py
│   ├── portfolio.py
│   ├── rebalance.py
│   ├── agents.py
│   ├── metrics.py
│   ├── validation.py
│   ├── scaling.py
│   └── utils.py
│
├── pyproject.toml              # Dependencies (uv)
└── README.md
```

## 🚀 Installation

### uv 

```bash
# Install uv
pip install uv

# Install dependencies
uv sync
```

## ▶️ Run

```bash
# Jupyter Notebook
jupyter notebook notebooks/hedge_fund.ipynb

# Or JupyterLab
jupyter lab notebooks/hedge_fund.ipynb
```