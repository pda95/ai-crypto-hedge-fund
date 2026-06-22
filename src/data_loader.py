from pathlib import Path

import pandas as pd

# Корень проекта = на один уровень выше папки src/
# data_loader.py лежит в src/, поэтому .parent.parent -> корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_binance_data_csv(filename="btc_ccxt.csv"):
    """
    Загружает данные из CSV-файла в папке data/.
    Путь строится относительно корня проекта,
    поэтому работает независимо от того, откуда запущен код.
    """
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            f"Сначала скачайте данные: uv run python src/data_loader.py"
        )

    df = pd.read_csv(path)

    # Преобразуем timestamp в datetime, если есть такая колонка
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df

def load_portfolio_data(filename="portfolio_prices.csv"):
    """Читает таблицу цен портфеля из CSV (без сети)."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            f"Сначала скачайте: uv run python -c "
            f"'from src.data_loader import download_portfolio_data; download_portfolio_data()'"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_top_100_pairs(filename="top100_prices.csv"):
    """Читает таблицу цен 100 пар из CSV (без сети)."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            f"Сначала скачайте: uv run python -c "
            f"'from src.data_loader import download_top_100_pairs; download_top_100_pairs()'"
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)