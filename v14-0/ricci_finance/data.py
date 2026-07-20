from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


DEFAULT_TICKERS = [
    "NVDA", "AMD", "AVGO", "TSM", "MU", "MRVL", "AMAT", "LRCX", "KLAC",
    "ANET", "AAOI", "COHR", "LITE", "SMCI", "PLTR", "IONQ", "QBTS", "QUBT",
    "RGTI", "NBIS",
]


def parse_tickers(text_or_list: str | Sequence[str]) -> list[str]:
    if isinstance(text_or_list, str):
        raw = text_or_list.replace("\n", ",").split(",")
    else:
        raw = list(text_or_list)
    return list(dict.fromkeys(str(x).strip().upper() for x in raw if str(x).strip()))


def _extract_field(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if field in level0:
            out = data[field].copy()
        else:
            raise KeyError(f"Field {field!r} not returned by yfinance.")
    else:
        col = field if field in data.columns else data.columns[0]
        out = data[[col]].copy()
        out.columns = [tickers[0]]
    return out.dropna(axis=1, how="all").sort_index()


def download_market_data(
    tickers: Sequence[str],
    period: str = "1y",
    interval: str = "1d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if yf is None:
        raise ImportError("Install yfinance: pip install yfinance")
    tickers = parse_tickers(tickers)
    if len(tickers) < 1:
        raise ValueError("At least one ticker is required.")
    try:
        yf.set_tz_cache_location(".")
    except Exception:
        pass
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    close = _extract_field(raw, "Close", tickers)
    volume = _extract_field(raw, "Volume", tickers)
    common = close.columns.intersection(volume.columns)
    close = close[common]
    volume = volume[common]
    dollar_volume = close * volume
    return close, volume, dollar_volume


def make_demo_market_data(
    tickers: Sequence[str] = DEFAULT_TICKERS[:8],
    n_days: int = 260,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = parse_tickers(tickers)
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, [0.010, 0.014, 0.020], size=(n_days, 3))
    loadings = rng.uniform(-0.3, 1.2, size=(len(tickers), 3))
    noise = rng.normal(0, 0.012, size=(n_days, len(tickers)))
    returns = factors @ loadings.T + noise
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    volumes = rng.lognormal(mean=15.5, sigma=0.7, size=(n_days, len(tickers)))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    close = pd.DataFrame(prices, index=idx, columns=tickers)
    volume = pd.DataFrame(volumes, index=idx, columns=tickers)
    return close, volume, close * volume


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    clean = prices.replace([np.inf, -np.inf], np.nan)
    returns = np.log(clean / clean.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def validate_market_data(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    minimum_rows: int = 60,
) -> list[str]:
    warnings: list[str] = []
    if prices.empty:
        return ["Price table is empty."]
    if len(prices) < minimum_rows:
        warnings.append(f"Only {len(prices)} rows; at least {minimum_rows} are recommended.")
    if prices.index.has_duplicates:
        warnings.append("Duplicate timestamps detected.")
    bad = prices.isna().mean()
    bad = bad[bad > 0.30]
    if not bad.empty:
        warnings.append("High missing-data ratio: " + ", ".join(bad.index.astype(str)))
    nonpositive = [c for c in prices if (prices[c].dropna() <= 0).any()]
    if nonpositive:
        warnings.append("Non-positive prices: " + ", ".join(nonpositive))
    if volumes is not None and not volumes.empty:
        zero_cols = [c for c in volumes if volumes[c].fillna(0).eq(0).all()]
        if zero_cols:
            warnings.append("All-zero volume: " + ", ".join(zero_cols))
    return warnings
