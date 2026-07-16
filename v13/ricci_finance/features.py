from __future__ import annotations
import numpy as np
import pandas as pd


def compute_market_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    market_return: pd.Series | None = None,
    window: int = 63,
) -> dict[str, pd.DataFrame]:
    returns = np.log(close / close.shift(1))
    volatility = returns.rolling(window, min_periods=max(5, window // 4)).std() * np.sqrt(252)
    momentum = close.pct_change(window, fill_method=None)
    drawdown = close.div(close.rolling(window, min_periods=2).max()).sub(1.0)
    dollar_volume = close * volume
    relative_volume = volume.div(volume.rolling(window, min_periods=5).median())
    out = {
        "returns": returns,
        "volatility": volatility,
        "momentum": momentum,
        "drawdown": drawdown,
        "dollar_volume": dollar_volume,
        "relative_volume": relative_volume,
    }
    if market_return is not None:
        beta = returns.rolling(window).cov(market_return).div(
            market_return.rolling(window).var(), axis=0
        )
        out["beta"] = beta
    return out
