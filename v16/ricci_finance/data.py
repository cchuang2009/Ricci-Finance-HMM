from __future__ import annotations
import numpy as np
import pandas as pd

def prepare_market_data(tickers, period="5y", interval="1d"):
    import yfinance as yf
    yf.set_tz_cache_location(".")
    raw = yf.download(list(tickers), period=period, interval=interval, auto_adjust=True,
                      group_by="column", progress=False, threads=True)
    if raw.empty:
        raise ValueError("yfinance returned no data")
    def field(name):
        if isinstance(raw.columns, pd.MultiIndex):
            if name not in raw.columns.get_level_values(0):
                return pd.DataFrame(index=raw.index)
            out=raw[name].copy()
        else:
            out=raw[[name]].copy() if name in raw.columns else pd.DataFrame(index=raw.index)
            if len(tickers)==1 and not out.empty: out.columns=[list(tickers)[0]]
        return out.apply(pd.to_numeric, errors="coerce")
    close=field("Close").dropna(axis=1, how="all").sort_index()
    volume=field("Volume").reindex(columns=close.columns).sort_index()
    close=close.ffill(limit=5)
    returns=np.log(close).diff().replace([np.inf,-np.inf],np.nan)
    returns=returns.dropna(how="all")
    dollar_volume=(close*volume).reindex(returns.index)
    close=close.reindex(returns.index)
    volume=volume.reindex(returns.index)
    return {"close":close,"volume":volume,"returns":returns,"dollar_volume":dollar_volume}
