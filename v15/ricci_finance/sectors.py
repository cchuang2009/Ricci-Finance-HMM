from __future__ import annotations

import pandas as pd

KNOWN = {
    "AAPL": "Technology", "AMAT": "Equipment", "AMD": "Semiconductors",
    "ANET": "Networking", "AVGO": "Semiconductors", "BNT": "QuantumComputing",
    "CBRS": "Semiconductors", "INTC": "Semiconductors", "IONQ": "QuantumComputing",
    "KLAC": "Equipment", "LITE": "Networking", "LRCX": "Equipment",
    "META": "Internet", "MRVL": "Semiconductors", "MU": "Memory",
    "NVDA": "Semiconductors", "QBTS": "QuantumComputing", "QNT": "QuantumComputing",
    "SPCX": "Other",
}


def parse_sector_map(text: str, required_tickers=()) -> dict[str, str]:
    """Parse editable ``TICKER=Sector`` lines and require full ticker coverage."""
    result: dict[str, str] = {}
    errors: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errors.append(f"line {line_no}: use TICKER=Sector")
            continue
        ticker, sector = (part.strip() for part in line.split("=", 1))
        ticker = ticker.upper()
        if not ticker or not sector:
            errors.append(f"line {line_no}: ticker and sector must both be non-empty")
            continue
        result[ticker] = sector

    required = [str(t).strip().upper() for t in required_tickers]
    missing = [ticker for ticker in required if not result.get(ticker)]
    if errors or missing:
        messages = []
        if errors:
            messages.append("Invalid sector rows: " + "; ".join(errors))
        if missing:
            messages.append("Missing required sectors for: " + ", ".join(missing))
        raise ValueError(" ".join(messages))
    return result


def assign_sectors(nodes, custom_map=None, require_all: bool = False):
    mapping = dict(KNOWN)
    if custom_map:
        mapping.update({str(k).upper(): str(v) for k, v in custom_map.items()})
    output = {str(node): mapping.get(str(node).upper(), "Other") for node in nodes}
    if require_all:
        missing = [str(node) for node in nodes if str(node).upper() not in mapping]
        if missing:
            raise ValueError("Missing sector mapping for: " + ", ".join(sorted(missing)))
    return output


def sector_momentum(close, sector_map, latest_date=None, lookback=5):
    x = close.loc[:latest_date] if latest_date is not None else close
    if len(x) <= lookback:
        return pd.Series(dtype=float, name="momentum")
    r = x.pct_change(lookback).iloc[-1].dropna()
    d = pd.DataFrame({"ticker": r.index.astype(str), "momentum": r.to_numpy(dtype=float)})
    d["sector"] = d["ticker"].map(sector_map).fillna("Other")
    return d.groupby("sector")["momentum"].mean().sort_values(ascending=False)


def sector_flow_matrix(G, sectors, momentum=None):
    names = sorted(set(sectors.values()))
    out = pd.DataFrame(0.0, index=names, columns=names)
    for u, v, data in G.edges(data=True):
        a = sectors.get(str(u), "Other")
        b = sectors.get(str(v), "Other")
        value = float(data.get("edge_capital_flow", 0))
        out.loc[a, b] += value
        out.loc[b, a] += value
    return out
