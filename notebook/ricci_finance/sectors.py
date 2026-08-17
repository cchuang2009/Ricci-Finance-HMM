from __future__ import annotations

from collections.abc import Mapping
import pandas as pd

from ricci_finance.sector_objects import normalize_memberships

KNOWN = {
    "AAPL": "Technology", "AMAT": "Equipment", "AMD": "Semiconductors",
    "ANET": "Networking", "AVGO": "Semiconductors", "BNT": "QuantumComputing",
    "CBRS": "Semiconductors", "INTC": "Semiconductors", "IONQ": "QuantumComputing",
    "KLAC": "Equipment", "LITE": "Networking", "LRCX": "Equipment",
    "META": "Internet", "MRVL": "Semiconductors", "MU": "Memory",
    "NVDA": "Semiconductors", "QBTS": "QuantumComputing", "QNT": "QuantumComputing",
    "SPCX": "Other",
}


def parse_sector_map(text: str, required_tickers=()) -> dict[str, dict[str, float]]:
    """Parse either ``TICKER=Sector`` or weighted multi-sector rows.

    Weighted form::

        NVDA=Semiconductors:6|AI:3|DataCenter:1

    Values are normalized per ticker, so they need not already sum to one.
    """
    result: dict[str, dict[str, float]] = {}
    errors: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errors.append(f"line {line_no}: use TICKER=Sector[:weight]|Sector[:weight]")
            continue
        ticker, value_text = (part.strip() for part in line.split("=", 1))
        ticker = ticker.upper()
        if not ticker or not value_text:
            errors.append(f"line {line_no}: ticker and sectors must both be non-empty")
            continue

        raw_memberships: dict[str, float] = {}
        try:
            for item in value_text.split("|"):
                item = item.strip()
                if not item:
                    continue
                if ":" in item:
                    label, weight = item.rsplit(":", 1)
                    raw_memberships[label.strip()] = float(weight.strip())
                else:
                    raw_memberships[item] = 1.0
            result[ticker] = normalize_memberships(raw_memberships)
        except (TypeError, ValueError):
            errors.append(f"line {line_no}: invalid sector weight")

    required = [str(t).strip().upper() for t in required_tickers]
    missing = [ticker for ticker in required if ticker not in result]
    if errors or missing:
        messages = []
        if errors:
            messages.append("Invalid sector rows: " + "; ".join(errors))
        if missing:
            messages.append("Missing required sectors for: " + ", ".join(missing))
        raise ValueError(" ".join(messages))
    return result


def assign_sector_memberships(nodes, custom_map=None, require_all: bool = False):
    custom_map = {str(k).upper(): v for k, v in (custom_map or {}).items()}
    output: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for node in nodes:
        ticker = str(node).upper()
        if ticker in custom_map:
            output[str(node)] = normalize_memberships(custom_map[ticker])
        elif ticker in KNOWN:
            output[str(node)] = {KNOWN[ticker]: 1.0}
        else:
            output[str(node)] = {"Other": 1.0}
            missing.append(str(node))
    if require_all and missing:
        raise ValueError("Missing sector mapping for: " + ", ".join(sorted(missing)))
    return output


def primary_sector_map(memberships):
    output: dict[str, str] = {}
    for ticker, values in memberships.items():
        normalized = normalize_memberships(values)
        output[str(ticker)] = max(normalized, key=normalized.get)
    return output


def assign_sectors(nodes, custom_map=None, require_all: bool = False):
    """Backward-compatible primary-sector mapping used by V15 visualizations."""
    memberships = assign_sector_memberships(nodes, custom_map, require_all=require_all)
    return primary_sector_map(memberships)


def sector_momentum(close, sector_map, latest_date=None, lookback=5):
    x = close.loc[:latest_date] if latest_date is not None else close
    if len(x) <= lookback:
        return pd.Series(dtype=float, name="momentum")
    r = x.pct_change(lookback).iloc[-1].dropna()

    numerator: dict[str, float] = {}
    denominator: dict[str, float] = {}
    for ticker, value in r.items():
        memberships = normalize_memberships(sector_map.get(str(ticker), "Other"))
        for sector, weight in memberships.items():
            numerator[sector] = numerator.get(sector, 0.0) + float(value) * weight
            denominator[sector] = denominator.get(sector, 0.0) + weight
    result = {
        sector: numerator[sector] / max(denominator[sector], 1e-12)
        for sector in numerator
    }
    return pd.Series(result, name="momentum").sort_values(ascending=False)


def sector_flow_matrix(G, sectors, momentum=None):
    membership_map = {
        str(ticker): normalize_memberships(value)
        for ticker, value in sectors.items()
    }
    names = sorted({sector for values in membership_map.values() for sector in values})
    out = pd.DataFrame(0.0, index=names, columns=names)
    for u, v, data in G.edges(data=True):
        left = membership_map.get(str(u), {"Other": 1.0})
        right = membership_map.get(str(v), {"Other": 1.0})
        value = float(data.get("edge_capital_flow", 0.0))
        for a, wa in left.items():
            for b, wb in right.items():
                allocated = value * wa * wb
                out.loc[a, b] += allocated
                out.loc[b, a] += allocated
    return out
