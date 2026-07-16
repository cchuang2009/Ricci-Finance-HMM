from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import networkx as nx


def attach_capital_attributes(
    G: nx.Graph,
    dollar_volume_window: pd.DataFrame | None,
    capital_alpha: float = 0.35,
    use_capital_weighting: bool = True,
) -> nx.Graph:
    """Attach node market mass and edge transport to an existing graph.

    Capital mass is the median recent dollar volume in the window. Edge transport
    is the geometric mean of endpoint masses multiplied by positive correlation.
    Optional capital weighting contracts the correlation-only distance for pairs
    with similar capital scale, without changing the correlation itself.
    """
    H = G.copy()
    if dollar_volume_window is None or dollar_volume_window.empty:
        for n in H.nodes:
            H.nodes[n].setdefault("capital_mass", 0.0)
            H.nodes[n].setdefault("capital_share", 0.0)
        for u, v, d in H.edges(data=True):
            raw = float(d.get("distance", d.get("weight", 1.0)))
            d.setdefault("raw_distance", raw)
            d.setdefault("capital_similarity", 0.0)
            d.setdefault("edge_capital_flow", 0.0)
        return H

    dv = dollar_volume_window.reindex(columns=list(H.nodes()))
    mass = dv.replace([np.inf, -np.inf], np.nan).median(axis=0, skipna=True).fillna(0.0)
    mass = mass.clip(lower=0.0)
    total = float(mass.sum())

    for n in H.nodes:
        m = float(mass.get(n, 0.0))
        H.nodes[n]["capital_mass"] = m
        H.nodes[n]["capital_share"] = m / total if total > 0 else 0.0

    positive = mass[mass > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    log_mass = np.log1p(mass / max(scale, 1e-12))

    for u, v, d in H.edges(data=True):
        raw = float(d.get("distance", d.get("weight", 1.0)))
        mu = float(mass.get(u, 0.0))
        mv = float(mass.get(v, 0.0))
        lu = float(log_mass.get(u, 0.0))
        lv = float(log_mass.get(v, 0.0))
        similarity = float(np.exp(-abs(lu - lv)))
        rho = float(d.get("correlation", 0.0))
        flow = float(np.sqrt(max(mu, 0.0) * max(mv, 0.0)) * max(rho, 0.0))
        effective = raw
        if use_capital_weighting:
            effective = raw / (1.0 + float(capital_alpha) * similarity)
        d["raw_distance"] = raw
        d["capital_similarity"] = similarity
        d["edge_capital_flow"] = flow
        d["distance"] = effective
        d["weight"] = effective
    return H


def capital_flow_table(G: nx.Graph) -> pd.DataFrame:
    total = sum(float(d.get("edge_capital_flow", 0.0)) for *_, d in G.edges(data=True))
    rows = []
    for u, v, d in G.edges(data=True):
        flow = float(d.get("edge_capital_flow", 0.0))
        rows.append({
            "u": u,
            "v": v,
            "edge_capital_flow": flow,
            "edge_flow_share": flow / total if total > 0 else 0.0,
            "capital_similarity": float(d.get("capital_similarity", np.nan)),
            "effective_distance": float(d.get("distance", d.get("weight", np.nan))),
            "raw_corr_distance": float(d.get("raw_distance", np.nan)),
            "correlation": float(d.get("correlation", np.nan)),
            "ricciCurvature": float(d.get("ricciCurvature", 0.0)),
            "confidence": float(d.get("confidence", np.nan)),
        })
    return pd.DataFrame(rows).sort_values("edge_capital_flow", ascending=False) if rows else pd.DataFrame()


def node_capital_table(G: nx.Graph) -> pd.DataFrame:
    rows = [{
        "ticker": n,
        "status": d.get("status", "active"),
        "valid_obs": int(d.get("valid_obs", 0)),
        "capital_mass": float(d.get("capital_mass", 0.0)),
        "capital_share": float(d.get("capital_share", 0.0)),
        "degree": int(G.degree(n)),
    } for n, d in G.nodes(data=True)]
    return pd.DataFrame(rows).sort_values("capital_mass", ascending=False) if rows else pd.DataFrame()


def cluster_capital_table(
    G: nx.Graph,
    node_cluster: Dict[str, int] | None = None,
) -> pd.DataFrame:
    if node_cluster is None:
        from .graph import compute_components
        node_cluster = compute_components(G)
    df = pd.DataFrame([{
        "cluster": int(node_cluster.get(n, -1)),
        "ticker": n,
        "capital_mass": float(d.get("capital_mass", 0.0)),
    } for n, d in G.nodes(data=True)])
    if df.empty:
        return pd.DataFrame()
    total = float(df["capital_mass"].sum())
    out = df.groupby("cluster").agg(
        tickers=("ticker", lambda x: ", ".join(sorted(map(str, x)))),
        capital_mass=("capital_mass", "sum"),
        nodes=("ticker", "count"),
    ).reset_index()
    out["capital_share"] = out["capital_mass"] / total if total > 0 else 0.0
    return out.sort_values("capital_mass", ascending=False)
