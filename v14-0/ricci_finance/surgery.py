from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx


def surgery_risk_direction_table(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        kappa = float(d.get("ricciCurvature", 0.0))
        distance = float(d.get("distance", d.get("weight", 1.0)))
        flow = float(d.get("edge_capital_flow", 0.0))
        confidence = float(d.get("confidence", 1.0))
        bridge_bonus = 1.25 if bool(d.get("bridge", False)) or nx.has_bridges(G) and (u, v) in set(nx.bridges(G)) else 1.0
        risk = max(0.0, -kappa) * max(distance, 0.0) * np.log1p(max(flow, 0.0)) * confidence * bridge_bonus
        rows.append({
            "u": u, "v": v,
            "ricciCurvature": kappa,
            "distance": distance,
            "edge_capital_flow": flow,
            "confidence": confidence,
            "bridge_like": bridge_bonus > 1.0,
            "surgery_risk_direction": float(risk),
            "interpretation": "possible future separation" if risk > 0 else "normal / coherent",
        })
    cols = ["u","v","ricciCurvature","distance","edge_capital_flow","confidence",
            "bridge_like","surgery_risk_direction","interpretation"]
    return pd.DataFrame(rows).sort_values("surgery_risk_direction", ascending=False) if rows else pd.DataFrame(columns=cols)


def graph_topology_stats(G: nx.Graph) -> dict:
    comps = list(nx.connected_components(G))
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(comps),
        "largest_component": max((len(c) for c in comps), default=0),
        "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        "bridges": len(list(nx.bridges(G))),
    }


def perform_financial_surgery(
    G: nx.Graph,
    curvature_threshold: float = -0.20,
    long_edge_quantile: float = 0.70,
    require_bridge_or_long: bool = True,
):
    """Optional experimental surgery retained for lecture comparison.

    The default v12 app uses risk direction only. This function cuts qualifying
    edges only when explicitly called by the user.
    """
    H = G.copy()
    distances = [float(d.get("distance", d.get("weight", 1.0))) for *_, d in H.edges(data=True)]
    long_cut = float(np.quantile(distances, long_edge_quantile)) if distances else float("inf")
    bridges = {tuple(sorted(e)) for e in nx.bridges(H)}
    removed, rows = [], []
    for u, v, d in list(H.edges(data=True)):
        k = float(d.get("ricciCurvature", 0.0))
        w = float(d.get("distance", d.get("weight", 1.0)))
        is_bridge = tuple(sorted((u, v))) in bridges
        is_long = w >= long_cut
        qualifies = k <= curvature_threshold and ((is_bridge or is_long) if require_bridge_or_long else True)
        if qualifies:
            H.remove_edge(u, v)
            removed.append((u, v))
        rows.append({
            "u": u, "v": v, "ricciCurvature": k, "distance": w,
            "is_bridge": is_bridge, "is_long": is_long, "removed": qualifies,
        })
    return H, removed, pd.DataFrame(rows).sort_values(["removed", "ricciCurvature"], ascending=[False, True])
