from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
except Exception:
    OllivierRicci = None


def compute_ricci_curvature(
    G: nx.Graph,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
) -> nx.Graph:
    H = G.copy()
    if H.number_of_edges() == 0:
        return H
    if OllivierRicci is not None:
        try:
            orc = OllivierRicci(
                H, alpha=alpha, method=method, weight="weight",
                proc=proc, verbose="ERROR"
            )
            orc.compute_ricci_curvature()
            return orc.G.copy()
        except Exception:
            pass
    for u, v in H.edges():
        cn = len(list(nx.common_neighbors(H, u, v)))
        deg_sum = max(1, H.degree(u) + H.degree(v) - 2)
        H[u][v]["ricciCurvature"] = float(2 * cn / deg_sum - 0.5)
    return H


def run_ricci_flow(
    G: nx.Graph,
    iterations: int = 8,
    step_size: float = 0.25,
    alpha: float = 0.5,
    method: str = "OTD",
    proc: int = 1,
    normalize_mean: bool = True,
) -> tuple[nx.Graph, pd.DataFrame]:
    H = G.copy()
    history = []
    initial_mean = np.mean([d.get("weight", 1.0) for *_, d in H.edges(data=True)]) if H.edges else 1.0
    for it in range(iterations + 1):
        curv = [float(d.get("ricciCurvature", 0.0)) for *_, d in H.edges(data=True)]
        weights = [float(d.get("weight", 1.0)) for *_, d in H.edges(data=True)]
        history.append({
            "iteration": it,
            "avg_weight": float(np.mean(weights)) if weights else np.nan,
            "avg_ricci": float(np.mean(curv)) if curv else np.nan,
            "ricci_min": float(np.min(curv)) if curv else np.nan,
            "ricci_max": float(np.max(curv)) if curv else np.nan,
        })
        if it == iterations:
            break
        for u, v, d in H.edges(data=True):
            w = float(d.get("weight", 1.0))
            k = float(d.get("ricciCurvature", 0.0))
            nw = float(np.clip(w * (1.0 - step_size * k), 1e-6, 1e6))
            H[u][v]["weight"] = nw
            H[u][v]["distance"] = nw
        if normalize_mean and H.number_of_edges():
            now = np.mean([d["weight"] for *_, d in H.edges(data=True)])
            scale = initial_mean / now if now else 1.0
            for u, v in H.edges():
                H[u][v]["weight"] *= scale
                H[u][v]["distance"] = H[u][v]["weight"]
        H = compute_ricci_curvature(H, alpha=alpha, method=method, proc=proc)
    return H, pd.DataFrame(history)


def compare_before_after_flow(before: nx.Graph, after: nx.Graph) -> pd.DataFrame:
    keys = {tuple(sorted(e)) for e in before.edges()} | {tuple(sorted(e)) for e in after.edges()}
    rows = []
    for u, v in sorted(keys):
        b = before.get_edge_data(u, v, default={})
        a = after.get_edge_data(u, v, default={})
        bw = float(b.get("weight", np.nan)) if b else np.nan
        aw = float(a.get("weight", np.nan)) if a else np.nan
        bk = float(b.get("ricciCurvature", np.nan)) if b else np.nan
        ak = float(a.get("ricciCurvature", np.nan)) if a else np.nan
        rows.append({
            "u": u, "v": v, "before_weight": bw, "after_weight": aw,
            "delta_weight": aw - bw, "before_ricci": bk, "after_ricci": ak,
            "delta_ricci": ak - bk,
        })
    return pd.DataFrame(rows)
