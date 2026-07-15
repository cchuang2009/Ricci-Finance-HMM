from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx


def summarize_edges(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "u": u, "v": v,
            "distance": float(d.get("distance", d.get("weight", np.nan))),
            "raw_distance": float(d.get("raw_distance", np.nan)),
            "correlation": float(d.get("correlation", np.nan)),
            "ricciCurvature": float(d.get("ricciCurvature", 0.0)),
            "edge_capital_flow": float(d.get("edge_capital_flow", 0.0)),
            "capital_similarity": float(d.get("capital_similarity", np.nan)),
            "overlap_n": int(d.get("overlap_n", 0)),
            "confidence": float(d.get("confidence", np.nan)),
            "edge_source": "bridge" if d.get("bridge", False) else "graph",
        })
    cols = ["u","v","distance","raw_distance","correlation","ricciCurvature",
            "edge_capital_flow","capital_similarity","overlap_n","confidence","edge_source"]
    return pd.DataFrame(rows).sort_values("distance") if rows else pd.DataFrame(columns=cols)
