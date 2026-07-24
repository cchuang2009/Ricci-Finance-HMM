from __future__ import annotations
import numpy as np
import networkx as nx

def edge_jaccard(a,b):
    if a is None or b is None: return 1.0
    ea={tuple(sorted(e)) for e in a.edges}; eb={tuple(sorted(e)) for e in b.edges}
    return len(ea&eb)/len(ea|eb) if ea|eb else 1.0

def graph_feature_row(G,date):
    curv=np.array([float(d.get("ricciCurvature",0)) for *_,d in G.edges(data=True)],float)
    corr=np.array([float(d.get("correlation",0)) for *_,d in G.edges(data=True)],float)
    n=max(G.number_of_nodes(),1); m=G.number_of_edges()
    return {"date":date,"nodes":n,"edges":m,"density":nx.density(G),
            "avg_degree":2*m/n,"avg_ricci":float(curv.mean()) if curv.size else 0.0,
            "std_ricci":float(curv.std()) if curv.size else 0.0,
            "negative_edge_ratio":float((curv<0).mean()) if curv.size else 0.0,
            "avg_abs_correlation":float(np.abs(corr).mean()) if corr.size else 0.0,
            "components":nx.number_connected_components(G)}

def capital_feature_row(G):
    nv=np.array([float(d.get("capital_share",0)) for _,d in G.nodes(data=True)])
    ev=np.array([float(d.get("edge_capital_flow",0)) for *_,d in G.edges(data=True)])
    def h(x):
        x=np.clip(x[np.isfinite(x)],0,None); s=x.sum()
        return float(((x/s)**2).sum()) if s>0 else 0.0
    return {"capital_concentration":h(nv),"edge_capital_concentration":h(ev)}
