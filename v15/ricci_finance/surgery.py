from __future__ import annotations
def graph_surgery(G, curvature_threshold=-1.0, min_component_size=2):
    H=G.copy(); removed=[]
    for u,v,d in list(H.edges(data=True)):
        if float(d.get("ricciCurvature",0))<curvature_threshold: H.remove_edge(u,v); removed.append((u,v))
    return H,{"removed_edges":removed}
