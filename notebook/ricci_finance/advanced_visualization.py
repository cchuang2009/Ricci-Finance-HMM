from __future__ import annotations

import math
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .visualization import PASTEL, sector_palette


def reduce_embeddings(embeddings: np.ndarray, dimensions: int = 2, method: str = "PCA", random_state: int = 42) -> np.ndarray:
    values = np.asarray(embeddings, dtype=float)
    if len(values) == 0:
        return np.empty((0, dimensions))
    dimensions = min(dimensions, values.shape[1], max(1, len(values)))
    selected = method.upper()
    if selected == "UMAP":
        try:
            import umap
            return umap.UMAP(n_components=dimensions, random_state=random_state).fit_transform(values)
        except Exception:
            selected = "PCA"
    if selected in {"T-SNE", "TSNE"} and len(values) >= 4:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, len(values) // 3), len(values) - 1)
        return TSNE(n_components=dimensions, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state).fit_transform(values)
    from sklearn.decomposition import PCA
    return PCA(n_components=dimensions, random_state=random_state).fit_transform(values)


def embedding_figure(coords, labels, dates, dimensions=2, title="GNN latent embeddings"):
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    hover = [str(pd.Timestamp(d).date()) for d in dates]
    if dimensions == 3 and coords.shape[1] >= 3:
        fig = go.Figure(go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode="markers+text",
            text=[str(x) for x in labels], customdata=hover,
            hovertemplate="Date=%{customdata}<br>Cluster/state=%{text}<extra></extra>",
            marker=dict(size=7, color=labels, colorscale="Viridis", showscale=True)))
        fig.update_layout(scene=dict(xaxis_title="Embedding 1", yaxis_title="Embedding 2", zaxis_title="Embedding 3"))
    else:
        fig = go.Figure(go.Scatter(x=coords[:,0], y=coords[:,1], mode="markers+text",
            text=[str(x) for x in labels], customdata=hover,
            hovertemplate="Date=%{customdata}<br>Cluster/state=%{text}<extra></extra>",
            marker=dict(size=10, color=labels, colorscale="Viridis", showscale=True)))
        fig.update_xaxes(title="Embedding 1"); fig.update_yaxes(title="Embedding 2")
    fig.update_layout(title=title, template="plotly_white", height=650, margin=dict(l=20,r=20,t=60,b=20))
    return fig


def attention_heatmap(attention: np.ndarray, nodes: Sequence[str], title="GAT attention") -> go.Figure:
    matrix = np.asarray(attention, dtype=float)
    fig = go.Figure(go.Heatmap(z=matrix, x=list(nodes), y=list(nodes), colorscale="Viridis", colorbar=dict(title="Attention")))
    fig.update_layout(title=title, template="plotly_white", height=700, xaxis_title="Attended node", yaxis_title="Query node")
    return fig


def community_animation(frames: Sequence[dict], community_history: Sequence[Mapping[str,int]], interval_ms: int = 800) -> go.Figure:
    all_nodes = sorted({str(n) for frame in frames for n in frame["graph"].nodes})
    union = nx.Graph()
    union.add_nodes_from(all_nodes)
    for frame in frames:
        union.add_edges_from((str(u), str(v)) for u,v in frame["graph"].edges)
    pos = nx.spring_layout(union, seed=42, iterations=180)

    def traces(index):
        graph = frames[index]["graph"]
        labels = community_history[index]
        ex, ey = [], []
        for u,v in graph.edges:
            ex += [pos[str(u)][0], pos[str(v)][0], None]
            ey += [pos[str(u)][1], pos[str(v)][1], None]
        edge = go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1,color="rgba(100,120,140,.30)"),hoverinfo="skip")
        nodes = [str(n) for n in graph.nodes]
        node = go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
            text=nodes, textposition="middle center",
            marker=dict(size=28,color=[labels.get(n,-1) for n in nodes],colorscale="Turbo",showscale=False,line=dict(width=1,color="white")),
            customdata=[labels.get(n,-1) for n in nodes], hovertemplate="%{text}<br>Community=%{customdata}<extra></extra>")
        return [edge,node]

    plot_frames = [go.Frame(data=traces(i), name=str(i)) for i in range(len(frames))]
    fig = go.Figure(data=traces(0), frames=plot_frames)
    fig.update_layout(title="Temporal Ricci-community evolution", template="plotly_white", height=720,
        xaxis=dict(visible=False), yaxis=dict(visible=False,scaleanchor="x"),
        updatemenus=[dict(type="buttons",buttons=[dict(label="Play",method="animate",args=[None,{"frame":{"duration":interval_ms,"redraw":True},"fromcurrent":True}]),dict(label="Pause",method="animate",args=[[None],{"frame":{"duration":0,"redraw":False},"mode":"immediate"}])])],
        sliders=[dict(steps=[dict(method="animate",args=[[str(i)],{"mode":"immediate","frame":{"duration":0,"redraw":True}}],label=str(pd.Timestamp(frame["date"]).date())) for i,frame in enumerate(frames)])])
    return fig


def galaxy_animation(frames: Sequence[dict], dynamic_memberships: Sequence[Mapping[str,Mapping[str,float]]], embeddings: np.ndarray | None = None, interval_ms: int = 800) -> go.Figure:
    sectors = sorted({s for frame in dynamic_memberships for weights in frame.values() for s in weights})
    angles = {s: 2*math.pi*i/max(len(sectors),1) for i,s in enumerate(sectors)}
    palette = {s: PASTEL[i % len(PASTEL)] for i,s in enumerate(sectors)}

    def node_trace(i):
        graph = frames[i]["graph"]
        membership = dynamic_memberships[i]
        nodes = [str(n) for n in graph.nodes]
        x=[];y=[];z=[];colors=[];sizes=[];hover=[]
        emb_z = None
        if embeddings is not None and i < len(embeddings):
            emb_z = float(np.asarray(embeddings)[i,0])
        for j,n in enumerate(nodes):
            weights = membership.get(n,{"Other":1.0})
            vx=sum(w*math.cos(angles.get(s,0.0)) for s,w in weights.items())
            vy=sum(w*math.sin(angles.get(s,0.0)) for s,w in weights.items())
            angle=math.atan2(vy,vx)
            share=max(float(graph.nodes[n].get("capital_share",0.0)),0.0)
            radius=1.5+5.0*math.sqrt(share+1e-8)
            height=emb_z if emb_z is not None else float(graph.nodes[n].get("ricciCurvature",0.0))
            x.append(radius*math.cos(angle));y.append(radius*math.sin(angle));z.append(height)
            primary=max(weights,key=weights.get);colors.append(palette.get(primary,"#CCCCCC"));sizes.append(10+40*math.sqrt(share+1e-8))
            hover.append(f"<b>{n}</b><br>Primary={primary}<br>Capital={share:.2%}<br>Height={height:.3f}<br>"+"<br>".join(f"{s}: {w:.2f}" for s,w in sorted(weights.items(),key=lambda x:-x[1])[:5]))
        return go.Scatter3d(x=x,y=y,z=z,mode="markers+text",text=nodes,textposition="middle center",marker=dict(size=sizes,color=colors,opacity=.78,line=dict(width=1,color="white")),customdata=hover,hovertemplate="%{customdata}<extra></extra>")

    plot_frames=[go.Frame(data=[node_trace(i)],name=str(i)) for i in range(len(frames))]
    fig=go.Figure(data=[node_trace(0)],frames=plot_frames)
    fig.update_layout(title="Dynamic 3D sector Galaxy",height=800,template="plotly_white",margin=dict(l=0,r=0,t=70,b=0),scene=dict(bgcolor="#172233",xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(title="Ricci curvature / GNN embedding")),updatemenus=[dict(type="buttons",buttons=[dict(label="Play",method="animate",args=[None,{"frame":{"duration":interval_ms,"redraw":True},"fromcurrent":True}]),dict(label="Pause",method="animate",args=[[None],{"mode":"immediate"}])])],sliders=[dict(steps=[dict(method="animate",args=[[str(i)],{"mode":"immediate","frame":{"duration":0,"redraw":True}}],label=str(pd.Timestamp(frame["date"]).date())) for i,frame in enumerate(frames)])])
    return fig
