from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,Sequence
import numpy as np
import networkx as nx
@dataclass
class GNNResult:
    labels: np.ndarray; predictions: np.ndarray; probabilities: np.ndarray; train_indices: np.ndarray; test_indices: np.ndarray
    accuracy: float; balanced_accuracy: float; losses: list[float]; epochs: int; device: str; class_weights: np.ndarray; note: str

def graph_to_dense(G,nodes,sectors,sector_vocab):
    import torch
    idx={n:i for i,n in enumerate(nodes)}; N=len(nodes)
    A=np.zeros((N,N),np.float32); X=np.zeros((N,6+len(sector_vocab)),np.float32)
    for n,i in idx.items():
        if n in G:
            d=G.nodes[n]; X[i,:6]=[G.degree(n),float(d.get("capital_share",0)),float(d.get("ricciCurvature",0)),1.0,0.0,0.0]
            sec=sectors.get(str(n),"Other"); X[i,6+sector_vocab[sec]]=1
    for u,v,d in G.edges(data=True):
        if u in idx and v in idx:
            w=abs(float(d.get("correlation",1))); A[idx[u],idx[v]]=w; A[idx[v],idx[u]]=w
    A+=np.eye(N,dtype=np.float32); deg=np.maximum(A.sum(1),1e-8); D=np.diag(1/np.sqrt(deg)); A=D@A@D
    # graph-level features are represented by mean pooled node embeddings
    return torch.tensor(X),torch.tensor(A)

def train_gcn_regime(graphs:Sequence[nx.Graph], labels:Sequence[int], sectors:Dict[str,str], epochs:int=120, hidden:int=24, random_state:int=42)->GNNResult:
    import torch
    from torch import nn
    graphs=list(graphs); y=np.asarray(labels,dtype=np.int64).reshape(-1)
    if len(graphs)!=len(y): raise ValueError(f"Graph/label length mismatch: graphs={len(graphs)}, labels={len(y)}")
    if len(graphs)<10: raise ValueError("At least 10 graph snapshots are required")
    torch.manual_seed(random_state); np.random.seed(random_state)
    nodes=sorted({n for g in graphs for n in g.nodes},key=str)
    sec_names=sorted(set(sectors.get(str(n),"Other") for n in nodes)); sv={s:i for i,s in enumerate(sec_names)}
    pairs=[graph_to_dense(g,nodes,sectors,sv) for g in graphs]
    classes=np.unique(y); remap={c:i for i,c in enumerate(classes)}; yr=np.array([remap[c] for c in y])
    split=max(2,min(len(y)-2,int(len(y)*.7))); train_idx=np.arange(split); test_idx=np.arange(split,len(y))
    counts=np.bincount(yr[train_idx],minlength=len(classes)); weights=len(train_idx)/(len(classes)*np.maximum(counts,1))
    device="cuda" if torch.cuda.is_available() else "cpu"
    class GCN(nn.Module):
        def __init__(self,fin,h,c): super().__init__(); self.w1=nn.Linear(fin,h); self.w2=nn.Linear(h,h); self.out=nn.Linear(h,c)
        def forward(self,x,a):
            h=torch.relu(self.w1(a@x)); h=torch.relu(self.w2(a@h)); return self.out(h.mean(0))
    model=GCN(pairs[0][0].shape[1],hidden,len(classes)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-4)
    lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device)); losses=[]
    model.train()
    for ep in range(int(epochs)):
        opt.zero_grad(); logits=torch.stack([model(x.to(device),a.to(device)) for x,a in pairs[:split]])
        loss=lossfn(logits,torch.tensor(yr[:split],device=device)); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        logits=torch.stack([model(x.to(device),a.to(device)) for x,a in pairs]); probs=torch.softmax(logits,1).cpu().numpy(); pred_r=probs.argmax(1)
    pred=np.array([classes[i] for i in pred_r]); acc=float((pred[test_idx]==y[test_idx]).mean())
    recalls=[]
    for c in np.unique(y[test_idx]):
        mask=y[test_idx]==c; recalls.append(float((pred[test_idx][mask]==c).mean()))
    bal=float(np.mean(recalls)) if recalls else float('nan')
    return GNNResult(y,pred,probs,train_idx,test_idx,acc,bal,losses,len(losses),device,weights,
        "Pure-PyTorch dense GCN with chronological 70/30 split and class-weighted loss")
