from __future__ import annotations
import pandas as pd
KNOWN={"NVDA":"Semiconductors","AMD":"Semiconductors","AVGO":"Semiconductors","MRVL":"Semiconductors","MU":"Memory","INTC":"Semiconductors","LRCX":"Equipment","KLAC":"Equipment","AMAT":"Equipment","ASML":"Equipment","ANET":"Networking","AAPL":"Technology","MSFT":"Software","GOOGL":"Internet","AMZN":"Consumer/Cloud","META":"Internet"}
def assign_sectors(nodes): return {str(n):KNOWN.get(str(n),"Other") for n in nodes}
def sector_momentum(close,sectors,latest_date=None,lookback=5):
    x=close.loc[:latest_date] if latest_date is not None else close
    r=x.pct_change(lookback).iloc[-1]
    d=pd.DataFrame({"ticker":r.index,"momentum":r.values}); d["sector"]=d.ticker.map(sectors).fillna("Other")
    return d.groupby("sector").momentum.mean().sort_values(ascending=False)
def sector_flow_matrix(G,sectors,momentum=None):
    names=sorted(set(sectors.values())); out=pd.DataFrame(0.0,index=names,columns=names)
    for u,v,d in G.edges(data=True):
        a=sectors.get(str(u),"Other"); b=sectors.get(str(v),"Other"); val=float(d.get("edge_capital_flow",0))
        out.loc[a,b]+=val; out.loc[b,a]+=val
    return out
