from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
DEFAULT_HMM_FEATURES=("avg_ricci","std_ricci","negative_edge_ratio","density","edge_stability","capital_concentration","edge_capital_concentration")
@dataclass
class HMMResult:
    model: object; scaler: object; states: np.ndarray; posterior: np.ndarray; valid_index: np.ndarray; feature_names: list[str]

def fit_gaussian_hmm(feature_df, features=None, n_states=3, random_state=42):
    features=list(features or DEFAULT_HMM_FEATURES)
    missing=[c for c in features if c not in feature_df]
    if missing: raise ValueError(f"Missing HMM features: {missing}")
    X=feature_df[features].replace([np.inf,-np.inf],np.nan)
    mask=X.notna().all(axis=1).to_numpy(); Xv=X.loc[mask]
    if len(Xv)<max(12,n_states*4): raise ValueError(f"Only {len(Xv)} valid HMM rows")
    scaler=StandardScaler(); Z=scaler.fit_transform(Xv)
    model=GaussianHMM(n_components=n_states,covariance_type="diag",n_iter=500,min_covar=1e-4,random_state=random_state)
    model.fit(Z)
    states=model.predict(Z)
    posterior=model.predict_proba(Z)
    return HMMResult(model,scaler,states,posterior,np.flatnonzero(mask),features)

def build_regime_labels(feature_df, states):
    df=feature_df.iloc[-len(states):].copy(); df["state"]=states
    means=df.groupby("state")["avg_ricci"].mean().sort_values()
    names={}
    labels=["Stress","Transition","Normal","Expansion","Euphoria","Extreme"]
    for rank,state in enumerate(means.index): names[int(state)]=labels[min(rank,len(labels)-1)]
    return names

def current_run_length(states):
    if len(states)==0:return 0
    last=states[-1]; n=0
    for x in states[::-1]:
        if x!=last: break
        n+=1
    return n

def switch_rate(states):
    s=np.asarray(states); return float(np.mean(s[1:]!=s[:-1])) if len(s)>1 else 0.0

def forecast_hmm_methods(model, posterior_last, viterbi_last, horizon=5, n_sim=5000, random_state=42, current_run_length=1):
    P=np.asarray(model.transmat_,float); p=np.asarray(posterior_last,float); rows=[]
    for h in range(1,horizon+1):
        p=p@P; state=int(np.argmax(p)); rows.append({"day":h,"state":state,"probability":float(p[state]),"method":"posterior propagation"})
    return pd.DataFrame(rows)
