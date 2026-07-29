import numpy as np
import pandas as pd

from ricci_finance.gnn import train_gcn_regime
from ricci_finance.hmm import DEFAULT_HMM_FEATURES, fit_gaussian_hmm
from ricci_finance.pipeline import build_rolling_frames
from ricci_finance.sectors import assign_sectors


def test_pipeline_forman_and_gnn():
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-01", periods=300)
    columns = ["NVDA", "AMD", "MU", "ANET", "LRCX"]
    returns = pd.DataFrame(
        rng.normal(0, 0.02, (300, 5)), index=index, columns=columns
    )
    dollar_volume = pd.DataFrame(
        rng.lognormal(20, 1, (300, 5)), index=index, columns=columns
    )
    frames, features = build_rolling_frames(
        returns,
        dollar_volume,
        window=40,
        step=5,
        max_frames=40,
        curvature_engine="forman",
    )
    hmm = fit_gaussian_hmm(
        features, list(DEFAULT_HMM_FEATURES), n_states=2
    )
    graphs = [frames[index]["graph"] for index in hmm.valid_index]
    result = train_gcn_regime(
        graphs,
        hmm.states,
        assign_sectors(columns),
        epochs=3,
        hidden=8,
    )
    assert len(result.predictions) == len(hmm.states)


def test_pipeline_ollivier_lp():
    rng = np.random.default_rng(7)
    index = pd.date_range("2022-01-01", periods=90)
    columns = ["A", "B", "C", "D"]
    returns = pd.DataFrame(
        rng.normal(0, 0.02, (90, 4)), index=index, columns=columns
    )
    dollar_volume = pd.DataFrame(
        rng.lognormal(18, 0.5, (90, 4)), index=index, columns=columns
    )
    frames, features = build_rolling_frames(
        returns,
        dollar_volume,
        window=30,
        step=20,
        max_frames=3,
        min_corr=0.0,
        curvature_engine="ollivier_lp",
    )
    assert len(frames) == len(features) > 0
    assert all(
        frame["graph"].graph["curvature_engine"] == "ollivier_lp"
        for frame in frames
    )
