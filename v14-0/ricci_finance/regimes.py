from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .models import FrameData
from .rolling import rolling_feature_table

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
except Exception:
    GaussianHMM = None
    StandardScaler = None


FEATURES = [
    "avg_ricci",
    "ricci_std",
    "ricci_min",
    "negative_edge_ratio",
    "clusters",
    "largest_component_ratio",
    "edges",
    "density",
    "component_entropy",
    "edge_stability",
    "total_edge_capital_flow",
    "max_node_capital_share",
]


def _fallback_result(
    out: pd.DataFrame,
    frames: Sequence[FrameData] | None,
    message: str,
) -> tuple[pd.DataFrame, dict[int, str]]:
    out["hmm_state"] = -1
    out["regime_name"] = message
    out["state_probability"] = np.nan

    if frames is not None:
        for frame in frames:
            frame.stats.hmm_state = -1
            frame.stats.regime_name = message
            frame.metadata["hmm_probabilities"] = {}
            frame.metadata["hmm_probability"] = np.nan

    return out, {-1: message}


def compute_hmm_regimes(
    feature_df: pd.DataFrame | None = None,
    *,
    frames: Sequence[FrameData] | None = None,
    returns: pd.DataFrame | None = None,
    starts: Sequence[int] | None = None,
    window_size: int | None = None,
    n_components: int = 3,
    forward_days: int = 5,
    random_state: int = 42,
    min_covar: float = 1e-4,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Fit an HMM and attach posterior state probabilities to each frame.

    The function deliberately uses ``fit`` followed by ``predict`` because
    some hmmlearn releases do not implement ``fit_predict``.
    """
    if feature_df is None:
        if frames is None:
            raise ValueError("Provide feature_df or frames.")
        feature_df = rolling_feature_table(list(frames))

    out = feature_df.copy().reset_index(drop=True)

    if (
        returns is not None
        and starts is not None
        and window_size is not None
    ):
        market_return = returns.mean(axis=1, skipna=True)
        forward_values: list[float] = []

        for start in starts:
            window_end = min(
                int(start) + int(window_size) - 1,
                len(market_return) - 1,
            )
            forecast_end = min(
                window_end + int(forward_days),
                len(market_return) - 1,
            )

            if window_end + 1 <= forecast_end:
                value = market_return.iloc[
                    window_end + 1 : forecast_end + 1
                ].sum()
                forward_values.append(float(value))
            else:
                forward_values.append(float("nan"))

        out[f"next_{forward_days}d_market_return"] = forward_values

    feature_columns = [
        column for column in FEATURES if column in out.columns
    ]

    if not feature_columns:
        raise ValueError("No valid HMM feature columns were found.")

    X = (
        out[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    X = X.interpolate(limit_direction="both").fillna(0.0)

    minimum_rows = max(10, int(n_components) * 3)

    if (
        GaussianHMM is None
        or StandardScaler is None
        or len(X) < minimum_rows
    ):
        return _fallback_result(
            out,
            frames,
            "HMM unavailable or too few frames",
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=int(n_components),
        covariance_type="diag",
        min_covar=float(min_covar),
        n_iter=500,
        random_state=int(random_state),
    )

    model.fit(X_scaled)
    states = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    out["hmm_state"] = states.astype(int)

    summary = out.groupby("hmm_state").agg(
        avg_ricci=("avg_ricci", "mean"),
        density=("density", "mean"),
        largest_ratio=("largest_component_ratio", "mean"),
        negative_ratio=("negative_edge_ratio", "mean"),
    )

    # Lower curvature and weaker connectivity are treated as more stressed.
    ordered_states = list(
        summary.sort_values(
            ["avg_ricci", "largest_ratio", "density"]
        ).index
    )

    default_names = [
        "stress / fragmentation",
        "transition / rotation",
        "coherent risk-on",
    ]

    labels: dict[int, str] = {}
    for index, state in enumerate(ordered_states):
        if index < len(default_names):
            labels[int(state)] = default_names[index]
        else:
            labels[int(state)] = f"regime_{index}"

    out["regime_name"] = out["hmm_state"].map(labels)

    for state in range(probabilities.shape[1]):
        out[f"state_probability_{state}"] = probabilities[:, state]

    selected_probability = probabilities[
        np.arange(len(states)),
        states.astype(int),
    ]
    out["state_probability"] = selected_probability

    if frames is not None:
        for index, frame in enumerate(frames):
            if index >= len(out):
                break

            state = int(out.loc[index, "hmm_state"])
            probability_map = {
                int(s): float(probabilities[index, s])
                for s in range(probabilities.shape[1])
            }

            frame.stats.hmm_state = state
            frame.stats.regime_name = str(
                out.loc[index, "regime_name"]
            )
            frame.metadata["hmm_probabilities"] = probability_map
            frame.metadata["hmm_probability"] = float(
                selected_probability[index]
            )
            frame.metadata["hmm_labels"] = labels

    return out, labels
