from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .models import FrameData


@dataclass(frozen=True)
class FrameStory:
    frame_index: int
    date: str
    regime_name: str
    state_probability: float
    headline: str
    narrative: str
    changes: tuple[str, ...]
    strongest_fragile_edge: str
    strongest_capital_edge: str


def _signed_change(
    current: float,
    previous: float,
    label: str,
    threshold: float,
    rising_text: str,
    falling_text: str,
) -> str | None:
    if not np.isfinite(current) or not np.isfinite(previous):
        return None

    delta = current - previous

    if delta >= threshold:
        return f"{rising_text} ({label} {previous:.3f} → {current:.3f})."

    if delta <= -threshold:
        return f"{falling_text} ({label} {previous:.3f} → {current:.3f})."

    return None


def _edge_labels(frame: FrameData) -> tuple[str, str]:
    fragile: tuple[str, str, float] | None = None
    capital: tuple[str, str, float] | None = None

    for u, v, data in frame.G.edges(data=True):
        curvature = float(data.get("ricciCurvature", 0.0))
        flow = float(data.get("edge_capital_flow", 0.0))

        if fragile is None or curvature < fragile[2]:
            fragile = (str(u), str(v), curvature)

        if capital is None or flow > capital[2]:
            capital = (str(u), str(v), flow)

    fragile_label = (
        f"{fragile[0]}–{fragile[1]} (κ={fragile[2]:.3f})"
        if fragile is not None
        else "No active edge"
    )
    capital_label = (
        f"{capital[0]}–{capital[1]} (flow={capital[2]:,.0f})"
        if capital is not None
        else "No active edge"
    )

    return fragile_label, capital_label


def describe_frame(
    frame: FrameData,
    previous: FrameData | None,
    frame_index: int,
) -> FrameStory:
    current = frame.stats
    probability = float(
        frame.metadata.get("hmm_probability", np.nan)
    )
    regime = current.regime_name

    changes: list[str] = []

    if previous is None:
        changes.append(
            "This is the first displayed rolling window; no prior frame is "
            "available for a frame-to-frame comparison."
        )
    else:
        prior = previous.stats

        candidates = [
            _signed_change(
                current.avg_ricci,
                prior.avg_ricci,
                "average curvature",
                0.03,
                "Network cohesion improved",
                "Network curvature weakened",
            ),
            _signed_change(
                current.negative_edge_ratio,
                prior.negative_edge_ratio,
                "negative-edge ratio",
                0.08,
                "Fragile bridge relationships increased",
                "The share of fragile bridges decreased",
            ),
            _signed_change(
                current.density,
                prior.density,
                "density",
                0.04,
                "Cross-network connectivity strengthened",
                "Cross-network connectivity weakened",
            ),
            _signed_change(
                current.largest_component_ratio,
                prior.largest_component_ratio,
                "largest-component ratio",
                0.08,
                "The dominant connected market basin expanded",
                "The dominant connected basin contracted",
            ),
            _signed_change(
                current.max_node_capital_share,
                prior.max_node_capital_share,
                "maximum node capital share",
                0.05,
                "Capital concentration increased",
                "Capital concentration became more distributed",
            ),
        ]

        changes.extend(item for item in candidates if item)

        cluster_delta = current.num_clusters - prior.num_clusters
        if cluster_delta > 0:
            changes.append(
                f"The graph split into {cluster_delta} additional cluster(s)."
            )
        elif cluster_delta < 0:
            changes.append(
                f"The graph consolidated by {-cluster_delta} cluster(s)."
            )

        if np.isfinite(current.edge_stability):
            if current.edge_stability < 0.45:
                changes.append(
                    f"Edge stability is low ({current.edge_stability:.2f}); "
                    "the network structure changed materially."
                )
            elif current.edge_stability > 0.75:
                changes.append(
                    f"Edge stability is high ({current.edge_stability:.2f}); "
                    "the network structure persisted."
                )

    if not changes:
        changes.append(
            "No graph statistic crossed the current narrative threshold; "
            "the frame is broadly stable relative to the previous window."
        )

    fragile_edge, capital_edge = _edge_labels(frame)

    if "stress" in regime.lower():
        headline = "Fragmentation and fragile bridges dominate"
    elif "transition" in regime.lower():
        headline = "The market is rotating between network basins"
    elif "coherent" in regime.lower():
        headline = "The network is relatively cohesive"
    else:
        headline = "Market-network regime is not classified"

    probability_text = (
        f" with {probability:.0%} posterior confidence"
        if np.isfinite(probability)
        else ""
    )

    narrative = (
        f"At {current.end_date}, the HMM classifies the rolling market graph "
        f"as “{regime}”{probability_text}. "
        + " ".join(changes[:3])
        + f" The most negative-curvature edge is {fragile_edge}; "
        f"the largest capital-transport edge is {capital_edge}."
    )

    return FrameStory(
        frame_index=int(frame_index),
        date=str(current.end_date),
        regime_name=str(regime),
        state_probability=probability,
        headline=headline,
        narrative=narrative,
        changes=tuple(changes),
        strongest_fragile_edge=fragile_edge,
        strongest_capital_edge=capital_edge,
    )


def build_frame_stories(
    frames: Sequence[FrameData],
) -> list[FrameStory]:
    stories: list[FrameStory] = []

    for index, frame in enumerate(frames):
        previous = frames[index - 1] if index > 0 else None
        stories.append(
            describe_frame(frame, previous, index)
        )

    return stories


def frame_story_table(
    stories: Sequence[FrameStory],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": story.frame_index,
                "date": story.date,
                "regime_name": story.regime_name,
                "state_probability": story.state_probability,
                "headline": story.headline,
                "narrative": story.narrative,
                "strongest_fragile_edge": story.strongest_fragile_edge,
                "strongest_capital_edge": story.strongest_capital_edge,
            }
            for story in stories
        ]
    )
