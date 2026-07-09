#!/usr/bin/env python3
"""
make_movie.py

Create an MP4 introduction movie for the Ricci-Curvature-Finance project.

The movie explains the project flow:
    load market data -> returns -> correlation -> financial graph
    -> Ollivier/Ricci curvature -> Ricci flow -> capital flow
    -> HMM regimes -> dynamic 3D manifold -> investment insight.

Designed to live beside your existing helper.py from Ricci Finance v11.
If helper.py is available, this script reuses its data generation and graph logic.
If not, it falls back to a compact internal synthetic-data implementation.

Usage
-----
Install dependencies:
    pip install moviepy matplotlib numpy pandas networkx pillow

Optional, if you want Plotly/true project integration:
    pip install plotly yfinance GraphRicciCurvature pot networkit hmmlearn scikit-learn

Run:
    python make_movie.py

Output:
    output/ricci_finance_intro.mp4

Streamlit embedding:
    st.video("output/ricci_finance_intro.mp4")
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# MoviePy v1/v2 compatible imports.
try:
    from moviepy.editor import (
        AudioClip,
        CompositeVideoClip,
        ImageClip,
        ImageSequenceClip,
        TextClip,
        VideoClip,
        concatenate_videoclips,
        vfx,
    )
except Exception:  # pragma: no cover - MoviePy v2 style fallback
    from moviepy import (  # type: ignore
        AudioClip,
        CompositeVideoClip,
        ImageClip,
        ImageSequenceClip,
        TextClip,
        VideoClip,
        concatenate_videoclips,
        vfx,
    )


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

W, H = 1920, 1080
FPS = 30
DEFAULT_TICKERS = [
    "NVDA", "AMD", "AVGO", "TSM", "MU", "MRVL", "AMAT", "LRCX", "KLAC",
    "ANET", "AAOI", "COHR", "LITE", "SMCI", "PLTR", "IONQ", "QBTS", "QUBT",
    "RGTI", "NBIS", "QNT", "SPCX",
]

BG = (7, 11, 24)
PANEL = (14, 23, 45)
TEXT = (235, 240, 255)
MUTED = (145, 160, 190)
CYAN = (56, 205, 255)
BLUE = (55, 125, 255)
RED = (255, 90, 90)
GOLD = (255, 205, 80)
GREEN = (90, 230, 160)
WHITE = (255, 255, 255)
GRAY = (120, 130, 150)


@dataclass
class MovieAssets:
    root: Path
    frames: Path
    output: Path
    stills: Path


@dataclass
class FrameBundle:
    prices: pd.DataFrame
    returns: pd.DataFrame
    corr: pd.DataFrame
    G: nx.Graph
    pos: Dict[str, Tuple[float, float]]
    curvatures: Dict[Tuple[str, str], float]
    capital_mass: Dict[str, float]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def ensure_dirs(root: Path, clean: bool = False) -> MovieAssets:
    frames = root / "frames"
    output = root / "output"
    stills = root / "stills"
    if clean and root.exists():
        shutil.rmtree(root)
    for p in [frames, output, stills]:
        p.mkdir(parents=True, exist_ok=True)
    return MovieAssets(root=root, frames=frames, output=output, stills=stills)


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = TEXT,
    anchor: str = "mm",
    stroke_width: int = 0,
    stroke_fill: Tuple[int, int, int] = BG,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor, stroke_width=stroke_width, stroke_fill=stroke_fill)


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius=28, fill=PANEL, outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def gradient_background(width: int = W, height: int = H) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(0, 1, height)[:, None]
    radial = np.sqrt((x - 0.72) ** 2 + (y - 0.34) ** 2)
    glow = np.clip(1 - radial * 1.4, 0, 1)
    arr[..., 0] = (BG[0] + 18 * glow + 6 * x).astype(np.uint8)
    arr[..., 1] = (BG[1] + 38 * glow + 8 * y).astype(np.uint8)
    arr[..., 2] = (BG[2] + 80 * glow + 18 * x).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def save_pil(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def ease(t: float) -> float:
    t = np.clip(t, 0, 1)
    return float(t * t * (3 - 2 * t))


def mix_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = np.clip(t, 0, 1)
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def curvature_color(kappa: float) -> Tuple[int, int, int]:
    k = float(np.clip(kappa, -0.6, 0.6))
    if k < 0:
        return mix_color(WHITE, RED, abs(k) / 0.6)
    return mix_color(WHITE, BLUE, k / 0.6)


def simple_beep_audio(duration: float, volume: float = 0.015):
    """Soft synthetic background tone. Use --silent to skip."""
    def make_frame(t):
        # Low, unobtrusive pad with two sine waves.
        return volume * (
            0.55 * np.sin(2 * np.pi * 110 * t) +
            0.25 * np.sin(2 * np.pi * 220 * t) +
            0.20 * np.sin(2 * np.pi * 330 * t)
        )
    try:
        return AudioClip(make_frame, duration=duration, fps=44100)
    except TypeError:
        return AudioClip(lambda t: make_frame(t), duration=duration)


# -----------------------------------------------------------------------------
# Data + graph generation
# -----------------------------------------------------------------------------


def try_project_helper_data(tickers: Sequence[str], n_days: int, seed: int) -> Optional[FrameBundle]:
    """Reuse your helper.py if it is next to this script or on PYTHONPATH."""
    try:
        import helper  # type: ignore
    except Exception:
        return None

    try:
        if hasattr(helper, "make_demo_market_data"):
            prices, volumes, dollar_volume = helper.make_demo_market_data(
                tickers=list(tickers), n_days=n_days, seed=seed
            )
        elif hasattr(helper, "make_demo_prices"):
            prices = helper.make_demo_prices(tickers=list(tickers), n_days=n_days, seed=seed)
            volumes = pd.DataFrame(1_000_000, index=prices.index, columns=prices.columns)
            dollar_volume = prices * volumes
        else:
            return None

        returns = helper.prices_to_returns(prices).dropna(axis=1, how="all")
        # Last rolling window for intro movie.
        window = returns.tail(70)
        if hasattr(helper, "build_frame"):
            fd = helper.build_frame(
                returns=returns,
                start=max(0, len(returns) - 70),
                window_size=min(70, len(returns)),
                graph_mode="knn+bridges",
                k_neighbors=3,
                max_bridges=4,
                min_corr=0.02,
                min_pair_obs=4,
            )
            G = fd.G.copy()
            base = helper.build_base_graph_for_layout([fd], all_nodes=returns.columns)
            pos = helper.compute_stable_layout(base, seed=seed, layout_k=0.42)
            corr = fd.corr
        else:
            corr = window.corr().fillna(0.0)
            G = build_graph_from_corr(corr, tickers=returns.columns)
            pos = nx.spring_layout(G, seed=seed, k=0.45, iterations=500)
            pos = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}

        dollar_volume = dollar_volume.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
        mass = dollar_volume.tail(40).mean().fillna(0.0).to_dict()
        if not mass or max(mass.values()) <= 0:
            mass = {n: 1.0 for n in returns.columns}

        curv = {}
        for u, v, d in G.edges(data=True):
            k = float(d.get("ricciCurvature", 0.0))
            if k == 0.0:
                k = fallback_edge_curvature(G, u, v)
                G[u][v]["ricciCurvature"] = k
            curv[tuple(sorted((str(u), str(v))))] = k

        return FrameBundle(
            prices=prices,
            returns=returns,
            corr=corr,
            G=G,
            pos=pos,
            curvatures=curv,
            capital_mass={str(k): float(v) for k, v in mass.items()},
        )
    except Exception as exc:
        print(f"[warning] helper.py integration failed; using internal synthetic data. Reason: {exc}")
        return None


def synthetic_prices(tickers: Sequence[str], n_days: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = list(tickers)
    factors = rng.normal(0, [0.010, 0.014, 0.019, 0.025], size=(n_days, 4))
    loadings = rng.uniform(-0.25, 1.2, size=(len(tickers), 4))
    # Make themes somewhat coherent.
    for i, t in enumerate(tickers):
        if t in {"NVDA", "AMD", "AVGO", "TSM", "SMCI", "PLTR"}:
            loadings[i] += [0.8, 0.1, 0.1, 0.0]
        elif t in {"MU", "MRVL", "AMAT", "LRCX", "KLAC"}:
            loadings[i] += [0.1, 0.8, 0.1, 0.0]
        elif t in {"ANET", "AAOI", "COHR", "LITE"}:
            loadings[i] += [0.1, 0.0, 0.8, 0.0]
        else:
            loadings[i] += [0.0, 0.0, 0.2, 0.8]
    noise = rng.normal(0, 0.012, size=(n_days, len(tickers)))
    returns = factors @ loadings.T + noise
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame(prices, index=idx, columns=tickers)
    # IPO-aware late starts.
    for symbol, frac in {"QNT": 0.70, "SPCX": 0.76}.items():
        if symbol in df.columns:
            start = int(frac * n_days)
            df.iloc[:start, df.columns.get_loc(symbol)] = np.nan
    return df


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def build_graph_from_corr(corr: pd.DataFrame, tickers: Iterable[str], k: int = 3) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from([str(t) for t in tickers])
    corr = corr.fillna(0.0)
    for u in corr.columns:
        nearest = corr.loc[u].drop(index=u, errors="ignore").sort_values(ascending=False).head(k)
        for v, rho in nearest.items():
            rho = float(rho)
            if rho > 0.02:
                d = float(np.sqrt(2 * (1 - np.clip(rho, -1, 1))))
                G.add_edge(str(u), str(v), correlation=rho, distance=d, weight=d)
    for u, v in list(G.edges()):
        G[u][v]["ricciCurvature"] = fallback_edge_curvature(G, u, v)
    return G


def fallback_edge_curvature(G: nx.Graph, u: str, v: str) -> float:
    common = len(list(nx.common_neighbors(G, u, v)))
    deg_sum = max(1, G.degree(u) + G.degree(v) - 2)
    bridge_penalty = 0.35 if nx.has_path(G, u, v) and common == 0 else 0.0
    return float(np.clip(2 * common / deg_sum - 0.25 - bridge_penalty, -0.6, 0.6))


def build_frame_bundle(tickers: Sequence[str], n_days: int, seed: int, use_helper: bool = True) -> FrameBundle:
    if use_helper:
        bundle = try_project_helper_data(tickers, n_days, seed)
        if bundle is not None:
            return bundle

    prices = synthetic_prices(tickers, n_days=n_days, seed=seed)
    returns = prices_to_returns(prices)
    corr = returns.tail(70).corr().fillna(0.0)
    G = build_graph_from_corr(corr, tickers=returns.columns, k=3)
    pos0 = nx.spring_layout(G, seed=seed, k=0.45, iterations=500)
    pos = {n: (float(p[0]), float(p[1])) for n, p in pos0.items()}
    rng = np.random.default_rng(seed + 1)
    mass = {t: float(rng.lognormal(mean=18.0, sigma=1.0)) for t in G.nodes()}
    curv = {tuple(sorted((u, v))): float(G[u][v].get("ricciCurvature", 0.0)) for u, v in G.edges()}
    return FrameBundle(prices=prices, returns=returns, corr=corr, G=G, pos=pos, curvatures=curv, capital_mass=mass)


# -----------------------------------------------------------------------------
# Drawing primitives
# -----------------------------------------------------------------------------


def normalize_positions(pos: Dict[str, Tuple[float, float]], box: Tuple[int, int, int, int]) -> Dict[str, Tuple[float, float]]:
    x0, y0, x1, y1 = box
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    if len(xs) == 0:
        return {}
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    dx = xmax - xmin if xmax > xmin else 1.0
    dy = ymax - ymin if ymax > ymin else 1.0
    out = {}
    for n, (x, y) in pos.items():
        px = x0 + (x - xmin) / dx * (x1 - x0)
        py = y0 + (1 - (y - ymin) / dy) * (y1 - y0)
        out[n] = (float(px), float(py))
    return out


def draw_network(
    img: Image.Image,
    bundle: FrameBundle,
    box: Tuple[int, int, int, int],
    progress: float = 1.0,
    show_labels: bool = True,
    show_curvature: bool = True,
    show_capital: bool = False,
    flow_phase: float = 0.0,
    z_lift: float = 0.0,
) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    pos = normalize_positions(bundle.pos, box)
    edges = list(bundle.G.edges(data=True))
    n_edges = max(1, int(len(edges) * progress))
    selected_edges = edges[:n_edges]

    # Draw edges.
    for u, v, data in selected_edges:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        kappa = float(data.get("ricciCurvature", 0.0))
        col = curvature_color(kappa) if show_curvature else (100, 160, 210)
        width = 2 + int(6 * min(1, abs(kappa) / 0.6))
        if show_capital:
            flow = math.sqrt(max(bundle.capital_mass.get(str(u), 1.0), 1.0) * max(bundle.capital_mass.get(str(v), 1.0), 1.0))
            mass_values = list(bundle.capital_mass.values()) or [1]
            denom = math.sqrt(max(mass_values) * max(mass_values))
            width += int(7 * min(1, flow / max(denom, 1)))
        draw.line((x0, y0 - z_lift * kappa * 60, x1, y1 - z_lift * kappa * 60), fill=(*col, 185), width=width)

        # Capital particles.
        if show_capital:
            for j in range(2):
                t = (flow_phase + 0.5 * j) % 1.0
                px = x0 * (1 - t) + x1 * t
                py = (y0 - z_lift * kappa * 60) * (1 - t) + (y1 - z_lift * kappa * 60) * t
                r = 5
                draw.ellipse((px - r, py - r, px + r, py + r), fill=(*GOLD, 220))

    # Draw nodes.
    mass_values = np.array(list(bundle.capital_mass.values()) or [1.0], dtype=float)
    mmin, mmax = float(np.nanmin(mass_values)), float(np.nanmax(mass_values))
    font = find_font(24, bold=True)
    small = find_font(18)
    nodes = list(bundle.G.nodes())
    n_nodes = max(1, int(len(nodes) * progress))
    for node in nodes[:n_nodes]:
        if node not in pos:
            continue
        x, y = pos[node]
        deg = bundle.G.degree(node)
        mass = bundle.capital_mass.get(str(node), mmin)
        mt = 0 if mmax <= mmin else (math.log1p(mass) - math.log1p(mmin)) / max(1e-9, math.log1p(mmax) - math.log1p(mmin))
        r = 15 + 2 * deg + (18 * mt if show_capital else 0)
        if str(node) in {"QNT", "SPCX"}:
            fill = (*GRAY, 230)
        else:
            fill = (*mix_color(CYAN, BLUE, min(1, deg / 6)), 235)
        yy = y - z_lift * 30 * (deg / max(1, max(dict(bundle.G.degree()).values())))
        draw.ellipse((x - r, yy - r, x + r, yy + r), fill=fill, outline=(*WHITE, 210), width=2)
        if show_labels:
            draw.text((x, yy - r - 8), str(node), font=small, fill=TEXT, anchor="ms")


def draw_heatmap(img: Image.Image, corr: pd.DataFrame, box: Tuple[int, int, int, int], progress: float = 1.0) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = box
    cols = list(corr.columns)[:18]
    n = len(cols)
    if n == 0:
        return
    size = min(x1 - x0, y1 - y0)
    cell = size / n
    visible_n = max(1, int(n * progress))
    font = find_font(15)
    for i, u in enumerate(cols[:visible_n]):
        for j, v in enumerate(cols[:visible_n]):
            rho = float(corr.loc[u, v]) if u in corr.index and v in corr.columns else 0.0
            if rho >= 0:
                col = mix_color((25, 34, 60), BLUE, min(1, rho))
            else:
                col = mix_color((25, 34, 60), RED, min(1, -rho))
            xx0 = x0 + j * cell
            yy0 = y0 + i * cell
            draw.rectangle((xx0, yy0, xx0 + cell + 1, yy0 + cell + 1), fill=(*col, 230))
    for i, c in enumerate(cols[:visible_n]):
        if i % 2 == 0:
            draw.text((x0 + i * cell + cell / 2, y1 + 12), c, font=font, fill=MUTED, anchor="ma")
            draw.text((x0 - 12, y0 + i * cell + cell / 2), c, font=font, fill=MUTED, anchor="rm")


def draw_flow_pipeline(img: Image.Image, labels: Sequence[str], active: int = -1) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    font = find_font(28, bold=True)
    small = find_font(20)
    x = 230
    y0 = 190
    gap = 82
    for i, label in enumerate(labels):
        y = y0 + i * gap
        active_now = i <= active
        col = CYAN if active_now else MUTED
        fill = (18, 39, 72, 235) if active_now else (15, 24, 45, 185)
        rounded_rect(draw, (x - 170, y - 30, x + 170, y + 30), radius=18, fill=fill, outline=(*col, 180), width=2)
        draw.text((x, y), label, font=small, fill=col, anchor="mm")
        if i < len(labels) - 1:
            arrow_col = (*col, 180) if active_now else (*MUTED, 90)
            draw.line((x, y + 34, x, y + gap - 34), fill=arrow_col, width=3)
            draw.polygon([(x - 8, y + gap - 40), (x + 8, y + gap - 40), (x, y + gap - 24)], fill=arrow_col)


def base_frame(title: str = "", subtitle: str = "") -> Image.Image:
    img = gradient_background()
    draw = ImageDraw.Draw(img, "RGBA")
    # Subtle grid.
    for x in range(0, W, 80):
        draw.line((x, 0, x, H), fill=(255, 255, 255, 12), width=1)
    for y in range(0, H, 80):
        draw.line((0, y, W, y), fill=(255, 255, 255, 10), width=1)
    if title:
        draw.text((80, 70), title, font=find_font(54, bold=True), fill=TEXT, anchor="la")
    if subtitle:
        draw.text((84, 132), subtitle, font=find_font(26), fill=MUTED, anchor="la")
    return img


def add_footer(img: Image.Image, text: str = "Ricci-Curvature-Finance v11 | Dynamic 3D Ricci-Capital Manifold") -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, H - 54, W, H), fill=(0, 0, 0, 80))
    draw.text((W - 60, H - 28), text, font=find_font(22), fill=MUTED, anchor="rm")


# -----------------------------------------------------------------------------
# Scene renderers
# -----------------------------------------------------------------------------


def render_scene_title(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame()
        draw = ImageDraw.Draw(img, "RGBA")
        # Network ghost zoom.
        ghost = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_network(ghost, bundle, (660, 250, 1760, 950), progress=ease(t), show_labels=False, show_curvature=True, z_lift=0.5 * t)
        ghost = ghost.filter(ImageFilter.GaussianBlur(radius=0.5))
        #img.alpha_composite(ghost)

        img = img.convert("RGBA")
        ghost = ghost.convert("RGBA")

        if ghost.size != img.size:
           ghost = ghost.resize(img.size)

        img = Image.alpha_composite(img, ghost)

        title_font = find_font(76, bold=True)
        sub_font = find_font(34)
        alpha = int(255 * ease(t))
        draw.text((100, 330), "Perelman / Ollivier Ricci Flow", font=title_font, fill=(*TEXT, alpha), anchor="la")
        draw.text((104, 420), "Applied to Financial Markets", font=title_font, fill=(*CYAN, alpha), anchor="la")
        draw.text((108, 520), "From market data to dynamic geometric regimes", font=sub_font, fill=(*MUTED, alpha), anchor="la")
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene01" / f"{i:04d}.png"))
    return paths


def render_scene_data(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    prices = bundle.prices.ffill().tail(120)
    cols = list(prices.columns[:8])
    norm = prices[cols] / prices[cols].iloc[0]
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame("1. Load market data", "Prices, volumes, and dollar-volume are transformed into returns.")
        draw = ImageDraw.Draw(img, "RGBA")
        draw_flow_pipeline(img, ["Market Data", "Price Cleaning", "Log Returns", "Dollar Volume"], active=int(t * 4))
        # Chart panel.
        box = (600, 230, 1760, 830)
        rounded_rect(draw, box, radius=32, fill=(11, 19, 38, 220), outline=(*CYAN, 120), width=2)
        x0, y0, x1, y1 = box
        visible = max(5, int(len(norm) * ease(t)))
        sub = norm.iloc[:visible]
        for j, c in enumerate(cols):
            vals = sub[c].fillna(method="ffill").fillna(1.0).to_numpy()
            if len(vals) < 2:
                continue
            mn, mx = np.nanmin(norm[c]), np.nanmax(norm[c])
            if mx <= mn:
                mx = mn + 1
            xs = np.linspace(x0 + 80, x1 - 60, len(vals))
            ys = y1 - 80 - (vals - mn) / (mx - mn) * (y1 - y0 - 150)
            col = [CYAN, BLUE, GREEN, GOLD, RED, (180, 120, 255), (255, 150, 80), (130, 210, 255)][j % 8]
            draw.line(list(zip(xs, ys)), fill=(*col, 210), width=3)
            draw.text((x1 - 45, ys[-1]), c, font=find_font(18, bold=True), fill=(*col, 230), anchor="lm")
        draw.text((x0 + 60, y0 + 45), "Normalized price paths", font=find_font(30, bold=True), fill=TEXT, anchor="la")
        draw.text((x0 + 60, y1 - 35), "returns = log(price_t / price_{t-1})", font=find_font(24), fill=MUTED, anchor="la")
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene02" / f"{i:04d}.png"))
    return paths


def render_scene_correlation(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame("2. Build correlation geometry", "Return co-movement becomes a financial distance matrix.")
        draw = ImageDraw.Draw(img, "RGBA")
        draw_flow_pipeline(img, ["Log Returns", "Correlation Matrix", "Mantegna Distance", "Financial Graph"], active=int(t * 4))
        rounded_rect(draw, (590, 180, 1180, 880), radius=28, fill=(11, 19, 38, 230), outline=(*CYAN, 120), width=2)
        draw_heatmap(img, bundle.corr, (670, 275, 1100, 705), progress=ease(t))
        formula = "dᵢⱼ = √(2(1 − ρᵢⱼ))"
        draw.text((1330, 420), formula, font=find_font(56, bold=True), fill=CYAN, anchor="mm")
        draw.text((1330, 500), "high correlation → short distance", font=find_font(30), fill=TEXT, anchor="mm")
        draw.text((1330, 555), "low correlation → long distance", font=find_font(30), fill=MUTED, anchor="mm")
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene03" / f"{i:04d}.png"))
    return paths


def render_scene_graph(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame("3. Convert the market into a network", "Tickers are nodes; edges are short financial distances.")
        draw = ImageDraw.Draw(img, "RGBA")
        draw_flow_pipeline(img, ["Correlation", "Distance", "kNN + Bridges", "Market Map"], active=int(t * 4))
        draw_network(img, bundle, (570, 185, 1770, 900), progress=ease(t), show_labels=True, show_curvature=False)
        draw.text((650, 900), "Gray late-start nodes can appear before enough price history exists.", font=find_font(24), fill=MUTED, anchor="la")
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene04" / f"{i:04d}.png"))
    return paths


def render_scene_curvature(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame("4. Measure Ricci curvature", "Curvature reveals coherent basins and fragile bridges.")
        draw = ImageDraw.Draw(img, "RGBA")
        draw_network(img, bundle, (520, 190, 1770, 890), progress=1.0, show_labels=True, show_curvature=True)
        # Legend cards.
        rounded_rect(draw, (80, 255, 460, 430), radius=24, fill=(30, 40, 70, 235), outline=(*BLUE, 160), width=3)
        draw.text((110, 300), "Positive κ", font=find_font(34, bold=True), fill=BLUE, anchor="la")
        draw.text((110, 352), "coherent cluster\ncapital basin\nredundant paths", font=find_font(24), fill=TEXT, anchor="la")
        rounded_rect(draw, (80, 485, 460, 660), radius=24, fill=(45, 25, 35, 235), outline=(*RED, 160), width=3)
        draw.text((110, 530), "Negative κ", font=find_font(34, bold=True), fill=RED, anchor="la")
        draw.text((110, 582), "bridge edge\nstress channel\nfragmentation risk", font=find_font(24), fill=TEXT, anchor="la")
        # Pulsing colorbar.
        for j in range(240):
            val = -0.6 + 1.2 * j / 239
            col = curvature_color(val)
            draw.line((W - 120, 280 + j * 2, W - 85, 280 + j * 2), fill=(*col, 255), width=2)
        draw.text((W - 105, 250), "κ", font=find_font(28, bold=True), fill=TEXT, anchor="mm")
        draw.text((W - 75, 280), "+", font=find_font(28, bold=True), fill=BLUE, anchor="lm")
        draw.text((W - 75, 760), "−", font=find_font(28, bold=True), fill=RED, anchor="lm")
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene05" / f"{i:04d}.png"))
    return paths


def render_scene_flow(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    base_pos = dict(bundle.pos)
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame("5. Run Ricci flow", "Positive-curvature links contract; negative-curvature links stretch.")
        draw = ImageDraw.Draw(img, "RGBA")
        draw.text((100, 250), "wₜ₊₁ = wₜ × (1 − step × κₜ)", font=find_font(48, bold=True), fill=CYAN, anchor="la")
        draw.text((100, 325), "flow turns a static graph into a geometric stress test", font=find_font(28), fill=MUTED, anchor="la")
        # Flowed positions: push bridge endpoints apart, pull positive neighbors together approximately.
        pos = {n: np.array(p, dtype=float) for n, p in base_pos.items()}
        center = np.mean(np.array(list(pos.values())), axis=0)
        strength = 0.18 * ease(t)
        for u, v, data in bundle.G.edges(data=True):
            k = float(data.get("ricciCurvature", 0.0))
            if u not in pos or v not in pos:
                continue
            mid = 0.5 * (pos[u] + pos[v])
            if k > 0:
                pos[u] += strength * k * (mid - pos[u])
                pos[v] += strength * k * (mid - pos[v])
            else:
                pos[u] += strength * abs(k) * (pos[u] - mid)
                pos[v] += strength * abs(k) * (pos[v] - mid)
        tmp = FrameBundle(bundle.prices, bundle.returns, bundle.corr, bundle.G, {n: tuple(p) for n, p in pos.items()}, bundle.curvatures, bundle.capital_mass)
        draw_network(img, tmp, (560, 200, 1760, 910), progress=1.0, show_labels=True, show_curvature=True, z_lift=0.2)
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene06" / f"{i:04d}.png"))
    return paths


def render_scene_capital(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame("6. Add capital-flow mass", "Correlation gives shape; dollar-volume gives market mass.")
        draw = ImageDraw.Draw(img, "RGBA")
        draw_flow_pipeline(img, ["Dollar Volume", "Node Mass", "Edge Transport", "Capital Basins"], active=int(t * 4))
        draw_network(img, bundle, (560, 190, 1760, 900), progress=1.0, show_labels=True, show_curvature=True, show_capital=True, flow_phase=t, z_lift=0.45)
        draw.text((650, 910), "node size = dollar-volume mass   |   edge width + particles = capital transport", font=find_font(25), fill=GOLD, anchor="la")
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene07" / f"{i:04d}.png"))
    return paths


def render_scene_hmm(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    states = ["risk-on", "rotation", "stress", "transition", "risk-on", "rotation", "stress"]
    state_colors = [GREEN, GOLD, RED, CYAN, GREEN, GOLD, RED]
    features = ["avg Ricci", "Ricci σ", "clusters", "density", "capital flow", "future return"]
    rng = np.random.default_rng(123)
    matrix = rng.normal(size=(len(states), len(features)))
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame("7. Infer hidden market regimes", "Rolling Ricci + capital features feed a Gaussian HMM.")
        draw = ImageDraw.Draw(img, "RGBA")
        # Feature matrix.
        x0, y0 = 110, 250
        cell_w, cell_h = 120, 54
        visible_rows = max(1, int(len(states) * ease(t)))
        draw.text((x0, y0 - 55), "Feature matrix", font=find_font(32, bold=True), fill=TEXT, anchor="la")
        for r in range(visible_rows):
            for c in range(len(features)):
                val = matrix[r, c]
                col = BLUE if val > 0 else RED
                alpha = int(80 + 130 * min(1, abs(val) / 2))
                draw.rectangle((x0 + c * cell_w, y0 + r * cell_h, x0 + (c + 1) * cell_w - 4, y0 + (r + 1) * cell_h - 4), fill=(*col, alpha))
        for c, f in enumerate(features):
            draw.text((x0 + c * cell_w + 55, y0 + visible_rows * cell_h + 15), f, font=find_font(17), fill=MUTED, anchor="ma")
        # Arrow to HMM.
        draw.line((910, 455, 1110, 455), fill=(*CYAN, 220), width=5)
        draw.polygon([(1110, 455), (1086, 440), (1086, 470)], fill=(*CYAN, 220))
        rounded_rect(draw, (1150, 340, 1490, 570), radius=42, fill=(13, 28, 58, 240), outline=(*CYAN, 190), width=3)
        draw.text((1320, 420), "Gaussian\nHMM", font=find_font(46, bold=True), fill=CYAN, anchor="mm")
        # Timeline.
        tx0, ty = 1080, 720
        draw.text((tx0, ty - 85), "Hidden-state timeline", font=find_font(32, bold=True), fill=TEXT, anchor="la")
        for r in range(visible_rows):
            x = tx0 + r * 110
            draw.ellipse((x - 28, ty - 28, x + 28, ty + 28), fill=(*state_colors[r], 230), outline=(*WHITE, 180), width=2)
            draw.text((x, ty + 52), states[r], font=find_font(18, bold=True), fill=state_colors[r], anchor="mm")
            if r < visible_rows - 1:
                draw.line((x + 30, ty, x + 80, ty), fill=(*MUTED, 180), width=3)
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene08" / f"{i:04d}.png"))
    return paths


def render_scene_3d(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    base = {n: np.array(p, dtype=float) for n, p in bundle.pos.items()}
    for i in range(n):
        t = i / max(1, n - 1)
        angle = 2 * math.pi * t
        pos3 = {}
        for node, p in base.items():
            deg = bundle.G.degree(node)
            z = 0.15 * deg + 0.08 * math.sin(angle + len(str(node)))
            x = p[0] * math.cos(angle) - z * math.sin(angle)
            y = p[1]
            pos3[node] = (x, y)
        tmp = FrameBundle(bundle.prices, bundle.returns, bundle.corr, bundle.G, pos3, bundle.curvatures, bundle.capital_mass)
        img = base_frame("8. Dynamic 3D Ricci-capital manifold", "Rolling windows show market geometry evolving through time.")
        draw = ImageDraw.Draw(img, "RGBA")
        draw_network(img, tmp, (260, 190, 1700, 900), progress=1.0, show_labels=True, show_curvature=True, show_capital=True, flow_phase=t, z_lift=1.0)
        draw.text((1320, 230), "x/y = topology", font=find_font(28, bold=True), fill=TEXT, anchor="la")
        draw.text((1320, 275), "z = Ricci stress", font=find_font(28, bold=True), fill=CYAN, anchor="la")
        draw.text((1320, 320), "size = capital mass", font=find_font(28, bold=True), fill=GOLD, anchor="la")
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene09" / f"{i:04d}.png"))
    return paths


def render_scene_conclusion(bundle: FrameBundle, assets: MovieAssets, seconds: float) -> List[Path]:
    paths = []
    n = int(seconds * FPS)
    labels = ["Market Data", "Geometry", "Curvature", "Flow", "Regimes", "Investment Insight"]
    for i in range(n):
        t = i / max(1, n - 1)
        img = base_frame()
        draw = ImageDraw.Draw(img, "RGBA")
        draw_network(img, bundle, (1130, 210, 1760, 830), progress=1.0, show_labels=False, show_curvature=True, show_capital=True, flow_phase=t, z_lift=0.75)
        title_alpha = int(255 * ease(t))
        draw.text((120, 220), "Financial Geometry", font=find_font(78, bold=True), fill=(*CYAN, title_alpha), anchor="la")
        draw.text((120, 315), "instead of only financial indicators", font=find_font(42), fill=(*TEXT, title_alpha), anchor="la")
        x0, y = 150, 520
        for j, lab in enumerate(labels):
            x = x0 + j * 260
            active = ease(t) > j / len(labels)
            col = CYAN if active else MUTED
            rounded_rect(draw, (x - 90, y - 45, x + 90, y + 45), radius=22, fill=(13, 28, 58, 230), outline=(*col, 160), width=2)
            draw.text((x, y), lab, font=find_font(23, bold=True), fill=col, anchor="mm")
            if j < len(labels) - 1:
                draw.line((x + 96, y, x + 164, y), fill=(*col, 170), width=4)
                draw.polygon([(x + 164, y), (x + 145, y - 12), (x + 145, y + 12)], fill=(*col, 170))
        draw.text((120, 760), "Goal: detect structural market evolution before it becomes obvious in price alone.", font=find_font(34), fill=TEXT, anchor="la")
        add_footer(img)
        paths.append(save_pil(img.convert("RGB"), assets.frames / "scene10" / f"{i:04d}.png"))
    return paths


# -----------------------------------------------------------------------------
# Movie assembly
# -----------------------------------------------------------------------------


def frames_to_clip(paths: Sequence[Path], fps: int = FPS):
    return ImageSequenceClip([str(p) for p in paths], fps=fps)


def add_fade(clip, fade: float = 0.35):
    try:
        return clip.fx(vfx.fadein, fade).fx(vfx.fadeout, fade)
    except Exception:
        # MoviePy v2 sometimes exposes effects differently; safe fallback.
        return clip


def build_movie(
    assets: MovieAssets,
    bundle: FrameBundle,
    out_path: Path,
    fps: int = FPS,
    silent: bool = False,
    fast: bool = False,
) -> Path:
    if fast:
        durations = {
            "title": 2.5, "data": 3.0, "corr": 3.0, "graph": 3.0, "curv": 3.0,
            "flow": 3.0, "capital": 3.0, "hmm": 3.0, "manifold": 3.0, "end": 2.5,
        }
    else:
        durations = {
            "title": 5.0, "data": 7.0, "corr": 7.0, "graph": 7.0, "curv": 8.0,
            "flow": 8.0, "capital": 8.0, "hmm": 8.0, "manifold": 9.0, "end": 6.0,
        }

    scene_paths: List[List[Path]] = []
    scene_paths.append(render_scene_title(bundle, assets, durations["title"]))
    scene_paths.append(render_scene_data(bundle, assets, durations["data"]))
    scene_paths.append(render_scene_correlation(bundle, assets, durations["corr"]))
    scene_paths.append(render_scene_graph(bundle, assets, durations["graph"]))
    scene_paths.append(render_scene_curvature(bundle, assets, durations["curv"]))
    scene_paths.append(render_scene_flow(bundle, assets, durations["flow"]))
    scene_paths.append(render_scene_capital(bundle, assets, durations["capital"]))
    scene_paths.append(render_scene_hmm(bundle, assets, durations["hmm"]))
    scene_paths.append(render_scene_3d(bundle, assets, durations["manifold"]))
    scene_paths.append(render_scene_conclusion(bundle, assets, durations["end"]))

    clips = [add_fade(frames_to_clip(paths, fps=fps), 0.3) for paths in scene_paths if paths]
    final = concatenate_videoclips(clips, method="compose")

    if not silent:
        try:
            final = final.set_audio(simple_beep_audio(final.duration))
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    codec = "libx264"
    audio_codec = "aac" if not silent else None
    final.write_videofile(
        str(out_path),
        fps=fps,
        codec=codec,
        audio_codec=audio_codec,
        preset="medium",
        bitrate="6000k",
        threads=max(1, os.cpu_count() or 1),
    )
    try:
        final.close()
        for c in clips:
            c.close()
    except Exception:
        pass
    return out_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Ricci Finance lecture intro MP4 with MoviePy.")
    parser.add_argument("--out", default="output/ricci_finance_intro.mp4", help="Output MP4 path.")
    parser.add_argument("--workdir", default="movie_build", help="Temporary/build directory for PNG frames.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated ticker list.")
    parser.add_argument("--n-days", type=int, default=260, help="Synthetic/history days.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument("--clean", action="store_true", help="Clean workdir before rendering.")
    parser.add_argument("--no-helper", action="store_true", help="Do not import project helper.py; use internal synthetic data.")
    parser.add_argument("--silent", action="store_true", help="Do not add synthetic background audio.")
    parser.add_argument("--fast", action="store_true", help="Render a shorter preview movie.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global FPS
    args = parse_args(argv)
    FPS = int(args.fps)

    tickers = [t.strip().upper() for t in args.tickers.replace("\n", ",").split(",") if t.strip()]
    if len(tickers) < 2:
        raise ValueError("Please provide at least two tickers.")

    assets = ensure_dirs(Path(args.workdir), clean=args.clean)
    print("[1/3] Building Ricci Finance intro data...")
    bundle = build_frame_bundle(tickers, n_days=int(args.n_days), seed=int(args.seed), use_helper=not args.no_helper)

    print("[2/3] Rendering animation frames...")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    print("[3/3] Assembling MP4 with MoviePy...")
    build_movie(assets=assets, bundle=bundle, out_path=out_path, fps=FPS, silent=bool(args.silent), fast=bool(args.fast))
    print(f"Done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
