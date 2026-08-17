from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence
import re


THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AI": ("artificial intelligence", "machine learning", "deep learning", "accelerated computing", "generative ai"),
    "DataCenter": ("data center", "datacenter", "hyperscale", "server", "accelerator"),
    "Networking": ("networking", "ethernet", "switching", "interconnect", "router", "network infrastructure"),
    "Semiconductors": ("semiconductor", "integrated circuit", "chip", "processor", "gpu", "cpu"),
    "Memory": ("memory", "dram", "nand", "flash storage", "high bandwidth memory", "hbm"),
    "Equipment": ("wafer fabrication equipment", "semiconductor equipment", "process control", "lithography", "etch", "deposition"),
    "Photonics": ("photonics", "optical", "laser", "transceiver", "fiber optic", "optoelectronic"),
    "QuantumComputing": ("quantum computing", "quantum computer", "qubit", "quantum processor", "quantum annealing"),
    "Cloud": ("cloud computing", "cloud platform", "cloud service", "software as a service", "saas"),
    "Internet": ("social media", "internet platform", "online advertising", "digital advertising", "search engine"),
    "ConsumerElectronics": ("consumer electronics", "smartphone", "personal computer", "wearable", "tablet"),
    "Services": ("subscription service", "digital services", "payment service", "app store", "services segment"),
    "Automotive": ("automotive", "autonomous driving", "vehicle", "adas"),
    "Robotics": ("robotics", "robot", "industrial automation"),
    "Space": ("spacecraft", "satellite", "launch vehicle", "space systems", "aerospace"),
    "Defense": ("defense", "military", "national security"),
}


@dataclass(frozen=True)
class SectorProfile:
    """Normalized, multi-label sector information for one ticker."""

    ticker: str
    official_sector: str = "Other"
    official_industry: str = "Other"
    memberships: dict[str, float] = field(default_factory=lambda: {"Other": 1.0})
    source: str = "fallback"
    description: str = ""

    @property
    def primary_sector(self) -> str:
        return max(self.memberships, key=self.memberships.get)

    def validate(self, tolerance: float = 1e-8) -> None:
        if not self.memberships:
            raise ValueError(f"{self.ticker}: memberships must not be empty")
        if any(float(value) < 0 for value in self.memberships.values()):
            raise ValueError(f"{self.ticker}: memberships must be non-negative")
        total = sum(float(value) for value in self.memberships.values())
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"{self.ticker}: membership sum is {total}, expected 1.0")


def normalize_memberships(
    values: Mapping[str, float] | str | None,
    *,
    fallback: str = "Other",
) -> dict[str, float]:
    """Normalize non-negative memberships so each ticker contributes exactly 1."""
    if values is None:
        return {fallback: 1.0}
    if isinstance(values, str):
        label = values.strip() or fallback
        return {label: 1.0}

    cleaned: dict[str, float] = {}
    for raw_label, raw_value in values.items():
        label = str(raw_label).strip()
        value = float(raw_value)
        if not label or value <= 0:
            continue
        cleaned[label] = cleaned.get(label, 0.0) + value

    total = sum(cleaned.values())
    if total <= 0:
        return {fallback: 1.0}
    return {label: value / total for label, value in cleaned.items()}


def _keyword_scores(description: str) -> dict[str, float]:
    text = re.sub(r"\s+", " ", (description or "").lower())
    scores: dict[str, float] = {}
    for theme, phrases in THEME_KEYWORDS.items():
        score = 0.0
        for phrase in phrases:
            count = text.count(phrase.lower())
            if count:
                # Multi-word phrases are more specific and receive a modest boost.
                score += count * (1.0 + 0.15 * max(len(phrase.split()) - 1, 0))
        if score > 0:
            scores[theme] = score
    return scores


def build_profile(
    ticker: str,
    *,
    official_sector: str | None = None,
    official_industry: str | None = None,
    description: str = "",
    manual_memberships: Mapping[str, float] | str | None = None,
    official_weight: float = 0.45,
    industry_weight: float = 0.25,
    theme_weight: float = 0.30,
) -> SectorProfile:
    """Create a normalized profile by blending official labels and inferred themes."""
    ticker = str(ticker).upper().strip()
    sector = (official_sector or "Other").strip() or "Other"
    industry = (official_industry or "Other").strip() or "Other"

    if manual_memberships is not None:
        memberships = normalize_memberships(manual_memberships)
        source = "manual"
    else:
        raw: dict[str, float] = {}
        if sector != "Other":
            raw[sector] = raw.get(sector, 0.0) + max(float(official_weight), 0.0)
        if industry != "Other":
            raw[industry] = raw.get(industry, 0.0) + max(float(industry_weight), 0.0)

        themes = normalize_memberships(_keyword_scores(description), fallback="Other")
        if themes != {"Other": 1.0}:
            for label, value in themes.items():
                raw[label] = raw.get(label, 0.0) + max(float(theme_weight), 0.0) * value

        memberships = normalize_memberships(raw)
        source = "automatic" if raw else "fallback"

    profile = SectorProfile(
        ticker=ticker,
        official_sector=sector,
        official_industry=industry,
        memberships=memberships,
        source=source,
        description=description or "",
    )
    profile.validate()
    return profile


def fetch_yfinance_profiles(
    tickers: Sequence[str],
    *,
    overrides: Mapping[str, Mapping[str, float] | str] | None = None,
) -> dict[str, SectorProfile]:
    """Fetch Yahoo metadata and infer normalized multi-sector profiles.

    Network or metadata failures degrade to overrides or ``Other`` rather than
    aborting the complete V16 pipeline.
    """
    import yfinance as yf

    overrides = {str(k).upper(): v for k, v in (overrides or {}).items()}
    output: dict[str, SectorProfile] = {}
    for raw_ticker in tickers:
        ticker = str(raw_ticker).upper().strip()
        override = overrides.get(ticker)
        try:
            obj = yf.Ticker(ticker)
            try:
                info = obj.get_info()
            except Exception:
                info = obj.info
            output[ticker] = build_profile(
                ticker,
                official_sector=info.get("sector"),
                official_industry=info.get("industry"),
                description=info.get("longBusinessSummary", ""),
                manual_memberships=override,
            )
        except Exception:
            output[ticker] = build_profile(
                ticker,
                manual_memberships=override,
            )
    return output


def memberships_map(profiles: Mapping[str, SectorProfile]) -> dict[str, dict[str, float]]:
    return {str(ticker): dict(profile.memberships) for ticker, profile in profiles.items()}


def primary_sector_map(profiles: Mapping[str, SectorProfile]) -> dict[str, str]:
    return {str(ticker): profile.primary_sector for ticker, profile in profiles.items()}
