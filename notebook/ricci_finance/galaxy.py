from __future__ import annotations

from .visualization import galaxy_figure, galaxy_positions

# Backward-compatible name retained for notebooks and earlier V15 callers.
stable_galaxy_positions = galaxy_positions

__all__ = ["galaxy_figure", "galaxy_positions", "stable_galaxy_positions"]
