"""v12 compatibility facade.

Existing v11 imports can continue using ``from helper import ...`` while code is
migrated to focused modules under ``ricci_finance``.
"""
from ricci_finance import *  # noqa: F401,F403
