from __future__ import annotations

import importlib.util
import sys

REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "networkx",
    "yfinance",
    "hmmlearn",
    "torch",
    "streamlit",
    "matplotlib",
]
FORBIDDEN = ["tensorflow", "ot", "GraphRicciCurvature"]

print("Python:", sys.version)
print("\nRequired packages")
missing = []
for name in REQUIRED:
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"  {name}: MISSING")
        missing.append(name)
    else:
        module = __import__(name)
        print(f"  {name}: OK {getattr(module, '__version__', '')}")

print("\nPackages intentionally not used by V15.3")
for name in FORBIDDEN:
    installed = importlib.util.find_spec(name) is not None
    print(f"  {name}: {'installed but unused' if installed else 'not installed'}")

if missing:
    raise SystemExit("Missing required packages: " + ", ".join(missing))

from ricci_finance.graph import compute_ollivier_ricci_lp
print("\nSciPy-LP Ollivier implementation: import OK")
