#!/usr/bin/env bash
set -euo pipefail
python -m pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-intel keras POT GraphRicciCurvature || true
python -m pip install -r requirements.txt
